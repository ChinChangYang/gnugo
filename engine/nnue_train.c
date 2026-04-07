/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 * This is GNU Go, a Go program. Contact gnugo@gnu.org, or see       *
 * http://www.gnu.org/software/gnugo/ for more information.          *
 *                                                                   *
 * Copyright 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,   *
 * 2008, 2009, 2010 and 2011 by the Free Software Foundation.        *
 *                                                                   *
 * This program is free software; you can redistribute it and/or     *
 * modify it under the terms of the GNU General Public License as    *
 * published by the Free Software Foundation - version 3 or          *
 * (at your option) any later version.                               *
 *                                                                   *
 * This program is distributed in the hope that it will be useful,   *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of    *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     *
 * GNU General Public License in file COPYING for more details.      *
 *                                                                   *
 * You should have received a copy of the GNU General Public         *
 * License along with this program; if not, write to the Free        *
 * Software Foundation, Inc., 51 Franklin Street, Fifth Floor,       *
 * Boston, MA 02111, USA.                                            *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "gnugo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nnue.h"
#include "nnue_train.h"
#include "alphabeta.h"
#include "liberty.h"

/* Simple xorshift for shuffling */
static unsigned int train_rng_state = 54321;

static unsigned int
train_xorshift(void)
{
  unsigned int x = train_rng_state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  train_rng_state = x;
  return x;
}

/* Clipped ReLU and its derivative */
static float
crelu(float x)
{
  if (x <= 0.0f) return 0.0f;
  if (x >= 1.0f) return 1.0f;
  return x;
}

static float
crelu_deriv(float x)
{
  return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
}

void
nnue_grad_zero(NNUEGradients *grad)
{
  memset(grad, 0, sizeof(NNUEGradients));
}

/* Forward pass that also stores intermediates for backprop */
static float
forward_with_intermediates(const float stm_features[NNUE_INPUT_DIM],
			   const float nstm_features[NNUE_INPUT_DIM],
			   float stm_accum_raw[NNUE_ACCUM_DIM],
			   float nstm_accum_raw[NNUE_ACCUM_DIM],
			   float concat[2 * NNUE_ACCUM_DIM],
			   float h1_raw[NNUE_HIDDEN1_DIM],
			   float h1_act[NNUE_HIDDEN1_DIM],
			   float h2_raw[NNUE_HIDDEN2_DIM],
			   float h2_act[NNUE_HIDDEN2_DIM])
{
  int i, j;
  float output;

  /* Compute STM accumulator */
  for (j = 0; j < NNUE_ACCUM_DIM; j++) {
    float sum = nnue_weights.l0_bias[j];
    for (i = 0; i < NNUE_INPUT_DIM; i++)
      if (stm_features[i] != 0.0f)
	sum += stm_features[i] * nnue_weights.l0_weight[i][j];
    stm_accum_raw[j] = sum;
  }

  /* Compute NSTM accumulator */
  for (j = 0; j < NNUE_ACCUM_DIM; j++) {
    float sum = nnue_weights.l0_bias[j];
    for (i = 0; i < NNUE_INPUT_DIM; i++)
      if (nstm_features[i] != 0.0f)
	sum += nstm_features[i] * nnue_weights.l0_weight[i][j];
    nstm_accum_raw[j] = sum;
  }

  /* Clipped ReLU + concatenate */
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[i] = crelu(stm_accum_raw[i]);
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[NNUE_ACCUM_DIM + i] = crelu(nstm_accum_raw[i]);

  /* Layer 1 */
  for (j = 0; j < NNUE_HIDDEN1_DIM; j++) {
    float sum = nnue_weights.l1_bias[j];
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
      sum += concat[i] * nnue_weights.l1_weight[i][j];
    h1_raw[j] = sum;
    h1_act[j] = crelu(sum);
  }

  /* Layer 2 */
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++) {
    float sum = nnue_weights.l2_bias[j];
    for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
      sum += h1_act[i] * nnue_weights.l2_weight[i][j];
    h2_raw[j] = sum;
    h2_act[j] = crelu(sum);
  }

  /* Layer 3: tanh output */
  output = nnue_weights.l3_bias;
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    output += h2_act[i] * nnue_weights.l3_weight[i];
  return tanhf(output);
}

float
nnue_backward(const NNUETrainSample *sample, NNUEGradients *grad)
{
  float stm_accum_raw[NNUE_ACCUM_DIM];
  float nstm_accum_raw[NNUE_ACCUM_DIM];
  float concat[2 * NNUE_ACCUM_DIM];
  float h1_raw[NNUE_HIDDEN1_DIM], h1_act[NNUE_HIDDEN1_DIM];
  float h2_raw[NNUE_HIDDEN2_DIM], h2_act[NNUE_HIDDEN2_DIM];
  float output, error, d_output;
  float d_h2[NNUE_HIDDEN2_DIM];
  float d_h1[NNUE_HIDDEN1_DIM];
  float d_concat[2 * NNUE_ACCUM_DIM];
  int i, j;

  /* Forward pass */
  output = forward_with_intermediates(sample->stm_features,
				      sample->nstm_features,
				      stm_accum_raw, nstm_accum_raw,
				      concat, h1_raw, h1_act,
				      h2_raw, h2_act);

  /* MSE loss: L = (output - target)^2 */
  error = output - sample->target;
  float loss = error * error;

  /* d_loss/d_output = 2 * error */
  /* d_output/d_pre_tanh = 1 - tanh^2 = 1 - output^2 */
  d_output = 2.0f * error * (1.0f - output * output);

  /* Layer 3 gradients */
  grad->l3_bias += d_output;
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    grad->l3_weight[i] += d_output * h2_act[i];

  /* Backprop through layer 3 to hidden2 */
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    d_h2[i] = d_output * nnue_weights.l3_weight[i] * crelu_deriv(h2_raw[i]);

  /* Layer 2 gradients */
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++) {
    grad->l2_bias[j] += d_h2[j];
    for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
      grad->l2_weight[i][j] += d_h2[j] * h1_act[i];
  }

  /* Backprop through layer 2 to hidden1 */
  for (i = 0; i < NNUE_HIDDEN1_DIM; i++) {
    float sum = 0.0f;
    for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
      sum += d_h2[j] * nnue_weights.l2_weight[i][j];
    d_h1[i] = sum * crelu_deriv(h1_raw[i]);
  }

  /* Layer 1 gradients */
  for (j = 0; j < NNUE_HIDDEN1_DIM; j++) {
    grad->l1_bias[j] += d_h1[j];
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
      grad->l1_weight[i][j] += d_h1[j] * concat[i];
  }

  /* Backprop through layer 1 to concat */
  for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++) {
    float sum = 0.0f;
    for (j = 0; j < NNUE_HIDDEN1_DIM; j++)
      sum += d_h1[j] * nnue_weights.l1_weight[i][j];
    d_concat[i] = sum;
  }

  /* Backprop through clipped ReLU to accumulators */
  /* STM accumulator: indices [0, NNUE_ACCUM_DIM) in concat */
  for (j = 0; j < NNUE_ACCUM_DIM; j++) {
    float d_accum = d_concat[j] * crelu_deriv(stm_accum_raw[j]);
    grad->l0_bias[j] += d_accum;
    for (i = 0; i < NNUE_INPUT_DIM; i++)
      if (sample->stm_features[i] != 0.0f)
	grad->l0_weight[i][j] += d_accum * sample->stm_features[i];
  }

  /* NSTM accumulator: indices [NNUE_ACCUM_DIM, 2*NNUE_ACCUM_DIM) */
  for (j = 0; j < NNUE_ACCUM_DIM; j++) {
    float d_accum = d_concat[NNUE_ACCUM_DIM + j]
      * crelu_deriv(nstm_accum_raw[j]);
    grad->l0_bias[j] += d_accum;
    for (i = 0; i < NNUE_INPUT_DIM; i++)
      if (sample->nstm_features[i] != 0.0f)
	grad->l0_weight[i][j] += d_accum * sample->nstm_features[i];
  }

  return loss;
}

void
nnue_sgd_update(NNUEGradients *grad, float lr, int batch_size)
{
  int i, j;
  float scale = lr / (float)batch_size;

  for (i = 0; i < NNUE_INPUT_DIM; i++)
    for (j = 0; j < NNUE_ACCUM_DIM; j++)
      nnue_weights.l0_weight[i][j] -= scale * grad->l0_weight[i][j];
  for (j = 0; j < NNUE_ACCUM_DIM; j++)
    nnue_weights.l0_bias[j] -= scale * grad->l0_bias[j];

  for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
    for (j = 0; j < NNUE_HIDDEN1_DIM; j++)
      nnue_weights.l1_weight[i][j] -= scale * grad->l1_weight[i][j];
  for (j = 0; j < NNUE_HIDDEN1_DIM; j++)
    nnue_weights.l1_bias[j] -= scale * grad->l1_bias[j];

  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
      nnue_weights.l2_weight[i][j] -= scale * grad->l2_weight[i][j];
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
    nnue_weights.l2_bias[j] -= scale * grad->l2_bias[j];

  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    nnue_weights.l3_weight[i] -= scale * grad->l3_weight[i];
  nnue_weights.l3_bias -= scale * grad->l3_bias;
}

/* Extract features for both perspectives from current board state */
static void
extract_both_features(float stm_features[NNUE_INPUT_DIM],
		      float nstm_features[NNUE_INPUT_DIM],
		      int color, int previous_pass)
{
  int i, j, pos, pt_idx;
  int own = color;
  int opp = OTHER_COLOR(color);
  int unconditional_own[BOARDMAX];
  int unconditional_opp[BOARDMAX];

  memset(stm_features, 0, NNUE_INPUT_DIM * sizeof(float));
  memset(nstm_features, 0, NNUE_INPUT_DIM * sizeof(float));

  unconditional_life(unconditional_own, own);
  unconditional_life(unconditional_opp, opp);

  for (i = 0; i < NNUE_BOARD_SIZE; i++) {
    for (j = 0; j < NNUE_BOARD_SIZE; j++) {
      pos = POS(i, j);
      pt_idx = i * NNUE_BOARD_SIZE + j;

      int base = pt_idx * NNUE_FEAT_PER_PT;

      if (board[pos] == own) {
	stm_features[base + 0] = 1.0f;
	nstm_features[base + 1] = 1.0f;
      }
      else if (board[pos] == opp) {
	stm_features[base + 1] = 1.0f;
	nstm_features[base + 0] = 1.0f;
      }

      if (board_ko_pos == pos) {
	stm_features[base + 2] = 1.0f;
	/* Ko is from STM perspective only */
      }

      if (IS_STONE(board[pos])) {
	int libs = countlib(pos);
	int lib_feat = -1;
	if (libs == 1) lib_feat = 3;
	else if (libs == 2) lib_feat = 4;
	else if (libs == 3) lib_feat = 5;

	if (lib_feat >= 0) {
	  stm_features[base + lib_feat] = 1.0f;
	  nstm_features[base + lib_feat] = 1.0f;
	}
      }

      if (unconditional_own[pos]) {
	stm_features[base + 6] = 1.0f;
	nstm_features[base + 7] = 1.0f;
      }
      if (unconditional_opp[pos]) {
	stm_features[base + 7] = 1.0f;
	nstm_features[base + 6] = 1.0f;
      }
    }
  }

  if (previous_pass) {
    stm_features[NNUE_NUM_POINTS * NNUE_FEAT_PER_PT] = 1.0f;
    nstm_features[NNUE_NUM_POINTS * NNUE_FEAT_PER_PT] = 1.0f;
  }
}

/* Play a random legal move for position diversity (opening book) */
static int
play_random_move(int color)
{
  int legal_moves[MAX_BOARD * MAX_BOARD];
  int num_legal = 0;
  int i, j, pos;

  for (i = 0; i < board_size; i++) {
    for (j = 0; j < board_size; j++) {
      pos = POS(i, j);
      if (board[pos] == EMPTY && is_allowed_move(pos, color))
	legal_moves[num_legal++] = pos;
    }
  }

  if (num_legal == 0)
    return PASS_MOVE;

  return legal_moves[train_xorshift() % num_legal];
}

/* Play one self-play game, collecting positions.
 * Returns number of samples added.
 */
static int
play_selfplay_game(NNUETrainSample *samples, int max_samples,
		   int current_samples, int node_limit,
		   int opening_moves)
{
  int color = BLACK;
  int consecutive_passes = 0;
  int move_num = 0;
  int added = 0;
  int previous_pass = 0;

  gnugo_clear_board(NNUE_BOARD_SIZE);

  while (consecutive_passes < 2 && move_num < 200) {
    int move;

    if (move_num < opening_moves) {
      /* Random opening for diversity */
      move = play_random_move(color);
    }
    else {
      /* Use alpha-beta search with current NNUE */
      move = alphabeta_genmove(color, node_limit);
    }

    /* Collect position before playing the move */
    if (current_samples + added < max_samples && move_num >= opening_moves) {
      NNUETrainSample *s = &samples[current_samples + added];
      extract_both_features(s->stm_features, s->nstm_features,
			    color, previous_pass);
      s->target = 0.0f;  /* Will be filled by deep search later */
      added++;
    }

    /* Play the move */
    if (move == PASS_MOVE) {
      consecutive_passes++;
      previous_pass = 1;
    }
    else {
      consecutive_passes = 0;
      previous_pass = 0;
      gnugo_play_move(move, color);
    }

    color = OTHER_COLOR(color);
    move_num++;
  }

  return added;
}

/* Label collected samples by running deep search on each position.
 * This requires replaying the game from scratch, so we store the
 * board state in the features and reconstruct from them.
 * Actually, since we stored features, we can just run the forward
 * pass with the features. But for deep search labeling, we need the
 * actual board state.
 *
 * Simpler approach: label a random subset of samples using deep search
 * on the current board. Since we want to be fast, we'll label positions
 * during collection by reconstructing board state.
 *
 * Even simpler: collect positions during self-play AND immediately
 * run deep search to get labels. This is what we do.
 */

/* Modified self-play that labels positions with deep search scores */
static int
play_and_label_game(NNUETrainSample *samples, int max_samples,
		    int current_samples, int shallow_limit,
		    int deep_limit, int opening_moves)
{
  int color = BLACK;
  int consecutive_passes = 0;
  int move_num = 0;
  int added = 0;
  int previous_pass = 0;

  gnugo_clear_board(NNUE_BOARD_SIZE);

  while (consecutive_passes < 2 && move_num < 200) {
    int move;

    if (move_num < opening_moves) {
      move = play_random_move(color);
    }
    else {
      move = alphabeta_genmove(color, shallow_limit);
    }

    /* Collect and label position before playing the move */
    if (current_samples + added < max_samples && move_num >= opening_moves) {
      NNUETrainSample *s = &samples[current_samples + added];
      float deep_score;

      extract_both_features(s->stm_features, s->nstm_features,
			    color, previous_pass);

      /* Label with deep search */
      deep_score = alphabeta_eval_position(color, deep_limit);
      s->target = deep_score;
      added++;
    }

    if (move == PASS_MOVE) {
      consecutive_passes++;
      previous_pass = 1;
    }
    else {
      consecutive_passes = 0;
      previous_pass = 0;
      gnugo_play_move(move, color);
    }

    color = OTHER_COLOR(color);
    move_num++;
  }

  return added;
}

/* Shuffle samples using Fisher-Yates */
static void
shuffle_samples(NNUETrainSample *samples, int n)
{
  int i;
  for (i = n - 1; i > 0; i--) {
    int j = train_xorshift() % (i + 1);
    NNUETrainSample tmp = samples[i];
    samples[i] = samples[j];
    samples[j] = tmp;
  }
}

/* Play games against random to measure strength */
static float
evaluate_vs_random(int num_games, int node_limit)
{
  int wins = 0;
  int g;

  for (g = 0; g < num_games; g++) {
    int color = BLACK;
    int consecutive_passes = 0;
    int move_num = 0;
    int nnue_color = (g % 2 == 0) ? BLACK : WHITE;
    float score;

    gnugo_clear_board(NNUE_BOARD_SIZE);

    while (consecutive_passes < 2 && move_num < 200) {
      int move;

      if (color == nnue_color) {
	move = alphabeta_genmove(color, node_limit);
      }
      else {
	move = play_random_move(color);
      }

      if (move == PASS_MOVE)
	consecutive_passes++;
      else {
	consecutive_passes = 0;
	gnugo_play_move(move, color);
      }

      color = OTHER_COLOR(color);
      move_num++;
    }

    score = nnue_score_position(komi);
    if ((nnue_color == WHITE && score > 0)
	|| (nnue_color == BLACK && score < 0))
      wins++;
  }

  return (float)wins / (float)num_games;
}

void
nnue_train_run(int generations, int games_per_gen, int node_limit,
	       int deep_node_limit, const char *weights_file)
{
  NNUETrainSample *samples;
  NNUEGradients *grad;
  int gen, g, epoch;
  float lr = 0.01f;
  int batch_size = 64;
  int num_epochs = 5;
  int opening_moves = 6;  /* Random opening book length */

  samples = (NNUETrainSample *)malloc(NNUE_MAX_SAMPLES * sizeof(NNUETrainSample));
  grad = (NNUEGradients *)malloc(sizeof(NNUEGradients));

  if (!samples || !grad) {
    fprintf(stderr, "nnue_train: out of memory\n");
    if (samples) free(samples);
    if (grad) free(grad);
    return;
  }

  /* Try to load existing weights, otherwise init random */
  if (!nnue_load(weights_file)) {
    fprintf(stderr, "No existing weights found, initializing randomly.\n");
    nnue_init_random(42);
  }

  /* Set rules for training: Tromp-Taylor */
  ko_rule = PSK;
  suicide_rule = FORBIDDEN;

  for (gen = 0; gen < generations; gen++) {
    int total_samples = 0;
    float total_loss = 0.0f;
    int total_batches = 0;
    float win_rate;

    fprintf(stderr, "=== Generation %d/%d (lr=%.5f) ===\n",
	    gen + 1, generations, lr);

    /* Phase 1: Generate positions and label with deep search */
    fprintf(stderr, "Generating and labeling positions from %d games...\n",
	    games_per_gen);

    for (g = 0; g < games_per_gen && total_samples < NNUE_MAX_SAMPLES; g++) {
      int added = play_and_label_game(samples, NNUE_MAX_SAMPLES,
				      total_samples, node_limit,
				      deep_node_limit, opening_moves);
      total_samples += added;

      if ((g + 1) % 1000 == 0)
	fprintf(stderr, "  %d/%d games, %d samples\n",
		g + 1, games_per_gen, total_samples);
    }

    fprintf(stderr, "Collected %d training samples.\n", total_samples);

    if (total_samples == 0) {
      fprintf(stderr, "No samples collected, skipping training.\n");
      continue;
    }

    /* Phase 2: Train NNUE on collected samples */
    for (epoch = 0; epoch < num_epochs; epoch++) {
      float epoch_loss = 0.0f;
      int epoch_batches = 0;
      int s;

      shuffle_samples(samples, total_samples);

      for (s = 0; s + batch_size <= total_samples; s += batch_size) {
	int b;
	float batch_loss = 0.0f;

	nnue_grad_zero(grad);

	for (b = 0; b < batch_size; b++)
	  batch_loss += nnue_backward(&samples[s + b], grad);

	nnue_sgd_update(grad, lr, batch_size);

	epoch_loss += batch_loss / batch_size;
	epoch_batches++;
      }

      if (epoch_batches > 0)
	fprintf(stderr, "  Epoch %d/%d: avg_loss=%.6f\n",
		epoch + 1, num_epochs, epoch_loss / epoch_batches);

      total_loss += epoch_loss;
      total_batches += epoch_batches;
    }

    /* Save weights after each generation */
    nnue_save(weights_file);
    fprintf(stderr, "Weights saved to %s\n", weights_file);

    /* Evaluate against random play */
    win_rate = evaluate_vs_random(50, node_limit);
    fprintf(stderr, "Win rate vs random (50 games): %.1f%%\n",
	    win_rate * 100.0f);

    /* Decay learning rate */
    lr *= 0.95f;
  }

  free(samples);
  free(grad);
}

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
