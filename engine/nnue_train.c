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

/* Forward pass storing intermediates for backprop. */
static float
forward_with_intermediates(const uint16_t stm_indices[], int num_stm,
			   const uint16_t nstm_indices[], int num_nstm,
			   float stm_accum_raw[NNUE_ACCUM_DIM],
			   float nstm_accum_raw[NNUE_ACCUM_DIM],
			   float concat[2 * NNUE_ACCUM_DIM],
			   float fc1_raw[NNUE_FC1_OUT_DIM],
			   float crelu_out[NNUE_HIDDEN1_DIM],
			   float sqrcrelu_out[NNUE_HIDDEN1_DIM],
			   float fc2_in[NNUE_FC2_IN_DIM],
			   float h2_raw[NNUE_HIDDEN2_DIM],
			   float h2_act[NNUE_HIDDEN2_DIM],
			   float *skip_out)
{
  int i, j, k;
  float output;

  /* Compute STM accumulator from sparse features */
  for (j = 0; j < NNUE_ACCUM_DIM; j++)
    stm_accum_raw[j] = nnue_weights.l0_bias[j];
  for (k = 0; k < num_stm; k++) {
    int idx = stm_indices[k];
    for (j = 0; j < NNUE_ACCUM_DIM; j++)
      stm_accum_raw[j] += nnue_weights.l0_weight[idx][j];
  }

  /* Compute NSTM accumulator from sparse features */
  for (j = 0; j < NNUE_ACCUM_DIM; j++)
    nstm_accum_raw[j] = nnue_weights.l0_bias[j];
  for (k = 0; k < num_nstm; k++) {
    int idx = nstm_indices[k];
    for (j = 0; j < NNUE_ACCUM_DIM; j++)
      nstm_accum_raw[j] += nnue_weights.l0_weight[idx][j];
  }

  /* Clipped ReLU + concatenate accumulators */
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[i] = crelu(stm_accum_raw[i]);
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[NNUE_ACCUM_DIM + i] = crelu(nstm_accum_raw[i]);

  /* FC1: concat[256] -> fc1_raw[33] (32 hidden + 1 skip) */
  for (j = 0; j < NNUE_FC1_OUT_DIM; j++) {
    float sum = nnue_weights.l1_bias[j];
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
      sum += concat[i] * nnue_weights.l1_weight[i][j];
    fc1_raw[j] = sum;
  }

  /* Extract skip connection */
  *skip_out = fc1_raw[NNUE_HIDDEN1_DIM];

  /* CReLU and SqrCReLU on the 32 hidden outputs, concatenate into fc2_in[64] */
  for (i = 0; i < NNUE_HIDDEN1_DIM; i++) {
    float cr = crelu(fc1_raw[i]);
    crelu_out[i] = cr;
    sqrcrelu_out[i] = cr * cr;
    fc2_in[i] = crelu_out[i];
    fc2_in[NNUE_HIDDEN1_DIM + i] = sqrcrelu_out[i];
  }

  /* FC2: fc2_in[64] -> hidden2[32] */
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++) {
    float sum = nnue_weights.l2_bias[j];
    for (i = 0; i < NNUE_FC2_IN_DIM; i++)
      sum += fc2_in[i] * nnue_weights.l2_weight[i][j];
    h2_raw[j] = sum;
    h2_act[j] = crelu(sum);
  }

  /* FC3: hidden2 -> output + skip -> tanh */
  output = nnue_weights.l3_bias + *skip_out;
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    output += h2_act[i] * nnue_weights.l3_weight[i];
  return tanhf(output);
}

float
nnue_backward(const NNUETrainSample *sample, NNUEGradients *grad,
	      float lambda)
{
  float stm_accum_raw[NNUE_ACCUM_DIM];
  float nstm_accum_raw[NNUE_ACCUM_DIM];
  float concat[2 * NNUE_ACCUM_DIM];
  float fc1_raw[NNUE_FC1_OUT_DIM];
  float crelu_out[NNUE_HIDDEN1_DIM], sqrcrelu_out[NNUE_HIDDEN1_DIM];
  float fc2_in[NNUE_FC2_IN_DIM];
  float h2_raw[NNUE_HIDDEN2_DIM], h2_act[NNUE_HIDDEN2_DIM];
  float skip_val;
  float output, error, loss, d_output;
  float d_h2[NNUE_HIDDEN2_DIM];
  float d_fc2_in[NNUE_FC2_IN_DIM];
  float d_fc1[NNUE_FC1_OUT_DIM];
  float d_concat[2 * NNUE_ACCUM_DIM];
  float target;
  int i, j, k;

  /* Forward pass with sparse features */
  output = forward_with_intermediates(sample->stm_indices, sample->num_stm,
				      sample->nstm_indices, sample->num_nstm,
				      stm_accum_raw, nstm_accum_raw,
				      concat, fc1_raw, crelu_out,
				      sqrcrelu_out, fc2_in,
				      h2_raw, h2_act, &skip_val);

  /* Blended target: lambda*search_score + (1-lambda)*game_result */
  target = lambda * sample->search_score
    + (1.0f - lambda) * sample->game_result;
  error = output - target;
  loss = error * error;

  /* d_loss/d_output = 2 * error */
  /* d_output/d_pre_tanh = 1 - tanh^2 = 1 - output^2 */
  d_output = 2.0f * error * (1.0f - output * output);

  /* Layer 3 gradients (output = l3_bias + skip + sum(h2_act * l3_weight)) */
  grad->l3_bias += d_output;
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    grad->l3_weight[i] += d_output * h2_act[i];

  /* Backprop through layer 3 to hidden2 */
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    d_h2[i] = d_output * nnue_weights.l3_weight[i] * crelu_deriv(h2_raw[i]);

  /* Layer 2 gradients: fc2_in[64] -> hidden2[32] */
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++) {
    grad->l2_bias[j] += d_h2[j];
    for (i = 0; i < NNUE_FC2_IN_DIM; i++)
      grad->l2_weight[i][j] += d_h2[j] * fc2_in[i];
  }

  /* Backprop through layer 2 to fc2_in */
  for (i = 0; i < NNUE_FC2_IN_DIM; i++) {
    float sum = 0.0f;
    for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
      sum += d_h2[j] * nnue_weights.l2_weight[i][j];
    d_fc2_in[i] = sum;
  }

  /* Backprop through CReLU and SqrCReLU to fc1_raw[0..31] */
  for (i = 0; i < NNUE_HIDDEN1_DIM; i++) {
    /* CReLU path: d_fc2_in[i] * crelu_deriv(fc1_raw[i]) */
    float d_crelu = d_fc2_in[i] * crelu_deriv(fc1_raw[i]);
    /* SqrCReLU path: d/dx(crelu(x)^2) = 2*crelu(x)*crelu_deriv(x) */
    float d_sqr = d_fc2_in[NNUE_HIDDEN1_DIM + i]
      * 2.0f * crelu_out[i] * crelu_deriv(fc1_raw[i]);
    d_fc1[i] = d_crelu + d_sqr;
  }

  /* Skip connection gradient: d_output flows directly to fc1_raw[32] */
  d_fc1[NNUE_HIDDEN1_DIM] = d_output;

  /* Layer 1 gradients: concat[256] -> fc1_raw[33] */
  for (j = 0; j < NNUE_FC1_OUT_DIM; j++) {
    grad->l1_bias[j] += d_fc1[j];
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
      grad->l1_weight[i][j] += d_fc1[j] * concat[i];
  }

  /* Backprop through layer 1 to concat */
  for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++) {
    float sum = 0.0f;
    for (j = 0; j < NNUE_FC1_OUT_DIM; j++)
      sum += d_fc1[j] * nnue_weights.l1_weight[i][j];
    d_concat[i] = sum;
  }

  /* Backprop through clipped ReLU to accumulators */
  /* STM accumulator: indices [0, NNUE_ACCUM_DIM) in concat */
  for (j = 0; j < NNUE_ACCUM_DIM; j++) {
    float d_accum = d_concat[j] * crelu_deriv(stm_accum_raw[j]);
    grad->l0_bias[j] += d_accum;
    for (k = 0; k < sample->num_stm; k++)
      grad->l0_weight[sample->stm_indices[k]][j] += d_accum;
  }

  /* NSTM accumulator: indices [NNUE_ACCUM_DIM, 2*NNUE_ACCUM_DIM) */
  for (j = 0; j < NNUE_ACCUM_DIM; j++) {
    float d_accum = d_concat[NNUE_ACCUM_DIM + j]
      * crelu_deriv(nstm_accum_raw[j]);
    grad->l0_bias[j] += d_accum;
    for (k = 0; k < sample->num_nstm; k++)
      grad->l0_weight[sample->nstm_indices[k]][j] += d_accum;
  }

  return loss;
}

/* ---- Adam optimizer ---- */

void
nnue_train_config_defaults(NNUETrainConfig *config)
{
  config->lr_max = 0.001f;
  config->lr_min = 0.00001f;
  config->beta1 = 0.9f;
  config->beta2 = 0.999f;
  config->epsilon = 1e-8f;
  config->lambda = 0.75f;
  config->batch_size = 1024;
  config->num_epochs = 10;
  config->total_steps = 0;
}

void
nnue_adam_init(NNUEAdamState *state)
{
  memset(&state->m, 0, sizeof(NNUEGradients));
  memset(&state->v, 0, sizeof(NNUEGradients));
  state->t = 0;
}

/* Apply Adam to a flat array of parameters.
 * b1c/b2c are precomputed bias corrections: 1 - beta^t.
 */
static void
adam_update_array(float *param, const float *grad, float *m, float *v,
		  int count, float lr, float beta1, float beta2,
		  float epsilon, float b1c, float b2c)
{
  int i;
  float m_hat, v_hat;
  for (i = 0; i < count; i++) {
    m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
    m_hat = m[i] / b1c;
    v_hat = v[i] / b2c;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
  }
}

void
nnue_adam_update(NNUEGradients *grad, NNUEAdamState *state,
		 const NNUETrainConfig *config, float lr)
{
  float b1 = config->beta1;
  float b2 = config->beta2;
  float eps = config->epsilon;

  float b1c, b2c;

  state->t++;
  b1c = 1.0f - powf(b1, (float)state->t);
  b2c = 1.0f - powf(b2, (float)state->t);

  adam_update_array((float *)nnue_weights.l0_weight,
		    (const float *)grad->l0_weight,
		    (float *)state->m.l0_weight,
		    (float *)state->v.l0_weight,
		    NNUE_INPUT_DIM * NNUE_ACCUM_DIM,
		    lr, b1, b2, eps, b1c, b2c);

  adam_update_array(nnue_weights.l0_bias, grad->l0_bias,
		    state->m.l0_bias, state->v.l0_bias,
		    NNUE_ACCUM_DIM,
		    lr, b1, b2, eps, b1c, b2c);

  adam_update_array((float *)nnue_weights.l1_weight,
		    (const float *)grad->l1_weight,
		    (float *)state->m.l1_weight,
		    (float *)state->v.l1_weight,
		    2 * NNUE_ACCUM_DIM * NNUE_FC1_OUT_DIM,
		    lr, b1, b2, eps, b1c, b2c);

  adam_update_array(nnue_weights.l1_bias, grad->l1_bias,
		    state->m.l1_bias, state->v.l1_bias,
		    NNUE_FC1_OUT_DIM,
		    lr, b1, b2, eps, b1c, b2c);

  adam_update_array((float *)nnue_weights.l2_weight,
		    (const float *)grad->l2_weight,
		    (float *)state->m.l2_weight,
		    (float *)state->v.l2_weight,
		    NNUE_FC2_IN_DIM * NNUE_HIDDEN2_DIM,
		    lr, b1, b2, eps, b1c, b2c);

  adam_update_array(nnue_weights.l2_bias, grad->l2_bias,
		    state->m.l2_bias, state->v.l2_bias,
		    NNUE_HIDDEN2_DIM,
		    lr, b1, b2, eps, b1c, b2c);

  adam_update_array(nnue_weights.l3_weight, grad->l3_weight,
		    state->m.l3_weight, state->v.l3_weight,
		    NNUE_HIDDEN2_DIM,
		    lr, b1, b2, eps, b1c, b2c);

  adam_update_array(&nnue_weights.l3_bias, &grad->l3_bias,
		    &state->m.l3_bias, &state->v.l3_bias,
		    1,
		    lr, b1, b2, eps, b1c, b2c);
}

float
nnue_cosine_lr(const NNUETrainConfig *config, int step)
{
  float progress;
  if (config->total_steps <= 0)
    return config->lr_max;
  progress = (float)step / (float)config->total_steps;
  if (progress > 1.0f) progress = 1.0f;
  return config->lr_min + 0.5f * (config->lr_max - config->lr_min)
    * (1.0f + cosf((float)M_PI * progress));
}

/* Extract sparse feature indices for both perspectives from current board state */
static void
extract_sparse_features(uint16_t stm_indices[], uint8_t *num_stm,
			uint16_t nstm_indices[], uint8_t *num_nstm,
			int color, int previous_pass)
{
  int i, j, pos, pt_idx, base;
  int own = color;
  int opp = OTHER_COLOR(color);
  int unconditional_own[BOARDMAX];
  int unconditional_opp[BOARDMAX];
  int ns = 0, nn = 0;

  unconditional_life(unconditional_own, own);
  unconditional_life(unconditional_opp, opp);

  for (i = 0; i < NNUE_BOARD_SIZE; i++) {
    for (j = 0; j < NNUE_BOARD_SIZE; j++) {
      pos = POS(i, j);
      pt_idx = i * NNUE_BOARD_SIZE + j;
      base = pt_idx * NNUE_FEAT_PER_PT;

      if (board[pos] == own) {
	stm_indices[ns++] = (uint16_t)(base + 0);
	nstm_indices[nn++] = (uint16_t)(base + 1);
      }
      else if (board[pos] == opp) {
	stm_indices[ns++] = (uint16_t)(base + 1);
	nstm_indices[nn++] = (uint16_t)(base + 0);
      }

      if (board_ko_pos == pos) {
	stm_indices[ns++] = (uint16_t)(base + 2);
	/* Ko is from STM perspective only */
      }

      if (IS_STONE(board[pos])) {
	int libs = countlib(pos);
	int lib_feat = -1;
	if (libs == 1) lib_feat = 3;
	else if (libs == 2) lib_feat = 4;
	else if (libs == 3) lib_feat = 5;

	if (lib_feat >= 0) {
	  stm_indices[ns++] = (uint16_t)(base + lib_feat);
	  nstm_indices[nn++] = (uint16_t)(base + lib_feat);
	}
      }

      if (unconditional_own[pos]) {
	stm_indices[ns++] = (uint16_t)(base + 6);
	nstm_indices[nn++] = (uint16_t)(base + 7);
      }
      if (unconditional_opp[pos]) {
	stm_indices[ns++] = (uint16_t)(base + 7);
	nstm_indices[nn++] = (uint16_t)(base + 6);
      }
    }
  }

  if (previous_pass) {
    uint16_t global_idx = (uint16_t)(NNUE_NUM_POINTS * NNUE_FEAT_PER_PT);
    stm_indices[ns++] = global_idx;
    nstm_indices[nn++] = global_idx;
  }

  *num_stm = (uint8_t)ns;
  *num_nstm = (uint8_t)nn;
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

/* Self-play that labels positions with deep search scores AND game outcome.
 * Plays the game to completion, then backfills game_result for all positions.
 */
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
  /* Track STM color for each collected sample, for game_result backfill */
  int sample_stm_colors[200];
  int start_idx = current_samples;
  float final_score;
  int k;

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

      extract_sparse_features(s->stm_indices, &s->num_stm,
			      s->nstm_indices, &s->num_nstm,
			      color, previous_pass);

      /* Label with deep search */
      deep_score = alphabeta_eval_position(color, deep_limit);
      s->search_score = deep_score;
      s->game_result = 0.0f;  /* Backfilled below */
      sample_stm_colors[added] = color;
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

  /* Game complete — compute Tromp-Taylor score and backfill game_result.
   * nnue_score_position returns White score. Convert to STM perspective.
   */
  final_score = nnue_score_position(komi);
  for (k = 0; k < added; k++) {
    float result;
    if (final_score > 0.0f)
      result = 1.0f;   /* White wins */
    else if (final_score < 0.0f)
      result = -1.0f;  /* Black wins */
    else
      result = 0.0f;   /* Draw */

    /* Convert from White's perspective to STM perspective */
    if (sample_stm_colors[k] == BLACK)
      result = -result;
    samples[start_idx + k].game_result = result;
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
	       int deep_node_limit, const char *weights_file,
	       const NNUETrainConfig *config)
{
  NNUETrainSample *samples;
  NNUEGradients *grad;
  NNUEAdamState *adam;
  NNUETrainConfig cfg;
  int gen, g, epoch;
  int opening_moves = 6;  /* Random opening book length */

  /* Use provided config or defaults */
  if (config) {
    cfg = *config;
  }
  else {
    nnue_train_config_defaults(&cfg);
  }

  samples = (NNUETrainSample *)malloc(NNUE_MAX_SAMPLES * sizeof(NNUETrainSample));
  grad = (NNUEGradients *)malloc(sizeof(NNUEGradients));
  adam = (NNUEAdamState *)malloc(sizeof(NNUEAdamState));

  if (!samples || !grad || !adam) {
    fprintf(stderr, "nnue_train: out of memory\n");
    if (samples) free(samples);
    if (grad) free(grad);
    if (adam) free(adam);
    return;
  }

  nnue_adam_init(adam);

  /* Try to load existing weights, otherwise init random */
  if (!nnue_load(weights_file)) {
    fprintf(stderr, "No existing weights found, initializing randomly.\n");
    nnue_init_random(42);
    nnue_quantize_weights();
  }

  /* Set rules for training: Tromp-Taylor */
  ko_rule = PSK;
  suicide_rule = FORBIDDEN;

  for (gen = 0; gen < generations; gen++) {
    int total_samples = 0;
    int global_step = 0;
    float win_rate;

    fprintf(stderr, "=== Generation %d/%d (Adam, batch=%d, lambda=%.2f) ===\n",
	    gen + 1, generations, cfg.batch_size, cfg.lambda);

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

    /* Compute total steps for cosine annealing schedule */
    cfg.total_steps = (total_samples / cfg.batch_size) * cfg.num_epochs;

    /* Phase 2: Train NNUE with Adam + cosine LR + blended loss */
    for (epoch = 0; epoch < cfg.num_epochs; epoch++) {
      float epoch_loss = 0.0f;
      int epoch_batches = 0;
      int s;

      shuffle_samples(samples, total_samples);

      for (s = 0; s + cfg.batch_size <= total_samples; s += cfg.batch_size) {
	int b;
	float batch_loss = 0.0f;
	float lr;

	nnue_grad_zero(grad);

	for (b = 0; b < cfg.batch_size; b++)
	  batch_loss += nnue_backward(&samples[s + b], grad, cfg.lambda);

	/* Scale gradients by batch size */
	{
	  float scale = 1.0f / (float)cfg.batch_size;
	  int n = (int)sizeof(NNUEGradients) / (int)sizeof(float);
	  float *gp = (float *)grad;
	  int gi;
	  for (gi = 0; gi < n; gi++)
	    gp[gi] *= scale;
	}

	/* Cosine-annealed learning rate */
	lr = nnue_cosine_lr(&cfg, global_step);
	nnue_adam_update(grad, adam, &cfg, lr);

	epoch_loss += batch_loss / cfg.batch_size;
	epoch_batches++;
	global_step++;
      }

      if (epoch_batches > 0)
	fprintf(stderr, "  Epoch %d/%d: avg_loss=%.6f lr=%.6f\n",
		epoch + 1, cfg.num_epochs,
		epoch_loss / epoch_batches,
		nnue_cosine_lr(&cfg, global_step));
    }

    /* Quantize weights for inference and save */
    nnue_quantize_weights();
    nnue_save(weights_file);
    fprintf(stderr, "Weights saved to %s\n", weights_file);

    /* Evaluate against random play */
    win_rate = evaluate_vs_random(200, node_limit);
    fprintf(stderr, "Win rate vs random (200 games): %.1f%%\n",
	    win_rate * 100.0f);
  }

  free(samples);
  free(grad);
  free(adam);
}

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
