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
#include "liberty.h"

/* Global NNUE state */
NNUEWeights nnue_weights;
NNUEAccumPair nnue_accum_stack[NNUE_MAX_STACK];
int nnue_accum_sp = 0;

/* Simple xorshift32 PRNG for weight initialization */
static unsigned int xorshift_state;

static unsigned int
xorshift32(void)
{
  unsigned int x = xorshift_state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  xorshift_state = x;
  return x;
}

/* Return a random float in [-scale, +scale] using He initialization */
static float
random_float(float scale)
{
  unsigned int r = xorshift32();
  return scale * (2.0f * ((float)r / 4294967295.0f) - 1.0f);
}

void
nnue_init_random(unsigned int seed)
{
  int i, j;
  float scale;

  xorshift_state = seed ? seed : 12345;

  /* He initialization: scale = sqrt(2/fan_in) */

  /* Layer 0: fan_in = NNUE_INPUT_DIM (649), but inputs are sparse binary.
   * Use a moderate scale since few inputs are active at once. */
  scale = sqrtf(2.0f / 20.0f);  /* ~20 active features typical */
  for (i = 0; i < NNUE_INPUT_DIM; i++)
    for (j = 0; j < NNUE_ACCUM_DIM; j++)
      nnue_weights.l0_weight[i][j] = random_float(scale);
  for (j = 0; j < NNUE_ACCUM_DIM; j++)
    nnue_weights.l0_bias[j] = 0.0f;

  /* Layer 1: fan_in = 256 */
  scale = sqrtf(2.0f / (2 * NNUE_ACCUM_DIM));
  for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
    for (j = 0; j < NNUE_HIDDEN1_DIM; j++)
      nnue_weights.l1_weight[i][j] = random_float(scale);
  for (j = 0; j < NNUE_HIDDEN1_DIM; j++)
    nnue_weights.l1_bias[j] = 0.0f;

  /* Layer 2: fan_in = 32 */
  scale = sqrtf(2.0f / NNUE_HIDDEN2_DIM);
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
      nnue_weights.l2_weight[i][j] = random_float(scale);
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
    nnue_weights.l2_bias[j] = 0.0f;

  /* Layer 3: fan_in = 32 */
  scale = sqrtf(2.0f / NNUE_HIDDEN2_DIM);
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    nnue_weights.l3_weight[i] = random_float(scale);
  nnue_weights.l3_bias = 0.0f;

  nnue_accum_sp = 0;
}

int
nnue_load(const char *filename)
{
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return 0;
  if (fread(&nnue_weights, sizeof(NNUEWeights), 1, fp) != 1) {
    fclose(fp);
    return 0;
  }
  fclose(fp);
  nnue_accum_sp = 0;
  return 1;
}

int
nnue_save(const char *filename)
{
  FILE *fp = fopen(filename, "wb");
  if (!fp)
    return 0;
  if (fwrite(&nnue_weights, sizeof(NNUEWeights), 1, fp) != 1) {
    fclose(fp);
    return 0;
  }
  fclose(fp);
  return 1;
}

/* Extract 649 binary features for one perspective.
 * perspective = the color whose "own" stones are feature 0.
 * features[] must be NNUE_INPUT_DIM floats, zeroed by caller.
 */
static void
extract_features(float features[NNUE_INPUT_DIM], int perspective,
		 int previous_move_was_pass)
{
  int i, j, pos, pt_idx;
  int own = perspective;
  int opp = OTHER_COLOR(perspective);
  int unconditional_own[BOARDMAX];
  int unconditional_opp[BOARDMAX];

  /* Compute pass-alive territories using Benson's algorithm */
  unconditional_life(unconditional_own, own);
  unconditional_life(unconditional_opp, opp);

  for (i = 0; i < NNUE_BOARD_SIZE; i++) {
    for (j = 0; j < NNUE_BOARD_SIZE; j++) {
      pos = POS(i, j);
      pt_idx = i * NNUE_BOARD_SIZE + j;

      /* Feature 0: own stone */
      if (board[pos] == own)
	features[pt_idx * NNUE_FEAT_PER_PT + 0] = 1.0f;

      /* Feature 1: opponent stone */
      if (board[pos] == opp)
	features[pt_idx * NNUE_FEAT_PER_PT + 1] = 1.0f;

      /* Feature 2: ko point (forbidden for STM = perspective) */
      if (board_ko_pos == pos)
	features[pt_idx * NNUE_FEAT_PER_PT + 2] = 1.0f;

      /* Features 3-5: liberty counts (for stones on this point) */
      if (IS_STONE(board[pos])) {
	int libs = countlib(pos);
	if (libs == 1)
	  features[pt_idx * NNUE_FEAT_PER_PT + 3] = 1.0f;
	else if (libs == 2)
	  features[pt_idx * NNUE_FEAT_PER_PT + 4] = 1.0f;
	else if (libs == 3)
	  features[pt_idx * NNUE_FEAT_PER_PT + 5] = 1.0f;
      }

      /* Feature 6: own pass-alive area */
      if (unconditional_own[pos])
	features[pt_idx * NNUE_FEAT_PER_PT + 6] = 1.0f;

      /* Feature 7: opponent pass-alive area */
      if (unconditional_opp[pos])
	features[pt_idx * NNUE_FEAT_PER_PT + 7] = 1.0f;
    }
  }

  /* Global feature 648: pass will end game */
  if (previous_move_was_pass)
    features[NNUE_NUM_POINTS * NNUE_FEAT_PER_PT] = 1.0f;
}

/* Compute accumulator from features: accum = bias + sum(active_feature * weight_row) */
static void
compute_accumulator(NNUEAccumulator *accum, const float features[NNUE_INPUT_DIM])
{
  int i, j;

  /* Start with bias */
  for (j = 0; j < NNUE_ACCUM_DIM; j++)
    accum->values[j] = nnue_weights.l0_bias[j];

  /* Add weight rows for active features */
  for (i = 0; i < NNUE_INPUT_DIM; i++) {
    if (features[i] != 0.0f) {
      for (j = 0; j < NNUE_ACCUM_DIM; j++)
	accum->values[j] += features[i] * nnue_weights.l0_weight[i][j];
    }
  }
}

/* Detect if the previous move was a pass by checking move history.
 * Only valid at root level (stackp == 0).
 */
int
detect_previous_pass(void)
{
  if (move_history_pointer > 0
      && move_history_pos[move_history_pointer - 1] == PASS_MOVE)
    return 1;
  return 0;
}

void
nnue_accumulator_refresh(int color, int previous_pass)
{
  float features[NNUE_INPUT_DIM];
  NNUEAccumPair *pair;
  UNUSED(color);

  pair = &nnue_accum_stack[nnue_accum_sp];

  /* Compute White's perspective accumulator */
  memset(features, 0, sizeof(features));
  extract_features(features, WHITE, previous_pass);
  compute_accumulator(&pair->white_accum, features);

  /* Compute Black's perspective accumulator */
  memset(features, 0, sizeof(features));
  extract_features(features, BLACK, previous_pass);
  compute_accumulator(&pair->black_accum, features);
}

void
nnue_accum_push(void)
{
  if (nnue_accum_sp + 1 >= NNUE_MAX_STACK) {
    fprintf(stderr, "nnue: accumulator stack overflow\n");
    return;
  }
  /* Copy current top to next position */
  nnue_accum_stack[nnue_accum_sp + 1] = nnue_accum_stack[nnue_accum_sp];
  nnue_accum_sp++;
}

void
nnue_accum_pop(void)
{
  if (nnue_accum_sp <= 0) {
    fprintf(stderr, "nnue: accumulator stack underflow\n");
    return;
  }
  nnue_accum_sp--;
}

/* Clipped ReLU: max(0, min(1, x)) */
static float
clipped_relu(float x)
{
  if (x <= 0.0f) return 0.0f;
  if (x >= 1.0f) return 1.0f;
  return x;
}

float
nnue_evaluate(int color)
{
  NNUEAccumPair *pair = &nnue_accum_stack[nnue_accum_sp];
  NNUEAccumulator *stm_accum, *nstm_accum;
  float concat[2 * NNUE_ACCUM_DIM];
  float hidden1[NNUE_HIDDEN1_DIM];
  float hidden2[NNUE_HIDDEN2_DIM];
  float output;
  int i, j;

  /* Select accumulators based on side to move */
  if (color == WHITE) {
    stm_accum = &pair->white_accum;
    nstm_accum = &pair->black_accum;
  }
  else {
    stm_accum = &pair->black_accum;
    nstm_accum = &pair->white_accum;
  }

  /* Apply clipped ReLU to accumulators and concatenate */
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[i] = clipped_relu(stm_accum->values[i]);
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[NNUE_ACCUM_DIM + i] = clipped_relu(nstm_accum->values[i]);

  /* Layer 1: concat -> hidden1 */
  for (j = 0; j < NNUE_HIDDEN1_DIM; j++) {
    float sum = nnue_weights.l1_bias[j];
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
      sum += concat[i] * nnue_weights.l1_weight[i][j];
    hidden1[j] = clipped_relu(sum);
  }

  /* Layer 2: hidden1 -> hidden2 */
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++) {
    float sum = nnue_weights.l2_bias[j];
    for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
      sum += hidden1[i] * nnue_weights.l2_weight[i][j];
    hidden2[j] = clipped_relu(sum);
  }

  /* Layer 3: hidden2 -> output (tanh) */
  output = nnue_weights.l3_bias;
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    output += hidden2[i] * nnue_weights.l3_weight[i];
  output = tanhf(output);

  return output;
}

/* Tromp-Taylor area scoring.
 * Returns score from White's perspective.
 */
float
nnue_score_position(float komi)
{
  int pos, i, j;
  int owner[BOARDMAX];
  int queue[MAX_BOARD * MAX_BOARD];
  int head, tail;
  int white_score = 0;
  int black_score = 0;

  memset(owner, 0, sizeof(owner));

  /* Count stones */
  for (i = 0; i < board_size; i++) {
    for (j = 0; j < board_size; j++) {
      pos = POS(i, j);
      if (board[pos] == WHITE) {
	white_score++;
	owner[pos] = WHITE;
      }
      else if (board[pos] == BLACK) {
	black_score++;
	owner[pos] = BLACK;
      }
    }
  }

  /* Flood-fill from each empty point to determine territory */
  for (i = 0; i < board_size; i++) {
    for (j = 0; j < board_size; j++) {
      pos = POS(i, j);
      if (board[pos] != EMPTY || owner[pos] != 0)
	continue;

      /* BFS to find connected empty region and its border colors */
      head = 0;
      tail = 0;
      queue[tail++] = pos;
      owner[pos] = -1;  /* mark as visited */

      int border_white = 0;
      int border_black = 0;
      int region_size = 0;
      int region[MAX_BOARD * MAX_BOARD];

      while (head < tail) {
	int cur = queue[head++];
	region[region_size++] = cur;

	/* Check all 4 neighbors */
	int neighbors[4];
	int num_neighbors = 0;
	int ci = I(cur);
	int cj = J(cur);

	if (ci > 0)            neighbors[num_neighbors++] = cur - (MAX_BOARD + 1);
	if (ci < board_size-1) neighbors[num_neighbors++] = cur + (MAX_BOARD + 1);
	if (cj > 0)            neighbors[num_neighbors++] = cur - 1;
	if (cj < board_size-1) neighbors[num_neighbors++] = cur + 1;

	int n;
	for (n = 0; n < num_neighbors; n++) {
	  int nb = neighbors[n];
	  if (board[nb] == WHITE)
	    border_white = 1;
	  else if (board[nb] == BLACK)
	    border_black = 1;
	  else if (board[nb] == EMPTY && owner[nb] == 0) {
	    owner[nb] = -1;  /* mark visited */
	    queue[tail++] = nb;
	  }
	}
      }

      /* Assign territory if bordered by exactly one color */
      if (border_white && !border_black) {
	int k;
	for (k = 0; k < region_size; k++)
	  owner[region[k]] = WHITE;
	white_score += region_size;
      }
      else if (border_black && !border_white) {
	int k;
	for (k = 0; k < region_size; k++)
	  owner[region[k]] = BLACK;
	black_score += region_size;
      }
      /* If bordered by both colors, it's dame (neutral) */
    }
  }

  return (float)(white_score - black_score) + komi;
}

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
