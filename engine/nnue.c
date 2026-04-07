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

/* Global NNUE state — float (for training) */
NNUEWeights nnue_weights;
NNUEAccumPair nnue_accum_stack[NNUE_MAX_STACK];
int nnue_accum_sp = 0;

/* Global NNUE state — quantized (for search) */
NNUEQuantizedWeights nnue_qweights;
NNUEQuantizedAccumPair nnue_qaccum_stack[NNUE_MAX_STACK];
int nnue_qaccum_sp = 0;

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

  /* Layer 1: fan_in = 256, outputs 33 (32 hidden + 1 skip) */
  scale = sqrtf(2.0f / (2 * NNUE_ACCUM_DIM));
  for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
    for (j = 0; j < NNUE_FC1_OUT_DIM; j++)
      nnue_weights.l1_weight[i][j] = random_float(scale);
  for (j = 0; j < NNUE_FC1_OUT_DIM; j++)
    nnue_weights.l1_bias[j] = 0.0f;

  /* Layer 2: fan_in = 64 (CReLU[32] + SqrCReLU[32]) */
  scale = sqrtf(2.0f / NNUE_FC2_IN_DIM);
  for (i = 0; i < NNUE_FC2_IN_DIM; i++)
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

/* Size of old v1 weight format (l1: 256x32, l2: 32x32) */
#define NNUE_V1_WEIGHTS_SIZE 370052

int
nnue_load(const char *filename)
{
  FILE *fp = fopen(filename, "rb");
  long file_size;
  if (!fp)
    return 0;

  /* Determine file size to detect format version */
  fseek(fp, 0, SEEK_END);
  file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  if (file_size == (long)sizeof(NNUEWeights)) {
    /* v2 format: matches current struct size */
    if (fread(&nnue_weights, sizeof(NNUEWeights), 1, fp) != 1) {
      fclose(fp);
      return 0;
    }
  }
  else if (file_size == NNUE_V1_WEIGHTS_SIZE) {
    /* v1 format: migrate to v2.
     * v1 layout: l0[649][128], l0_bias[128], l1[256][32], l1_bias[32],
     *            l2[32][32], l2_bias[32], l3[32], l3_bias
     */
    int i, j;
    float old_l1[2 * NNUE_ACCUM_DIM][NNUE_HIDDEN1_DIM];
    float old_l1_bias[NNUE_HIDDEN1_DIM];
    float old_l2[NNUE_HIDDEN2_DIM][NNUE_HIDDEN2_DIM];

    /* l0 weights and bias are unchanged in layout */
    if (fread(nnue_weights.l0_weight, sizeof(nnue_weights.l0_weight), 1, fp) != 1
	|| fread(nnue_weights.l0_bias, sizeof(nnue_weights.l0_bias), 1, fp) != 1) {
      fclose(fp);
      return 0;
    }

    /* Read old l1 (256x32) and l1_bias (32) */
    if (fread(old_l1, sizeof(old_l1), 1, fp) != 1
	|| fread(old_l1_bias, sizeof(old_l1_bias), 1, fp) != 1) {
      fclose(fp);
      return 0;
    }

    /* Migrate l1: copy 32 columns, zero-init column 32 (skip) */
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++) {
      for (j = 0; j < NNUE_HIDDEN1_DIM; j++)
	nnue_weights.l1_weight[i][j] = old_l1[i][j];
      nnue_weights.l1_weight[i][NNUE_HIDDEN1_DIM] = 0.0f;
    }
    for (j = 0; j < NNUE_HIDDEN1_DIM; j++)
      nnue_weights.l1_bias[j] = old_l1_bias[j];
    nnue_weights.l1_bias[NNUE_HIDDEN1_DIM] = 0.0f;

    /* Read old l2 (32x32) */
    if (fread(old_l2, sizeof(old_l2), 1, fp) != 1) {
      fclose(fp);
      return 0;
    }

    /* Migrate l2: CReLU rows copy, SqrCReLU rows zero-init */
    for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
      for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
	nnue_weights.l2_weight[i][j] = old_l2[i][j];
    for (i = NNUE_HIDDEN2_DIM; i < NNUE_FC2_IN_DIM; i++)
      for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
	nnue_weights.l2_weight[i][j] = 0.0f;

    /* l2_bias, l3_weight, l3_bias unchanged */
    if (fread(nnue_weights.l2_bias, sizeof(nnue_weights.l2_bias), 1, fp) != 1
	|| fread(nnue_weights.l3_weight, sizeof(nnue_weights.l3_weight), 1, fp) != 1
	|| fread(&nnue_weights.l3_bias, sizeof(nnue_weights.l3_bias), 1, fp) != 1) {
      fclose(fp);
      return 0;
    }

    fprintf(stderr, "Migrated v1 weights to v2 format (SqrCReLU+skip).\n");
  }
  else {
    fclose(fp);
    return 0;
  }

  fclose(fp);
  nnue_accum_sp = 0;
  nnue_qaccum_sp = 0;
  nnue_quantize_weights();
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

/* Squared Clipped ReLU: clipped_relu(x)^2 */
static float
sqr_clipped_relu(float x)
{
  float cr = clipped_relu(x);
  return cr * cr;
}

float
nnue_evaluate(int color)
{
  NNUEAccumPair *pair = &nnue_accum_stack[nnue_accum_sp];
  NNUEAccumulator *stm_accum, *nstm_accum;
  float concat[2 * NNUE_ACCUM_DIM];
  float fc1_raw[NNUE_FC1_OUT_DIM];
  float fc2_in[NNUE_FC2_IN_DIM];
  float hidden2[NNUE_HIDDEN2_DIM];
  float output, skip;
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

  /* Layer 1: concat -> fc1_raw[33] (32 hidden + 1 skip) */
  for (j = 0; j < NNUE_FC1_OUT_DIM; j++) {
    float sum = nnue_weights.l1_bias[j];
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
      sum += concat[i] * nnue_weights.l1_weight[i][j];
    fc1_raw[j] = sum;
  }

  /* Extract skip connection (index 32) */
  skip = fc1_raw[NNUE_HIDDEN1_DIM];

  /* Apply CReLU and SqrCReLU to the 32 hidden outputs, concatenate */
  for (i = 0; i < NNUE_HIDDEN1_DIM; i++) {
    fc2_in[i] = clipped_relu(fc1_raw[i]);
    fc2_in[NNUE_HIDDEN1_DIM + i] = sqr_clipped_relu(fc1_raw[i]);
  }

  /* Layer 2: fc2_in[64] -> hidden2[32] */
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++) {
    float sum = nnue_weights.l2_bias[j];
    for (i = 0; i < NNUE_FC2_IN_DIM; i++)
      sum += fc2_in[i] * nnue_weights.l2_weight[i][j];
    hidden2[j] = clipped_relu(sum);
  }

  /* Layer 3: hidden2 -> output + skip connection (tanh) */
  output = nnue_weights.l3_bias + skip;
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    output += hidden2[i] * nnue_weights.l3_weight[i];
  output = tanhf(output);

  return output;
}

/* ---- Quantization and quantized inference ---- */

static int16_t
clamp_i16(int32_t x)
{
  if (x < -32768) return -32768;
  if (x > 32767) return 32767;
  return (int16_t)x;
}

static int8_t
clamp_i8(int32_t x)
{
  if (x < -128) return -128;
  if (x > 127) return 127;
  return (int8_t)x;
}

static uint8_t
clamp_u7(int32_t v)
{
  if (v < 0) return 0;
  if (v > 127) return 127;
  return (uint8_t)v;
}

static int32_t
qround(float val, float scale)
{
  float r = roundf(val * scale);
  return (int32_t)r;
}

void
nnue_quantize_weights(void)
{
  int i, j;

  /* L0: qw = round(w * WEIGHT_SCALE) */
  for (i = 0; i < NNUE_INPUT_DIM; i++)
    for (j = 0; j < NNUE_ACCUM_DIM; j++)
      nnue_qweights.l0_weight[i][j] =
	clamp_i16(qround(nnue_weights.l0_weight[i][j], NNUE_WEIGHT_SCALE));
  for (j = 0; j < NNUE_ACCUM_DIM; j++)
    nnue_qweights.l0_bias[j] =
      clamp_i16(qround(nnue_weights.l0_bias[j], NNUE_WEIGHT_SCALE));

  /* L1: qw = round(w * WSC), qb = round(b * 127 * WSC) */
  for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
    for (j = 0; j < NNUE_FC1_OUT_DIM; j++)
      nnue_qweights.l1_weight[i][j] =
	clamp_i8(qround(nnue_weights.l1_weight[i][j], NNUE_WEIGHT_SCALE));
  for (j = 0; j < NNUE_FC1_OUT_DIM; j++)
    nnue_qweights.l1_bias[j] =
      qround(nnue_weights.l1_bias[j], 127.0f * NNUE_WEIGHT_SCALE);

  /* L2: same scheme */
  for (i = 0; i < NNUE_FC2_IN_DIM; i++)
    for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
      nnue_qweights.l2_weight[i][j] =
	clamp_i8(qround(nnue_weights.l2_weight[i][j], NNUE_WEIGHT_SCALE));
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++)
    nnue_qweights.l2_bias[j] =
      qround(nnue_weights.l2_bias[j], 127.0f * NNUE_WEIGHT_SCALE);

  /* L3 */
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    nnue_qweights.l3_weight[i] =
      clamp_i8(qround(nnue_weights.l3_weight[i], NNUE_WEIGHT_SCALE));
  nnue_qweights.l3_bias =
    qround(nnue_weights.l3_bias, 127.0f * NNUE_WEIGHT_SCALE);
}

/* Compute quantized accumulator for one perspective using sparse features.
 * Avoids dense float array and iterates only active features.
 */
static void
compute_qaccumulator(NNUEQuantizedAccum *accum, int perspective,
		     int previous_pass,
		     const int unconditional_own[],
		     const int unconditional_opp[])
{
  int i, j, pos, pt_idx, k, base;
  int own = perspective;
  int opp = OTHER_COLOR(perspective);
  uint16_t active[48]; /* max active features per perspective */
  int num_active = 0;

  /* Collect active feature indices (sparse) */
  for (i = 0; i < NNUE_BOARD_SIZE; i++) {
    for (j = 0; j < NNUE_BOARD_SIZE; j++) {
      pos = POS(i, j);
      pt_idx = i * NNUE_BOARD_SIZE + j;
      base = pt_idx * NNUE_FEAT_PER_PT;

      if (board[pos] == own)
	active[num_active++] = (uint16_t)(base + 0);
      else if (board[pos] == opp)
	active[num_active++] = (uint16_t)(base + 1);

      if (board_ko_pos == pos)
	active[num_active++] = (uint16_t)(base + 2);

      if (IS_STONE(board[pos])) {
	int libs = countlib(pos);
	if (libs == 1)
	  active[num_active++] = (uint16_t)(base + 3);
	else if (libs == 2)
	  active[num_active++] = (uint16_t)(base + 4);
	else if (libs == 3)
	  active[num_active++] = (uint16_t)(base + 5);
      }

      if (unconditional_own[pos])
	active[num_active++] = (uint16_t)(base + 6);
      if (unconditional_opp[pos])
	active[num_active++] = (uint16_t)(base + 7);
    }
  }

  if (previous_pass)
    active[num_active++] = (uint16_t)(NNUE_NUM_POINTS * NNUE_FEAT_PER_PT);

  /* Accumulate: bias + sum of weight rows for active features */
  for (j = 0; j < NNUE_ACCUM_DIM; j++)
    accum->values[j] = nnue_qweights.l0_bias[j];
  for (k = 0; k < num_active; k++) {
    int idx = active[k];
    for (j = 0; j < NNUE_ACCUM_DIM; j++)
      accum->values[j] += nnue_qweights.l0_weight[idx][j];
  }
}

void
nnue_qaccum_refresh(int color, int previous_pass)
{
  NNUEQuantizedAccumPair *pair;
  int unconditional_w[BOARDMAX];
  int unconditional_b[BOARDMAX];
  UNUSED(color);

  pair = &nnue_qaccum_stack[nnue_qaccum_sp];

  /* Compute pass-alive territories once for both perspectives */
  unconditional_life(unconditional_w, WHITE);
  unconditional_life(unconditional_b, BLACK);

  compute_qaccumulator(&pair->white_accum, WHITE, previous_pass,
		       unconditional_w, unconditional_b);
  compute_qaccumulator(&pair->black_accum, BLACK, previous_pass,
		       unconditional_b, unconditional_w);
}

void
nnue_qaccum_push(void)
{
  if (nnue_qaccum_sp + 1 >= NNUE_MAX_STACK) {
    fprintf(stderr, "nnue: quantized accumulator stack overflow\n");
    return;
  }
  nnue_qaccum_stack[nnue_qaccum_sp + 1] = nnue_qaccum_stack[nnue_qaccum_sp];
  nnue_qaccum_sp++;
}

void
nnue_qaccum_pop(void)
{
  if (nnue_qaccum_sp <= 0) {
    fprintf(stderr, "nnue: quantized accumulator stack underflow\n");
    return;
  }
  nnue_qaccum_sp--;
}

void
nnue_qaccum_update_after_move(int move, int color)
{
  /* Pass moves don't change the board — just refresh for pass-end feature */
  if (move == PASS_MOVE) {
    nnue_qaccum_refresh(OTHER_COLOR(color), 1);
    return;
  }

  /* Full refresh using quantized int16 arithmetic.
   *
   * A true incremental update would track which features changed at the
   * move point and its neighbors, then apply delta:
   *   accum[j] -= qweight[old_feat][j]; accum[j] += qweight[new_feat][j];
   * However, Go's pass-alive features (computed by Benson's algorithm)
   * can change globally with any move, making partial updates unreliable.
   *
   * The quantized refresh is already significantly faster than the old
   * float32 path: int16 additions vs float multiplications.
   * Future optimization: skip pass-alive features in internal nodes
   * and only compute them at leaf nodes.
   */
  nnue_qaccum_refresh(OTHER_COLOR(color), 0);
}

float
nnue_evaluate_quantized(int color)
{
  NNUEQuantizedAccumPair *pair = &nnue_qaccum_stack[nnue_qaccum_sp];
  NNUEQuantizedAccum *stm_accum, *nstm_accum;
  uint8_t concat[2 * NNUE_ACCUM_DIM];
  int32_t fc1_raw[NNUE_FC1_OUT_DIM];
  uint8_t fc2_in[NNUE_FC2_IN_DIM];
  int32_t h2_raw[NNUE_HIDDEN2_DIM];
  uint8_t h2_act[NNUE_HIDDEN2_DIM];
  int32_t output_raw;
  int32_t skip_raw;
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

  /* Quantized CReLU on accumulators: clamp(accum >> WSB, 0, 127) -> uint8 */
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[i] = clamp_u7((int32_t)stm_accum->values[i] >> NNUE_WEIGHT_SCALE_BITS);
  for (i = 0; i < NNUE_ACCUM_DIM; i++)
    concat[NNUE_ACCUM_DIM + i] = clamp_u7((int32_t)nstm_accum->values[i] >> NNUE_WEIGHT_SCALE_BITS);

  /* FC1: uint8 * int8 -> int32, bias is pre-scaled */
  for (j = 0; j < NNUE_FC1_OUT_DIM; j++) {
    int32_t sum = nnue_qweights.l1_bias[j];
    for (i = 0; i < 2 * NNUE_ACCUM_DIM; i++)
      sum += (int32_t)concat[i] * (int32_t)nnue_qweights.l1_weight[i][j];
    fc1_raw[j] = sum;
  }

  /* Extract skip connection (index 32) */
  skip_raw = fc1_raw[NNUE_HIDDEN1_DIM];

  /* Quantized CReLU and SqrCReLU on FC1 hidden outputs.
   * fc1_raw is in scale 127*WSC. To get uint8 [0,127]:
   * CReLU: clamp(fc1_raw / (127 * WSC), 0, 1) * 127
   *      = clamp(fc1_raw / WSC, 0, 127)
   *      ≈ clamp(fc1_raw >> (WSB + 7 - 7), 0, 127)
   *      Actually: CReLU output = clamp(fc1_raw / (127 * WSC), 0, 127)
   *      Simplified: divide by WSC to get [0..127] range.
   * SqrCReLU: min(127, (fc1_raw * fc1_raw) >> (2*WSB + 14 + 7 - 7))
   *         = min(127, cr * cr / 127)
   */
  for (i = 0; i < NNUE_HIDDEN1_DIM; i++) {
    int32_t cr = clamp_u7(fc1_raw[i] >> NNUE_WEIGHT_SCALE_BITS);
    int32_t sq = (cr * cr + 63) / 127;
    fc2_in[i] = (uint8_t)cr;
    fc2_in[NNUE_HIDDEN1_DIM + i] = clamp_u7(sq);
  }

  /* FC2: uint8 * int8 -> int32 */
  for (j = 0; j < NNUE_HIDDEN2_DIM; j++) {
    int32_t sum = nnue_qweights.l2_bias[j];
    for (i = 0; i < NNUE_FC2_IN_DIM; i++)
      sum += (int32_t)fc2_in[i] * (int32_t)nnue_qweights.l2_weight[i][j];
    h2_raw[j] = sum;
  }

  /* Quantized CReLU on FC2 output */
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    h2_act[i] = clamp_u7(h2_raw[i] >> NNUE_WEIGHT_SCALE_BITS);

  /* FC3: uint8 * int8 -> int32, plus skip */
  output_raw = nnue_qweights.l3_bias;
  for (i = 0; i < NNUE_HIDDEN2_DIM; i++)
    output_raw += (int32_t)h2_act[i] * (int32_t)nnue_qweights.l3_weight[i];

  /* Convert to float: output_raw is in scale 127*WSC.
   * skip_raw is also in scale 127*WSC.
   * Dequantize: float_val = raw / (127.0 * WSC)
   */
  output = (float)(output_raw + skip_raw) / (127.0f * NNUE_WEIGHT_SCALE);
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
