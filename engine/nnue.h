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

#ifndef _NNUE_H_
#define _NNUE_H_

/* NNUE (Efficiently Updatable Neural Network) for 9x9 Go evaluation.
 *
 * Architecture:
 *   Input(649 sparse binary) -> Accumulator[128] (per perspective, clipped ReLU)
 *   Concat(STM[128], NSTM[128]) -> FC[256->32] (clipped ReLU)
 *   FC[32->32] (clipped ReLU)
 *   FC[32->1] (tanh output in [-1, +1])
 *
 * Features per intersection (81 points on 9x9):
 *   0: own stone      1: opponent stone
 *   2: ko point       3: in atari (1 liberty)
 *   4: 2 liberties    5: 3 liberties
 *   6: own pass-alive 7: opponent pass-alive
 * Global feature:
 *   648: pass will end game
 */

#define NNUE_BOARD_SIZE   9
#define NNUE_NUM_POINTS   (NNUE_BOARD_SIZE * NNUE_BOARD_SIZE)  /* 81 */
#define NNUE_FEAT_PER_PT  8
#define NNUE_NUM_GLOBAL   1
#define NNUE_INPUT_DIM    (NNUE_NUM_POINTS * NNUE_FEAT_PER_PT + NNUE_NUM_GLOBAL) /* 649 */
#define NNUE_ACCUM_DIM    128
#define NNUE_HIDDEN1_DIM  32
#define NNUE_HIDDEN2_DIM  32
#define NNUE_OUTPUT_DIM   1

/* Maximum search stack depth for accumulator management */
#define NNUE_MAX_STACK    512

typedef struct {
  /* Layer 0: input -> accumulator (shared, perspective-flipped) */
  float l0_weight[NNUE_INPUT_DIM][NNUE_ACCUM_DIM];  /* 649 x 128 */
  float l0_bias[NNUE_ACCUM_DIM];                     /* 128 */

  /* Layer 1: concat(stm_accum, nstm_accum) -> hidden1 */
  float l1_weight[2 * NNUE_ACCUM_DIM][NNUE_HIDDEN1_DIM];  /* 256 x 32 */
  float l1_bias[NNUE_HIDDEN1_DIM];                          /* 32 */

  /* Layer 2: hidden1 -> hidden2 */
  float l2_weight[NNUE_HIDDEN2_DIM][NNUE_HIDDEN2_DIM];  /* 32 x 32 */
  float l2_bias[NNUE_HIDDEN2_DIM];                       /* 32 */

  /* Layer 3: hidden2 -> output */
  float l3_weight[NNUE_HIDDEN2_DIM];  /* 32 */
  float l3_bias;                       /* 1 */
} NNUEWeights;

typedef struct {
  float values[NNUE_ACCUM_DIM];  /* 128 */
} NNUEAccumulator;

typedef struct {
  NNUEAccumulator white_accum;  /* White's perspective accumulator */
  NNUEAccumulator black_accum;  /* Black's perspective accumulator */
} NNUEAccumPair;

/* Global NNUE state */
extern NNUEWeights nnue_weights;
extern NNUEAccumPair nnue_accum_stack[NNUE_MAX_STACK];
extern int nnue_accum_sp;

/* Initialize NNUE weights with small random values */
void nnue_init_random(unsigned int seed);

/* Load/save weights from/to binary file */
int nnue_load(const char *filename);
int nnue_save(const char *filename);

/* Refresh accumulators from scratch for the current board position.
 * color = side to move (BLACK or WHITE).
 * previous_pass = 1 if the previous move was a pass, 0 otherwise.
 */
void nnue_accumulator_refresh(int color, int previous_pass);

/* Push current accumulator state before trymove. */
void nnue_accum_push(void);

/* Pop accumulator state after popgo. */
void nnue_accum_pop(void);

/* Evaluate the current position from the side-to-move's perspective.
 * Returns a value in [-1, +1] where +1 = STM winning.
 * color = side to move.
 */
float nnue_evaluate(int color);

/* Tromp-Taylor area scoring.
 * Returns score from White's perspective (positive = White wins).
 * komi is added to White's score.
 */
float nnue_score_position(float komi);

/* Check if the previous game-level move was a pass.
 * Only valid at root level (stackp == 0); uses move_history.
 */
int detect_previous_pass(void);

#endif  /* _NNUE_H_ */

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
