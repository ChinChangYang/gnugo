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

#ifndef _NNUE_TRAIN_H_
#define _NNUE_TRAIN_H_

#include "nnue.h"

#include <stdint.h>

/* Maximum training samples */
#define NNUE_MAX_SAMPLES 500000

/* Maximum active features per perspective (generous upper bound) */
#define NNUE_MAX_ACTIVE_FEATURES 48

/* Training sample: sparse features + search score + game outcome */
typedef struct {
  uint16_t stm_indices[NNUE_MAX_ACTIVE_FEATURES];
  uint16_t nstm_indices[NNUE_MAX_ACTIVE_FEATURES];
  uint8_t  num_stm;          /* number of active STM features */
  uint8_t  num_nstm;         /* number of active NSTM features */
  float    search_score;     /* deep search score in [-1, +1] for STM */
  float    game_result;      /* Tromp-Taylor outcome from STM perspective */
} NNUETrainSample;

/* Gradient accumulators matching NNUEWeights layout */
typedef struct {
  float l0_weight[NNUE_INPUT_DIM][NNUE_ACCUM_DIM];
  float l0_bias[NNUE_ACCUM_DIM];
  float l1_weight[2 * NNUE_ACCUM_DIM][NNUE_FC1_OUT_DIM];
  float l1_bias[NNUE_FC1_OUT_DIM];
  float l2_weight[NNUE_FC2_IN_DIM][NNUE_HIDDEN2_DIM];
  float l2_bias[NNUE_HIDDEN2_DIM];
  float l3_weight[NNUE_HIDDEN2_DIM];
  float l3_bias;
} NNUEGradients;

/* Training hyperparameters */
typedef struct NNUETrainConfig {
  float lr_max;       /* max learning rate (default 0.001) */
  float lr_min;       /* min learning rate (default 0.00001) */
  float beta1;        /* Adam first moment decay (default 0.9) */
  float beta2;        /* Adam second moment decay (default 0.999) */
  float epsilon;      /* Adam numerical stability (default 1e-8) */
  float lambda;       /* blended loss: lambda*search + (1-lambda)*game_result */
  int   batch_size;   /* mini-batch size (default 1024) */
  int   num_epochs;   /* epochs per generation (default 10) */
  int   total_steps;  /* total batches across all epochs (computed) */
} NNUETrainConfig;

/* Adam optimizer state — same layout as NNUEGradients for m and v */
typedef struct {
  NNUEGradients m;    /* first moment (mean of gradients) */
  NNUEGradients v;    /* second moment (mean of squared gradients) */
  int t;              /* timestep counter */
} NNUEAdamState;

/* Run backpropagation for a single sample.
 * Accumulates gradients into grad.
 * Uses blended target: lambda*search_score + (1-lambda)*game_result.
 * Returns the MSE loss for this sample.
 */
float nnue_backward(const NNUETrainSample *sample, NNUEGradients *grad,
		    float lambda);

/* Zero out all gradients */
void nnue_grad_zero(NNUEGradients *grad);

/* Initialize Adam optimizer state */
void nnue_adam_init(NNUEAdamState *state);

/* Apply Adam update with given learning rate */
void nnue_adam_update(NNUEGradients *grad, NNUEAdamState *state,
		      const NNUETrainConfig *config, float lr);

/* Compute cosine-annealed learning rate */
float nnue_cosine_lr(const NNUETrainConfig *config, int step);

/* Initialize training config with defaults */
void nnue_train_config_defaults(NNUETrainConfig *config);

/* Run the full training pipeline.
 * generations = number of training generations
 * games_per_gen = self-play games per generation for position collection
 * node_limit = shallow search node limit for game play
 * deep_node_limit = deep search node limit for labeling
 * weights_file = path to save/load weights
 * config = training hyperparameters (NULL for defaults)
 */
void nnue_train_run(int generations, int games_per_gen, int node_limit,
		    int deep_node_limit, const char *weights_file,
		    const NNUETrainConfig *config);

#endif  /* _NNUE_TRAIN_H_ */

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
