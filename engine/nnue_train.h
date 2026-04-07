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

/* Maximum training samples in the sliding window */
#define NNUE_MAX_SAMPLES 50000

/* Training sample: board features + target score from deep search */
typedef struct {
  float stm_features[NNUE_INPUT_DIM];
  float nstm_features[NNUE_INPUT_DIM];
  float target;  /* deep search score in [-1, +1] for STM */
} NNUETrainSample;

/* Gradient accumulators matching NNUEWeights layout */
typedef struct {
  float l0_weight[NNUE_INPUT_DIM][NNUE_ACCUM_DIM];
  float l0_bias[NNUE_ACCUM_DIM];
  float l1_weight[2 * NNUE_ACCUM_DIM][NNUE_HIDDEN1_DIM];
  float l1_bias[NNUE_HIDDEN1_DIM];
  float l2_weight[NNUE_HIDDEN2_DIM][NNUE_HIDDEN2_DIM];
  float l2_bias[NNUE_HIDDEN2_DIM];
  float l3_weight[NNUE_HIDDEN2_DIM];
  float l3_bias;
} NNUEGradients;

/* Run backpropagation for a single sample.
 * Accumulates gradients into grad.
 * Returns the MSE loss for this sample.
 */
float nnue_backward(const NNUETrainSample *sample, NNUEGradients *grad);

/* Apply SGD update: weights -= lr * (grad / batch_size) */
void nnue_sgd_update(NNUEGradients *grad, float lr, int batch_size);

/* Zero out all gradients */
void nnue_grad_zero(NNUEGradients *grad);

/* Run the full training pipeline.
 * generations = number of training generations
 * games_per_gen = self-play games per generation for position collection
 * node_limit = shallow search node limit for game play
 * deep_node_limit = deep search node limit for labeling
 * weights_file = path to save/load weights
 */
void nnue_train_run(int generations, int games_per_gen, int node_limit,
		    int deep_node_limit, const char *weights_file);

#endif  /* _NNUE_TRAIN_H_ */

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
