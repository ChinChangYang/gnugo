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

#ifndef _ALPHABETA_H_
#define _ALPHABETA_H_

#include "hash.h"

/* Transposition table entry flags */
#define TT_EXACT   0
#define TT_ALPHA   1  /* upper bound (fail-low) */
#define TT_BETA    2  /* lower bound (fail-high) */

/* Transposition table size (power of 2) */
#define TT_SIZE    (1 << 18)  /* 262144 entries */
#define TT_MASK    (TT_SIZE - 1)

typedef struct {
  Hash_data hash;
  float score;
  int best_move;
  short depth;
  short flag;
  int valid;
} TTEntry;

typedef struct {
  TTEntry *table;
  int node_count;
  int node_limit;
  int consecutive_passes;
  int search_aborted;
} AlphaBetaState;

/* Initialize the alpha-beta search state. Call once at startup. */
void alphabeta_init(AlphaBetaState *state, int node_limit);

/* Free search state resources. */
void alphabeta_free(AlphaBetaState *state);

/* Clear the transposition table. */
void alphabeta_clear_tt(AlphaBetaState *state);

/* Generate a move using alpha-beta search with NNUE evaluation.
 * color = side to move (BLACK or WHITE).
 * node_limit = maximum number of nodes to search (0 = use state default).
 * Returns the best move found (may be PASS_MOVE).
 */
int alphabeta_genmove(int color, int node_limit);

/* Evaluate a position to a given depth using alpha-beta search.
 * Returns the evaluation score in [-1, +1] from color's perspective.
 * color = side to move.
 * node_limit = search budget.
 */
float alphabeta_eval_position(int color, int node_limit);

#endif  /* _ALPHABETA_H_ */

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
