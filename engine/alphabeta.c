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

#include "nnue.h"
#include "alphabeta.h"
#include "liberty.h"

/* Global search state (single instance for now) */
static AlphaBetaState global_ab_state;
static int global_ab_initialized = 0;

/* Move list for move generation */
#define MAX_AB_MOVES (MAX_BOARD * MAX_BOARD + 1)  /* board points + pass */

typedef struct {
  int move;
  int priority;
} ScoredMove;

/* Ensure global state is initialized */
static void
ensure_initialized(void)
{
  if (!global_ab_initialized) {
    alphabeta_init(&global_ab_state, 1000);
    global_ab_initialized = 1;
  }
}

void
alphabeta_init(AlphaBetaState *state, int node_limit)
{
  state->table = (TTEntry *)calloc(TT_SIZE, sizeof(TTEntry));
  state->node_limit = node_limit;
  state->node_count = 0;
  state->consecutive_passes = 0;
  state->search_aborted = 0;
}

void
alphabeta_free(AlphaBetaState *state)
{
  if (state->table) {
    free(state->table);
    state->table = NULL;
  }
}

void
alphabeta_clear_tt(AlphaBetaState *state)
{
  memset(state->table, 0, TT_SIZE * sizeof(TTEntry));
}

/* Hash lookup */
static int
hashdata_equal(const Hash_data *a, const Hash_data *b)
{
  int i;
  for (i = 0; i < NUM_HASHVALUES; i++)
    if (a->hashval[i] != b->hashval[i])
      return 0;
  return 1;
}

static TTEntry *
tt_probe(AlphaBetaState *state, const Hash_data *hash)
{
  unsigned long idx = hash->hashval[0] & TT_MASK;
  TTEntry *entry = &state->table[idx];
  if (entry->valid && hashdata_equal(&entry->hash, hash))
    return entry;
  return NULL;
}

static void
tt_store(AlphaBetaState *state, const Hash_data *hash,
	 float score, int best_move, int depth, int flag)
{
  unsigned long idx = hash->hashval[0] & TT_MASK;
  TTEntry *entry = &state->table[idx];

  /* Always replace (could use depth-preferred later) */
  entry->hash = *hash;
  entry->score = score;
  entry->best_move = best_move;
  entry->depth = (short)depth;
  entry->flag = (short)flag;
  entry->valid = 1;
}

/* Generate legal moves for color, sorted by priority.
 * Returns number of moves generated.
 * tt_best = best move from TT (or NO_MOVE).
 * last_move = the previous move played (for adjacency ordering).
 *
 * Note: uses is_legal() which only checks simple ko, not superko.
 * is_allowed_move() would enforce superko but requires stackp == 0,
 * so it cannot be used inside the search tree.
 */
static int
generate_moves(ScoredMove moves[], int color, int tt_best, int last_move)
{
  int i, j, pos;
  int num_moves = 0;

  for (i = 0; i < board_size; i++) {
    for (j = 0; j < board_size; j++) {
      pos = POS(i, j);
      if (board[pos] != EMPTY)
	continue;
      if (!is_legal(pos, color))
	continue;

      moves[num_moves].move = pos;

      /* Assign priority for move ordering */
      if (pos == tt_best) {
	moves[num_moves].priority = 10000;
      }
      else {
	int prio = 0;

	/* Captures: check if this move captures opponent stones */
	int opp = OTHER_COLOR(color);
	int di, dj;
	int deltas[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
	int d;
	for (d = 0; d < 4; d++) {
	  di = deltas[d][0];
	  dj = deltas[d][1];
	  int ni = i + di;
	  int nj = j + dj;
	  if (ni >= 0 && ni < board_size && nj >= 0 && nj < board_size) {
	    int nb = POS(ni, nj);
	    if (board[nb] == opp && countlib(nb) == 1)
	      prio += 100;
	  }
	}

	/* Adjacency to last move */
	if (last_move != NO_MOVE && last_move != PASS_MOVE) {
	  int li = I(last_move);
	  int lj = J(last_move);
	  int dist = abs(i - li) + abs(j - lj);
	  if (dist <= 2)
	    prio += 50 - dist * 10;
	}

	/* Atari escape: if own group is in atari, saving moves get priority */
	for (d = 0; d < 4; d++) {
	  di = deltas[d][0];
	  dj = deltas[d][1];
	  int ni = i + di;
	  int nj = j + dj;
	  if (ni >= 0 && ni < board_size && nj >= 0 && nj < board_size) {
	    int nb = POS(ni, nj);
	    if (board[nb] == color && countlib(nb) == 1)
	      prio += 80;
	  }
	}

	moves[num_moves].priority = prio;
      }
      num_moves++;
    }
  }

  /* Add pass move last with lowest priority */
  moves[num_moves].move = PASS_MOVE;
  moves[num_moves].priority = -1000;
  num_moves++;

  /* Simple insertion sort by priority (descending) */
  {
    int k, l;
    for (k = 1; k < num_moves; k++) {
      ScoredMove tmp = moves[k];
      l = k - 1;
      while (l >= 0 && moves[l].priority < tmp.priority) {
	moves[l + 1] = moves[l];
	l--;
      }
      moves[l + 1] = tmp;
    }
  }

  return num_moves;
}

/* Negamax alpha-beta search.
 * color = side to move.
 * alpha, beta = search window.
 * depth = remaining depth.
 * consecutive_passes = number of consecutive passes so far.
 * last_move = the previous move.
 * Returns evaluation in [-1, +1] from color's perspective.
 */
static float
negamax(AlphaBetaState *state, int color, float alpha, float beta,
	int depth, int consecutive_passes, int last_move)
{
  TTEntry *tt_entry;
  ScoredMove moves[MAX_AB_MOVES];
  int num_moves, i;
  int best_move = PASS_MOVE;
  float best_score = -2.0f;
  float orig_alpha = alpha;
  int tt_best = NO_MOVE;

  /* Check node budget */
  if (state->search_aborted)
    return 0.0f;

  /* Two consecutive passes = game over, use Tromp-Taylor scoring */
  if (consecutive_passes >= 2) {
    float tt_score = nnue_score_position(komi);
    /* Convert to score from color's perspective: +1 if color wins */
    if (color == WHITE)
      return tt_score > 0 ? 1.0f : (tt_score < 0 ? -1.0f : 0.0f);
    else
      return tt_score < 0 ? 1.0f : (tt_score > 0 ? -1.0f : 0.0f);
  }

  /* Probe transposition table */
  tt_entry = tt_probe(state, &board_hash);
  if (tt_entry && tt_entry->depth >= depth) {
    if (tt_entry->flag == TT_EXACT)
      return tt_entry->score;
    if (tt_entry->flag == TT_ALPHA && tt_entry->score <= alpha)
      return alpha;
    if (tt_entry->flag == TT_BETA && tt_entry->score >= beta)
      return beta;
  }
  if (tt_entry)
    tt_best = tt_entry->best_move;

  /* Leaf node: evaluate with NNUE */
  if (depth <= 0 || state->node_count >= state->node_limit) {
    if (state->node_count >= state->node_limit)
      state->search_aborted = 1;
    /* Refresh quantized accumulator and evaluate */
    nnue_qaccum_refresh(color, (last_move == PASS_MOVE) ? 1 : 0);
    return nnue_evaluate_quantized(color);
  }

  /* Generate and search moves */
  num_moves = generate_moves(moves, color, tt_best, last_move);

  for (i = 0; i < num_moves; i++) {
    int move = moves[i].move;
    float score;
    int new_consecutive = (move == PASS_MOVE) ? consecutive_passes + 1 : 0;

    if (move == PASS_MOVE) {
      /* Pass doesn't need trymove, just recurse */
      score = -negamax(state, OTHER_COLOR(color), -beta, -alpha,
		       depth - 1, new_consecutive, PASS_MOVE);
    }
    else {
      if (!trymove(move, color, "alphabeta", NO_MOVE)) {
	continue;
      }
      state->node_count++;

      nnue_qaccum_push();
      nnue_qaccum_update_after_move(move, color);

      score = -negamax(state, OTHER_COLOR(color), -beta, -alpha,
		       depth - 1, 0, move);

      nnue_qaccum_pop();
      popgo();
    }

    if (state->search_aborted)
      return best_score > -2.0f ? best_score : 0.0f;

    if (score > best_score) {
      best_score = score;
      best_move = move;
    }

    if (score > alpha)
      alpha = score;

    if (alpha >= beta)
      break;  /* Beta cutoff */
  }

  /* Store in TT */
  {
    int flag;
    if (best_score <= orig_alpha)
      flag = TT_ALPHA;
    else if (best_score >= beta)
      flag = TT_BETA;
    else
      flag = TT_EXACT;

    tt_store(state, &board_hash, best_score, best_move, depth, flag);
  }

  return best_score;
}

/* Main entry point: generate a move using iterative deepening alpha-beta */
int
alphabeta_genmove(int color, int node_limit)
{
  int depth;
  int best_move = PASS_MOVE;
  float best_score = -2.0f;
  AlphaBetaState *state;

  ensure_initialized();
  state = &global_ab_state;

  if (node_limit > 0)
    state->node_limit = node_limit;
  else
    state->node_limit = nnue_node_limit;

  state->node_count = 0;
  state->search_aborted = 0;
  alphabeta_clear_tt(state);

  /* Refresh NNUE accumulators for root position */
  nnue_qaccum_refresh(color, detect_previous_pass());

  /* Iterative deepening */
  for (depth = 1; depth <= 100; depth++) {
    float score;
    int root_best = PASS_MOVE;
    ScoredMove moves[MAX_AB_MOVES];
    int num_moves, i;
    float alpha = -2.0f;
    float beta = 2.0f;
    int tt_best = NO_MOVE;
    TTEntry *tt_entry;
    int prev_node_count = state->node_count;

    /* Check TT for root position */
    tt_entry = tt_probe(state, &board_hash);
    if (tt_entry)
      tt_best = tt_entry->best_move;

    /* Detect consecutive passes for root */
    int root_consecutive = 0;

    num_moves = generate_moves(moves, color, tt_best, NO_MOVE);

    for (i = 0; i < num_moves; i++) {
      int move = moves[i].move;
      int new_consecutive = (move == PASS_MOVE) ? root_consecutive + 1 : 0;

      if (move == PASS_MOVE) {
	score = -negamax(state, OTHER_COLOR(color), -beta, -alpha,
			 depth - 1, new_consecutive, PASS_MOVE);
      }
      else {
	if (!trymove(move, color, "ab_root", NO_MOVE))
	  continue;
	state->node_count++;

	nnue_qaccum_push();
	nnue_qaccum_update_after_move(move, color);

	score = -negamax(state, OTHER_COLOR(color), -beta, -alpha,
			 depth - 1, 0, move);

	nnue_qaccum_pop();
	popgo();
      }

      if (state->search_aborted)
	break;

      if (score > alpha) {
	alpha = score;
	root_best = move;
      }
    }

    /* If we completed at least depth 1, record result */
    if (!state->search_aborted || depth == 1) {
      if (root_best != PASS_MOVE || best_move == PASS_MOVE) {
	best_move = root_best;
	best_score = alpha;
      }
    }

    /* Stop if budget exhausted */
    if (state->search_aborted)
      break;

    /* Stop if no new nodes explored (search complete) */
    if (state->node_count == prev_node_count)
      break;
  }

  return best_move;
}

/* Evaluate a position's score using alpha-beta search */
float
alphabeta_eval_position(int color, int node_limit)
{
  int depth;
  float best_score = 0.0f;
  AlphaBetaState *state;

  ensure_initialized();
  state = &global_ab_state;

  state->node_limit = node_limit;
  state->node_count = 0;
  state->search_aborted = 0;
  alphabeta_clear_tt(state);

  nnue_qaccum_refresh(color, detect_previous_pass());

  for (depth = 1; depth <= 100; depth++) {
    float score;
    ScoredMove moves[MAX_AB_MOVES];
    int num_moves, i;
    float alpha = -2.0f;
    float beta = 2.0f;
    int tt_best = NO_MOVE;
    TTEntry *tt_entry;
    int prev_node_count = state->node_count;

    tt_entry = tt_probe(state, &board_hash);
    if (tt_entry)
      tt_best = tt_entry->best_move;

    num_moves = generate_moves(moves, color, tt_best, NO_MOVE);

    for (i = 0; i < num_moves; i++) {
      int move = moves[i].move;
      int new_consecutive = (move == PASS_MOVE) ? 1 : 0;

      if (move == PASS_MOVE) {
	score = -negamax(state, OTHER_COLOR(color), -beta, -alpha,
			 depth - 1, new_consecutive, PASS_MOVE);
      }
      else {
	if (!trymove(move, color, "ab_eval", NO_MOVE))
	  continue;
	state->node_count++;

	nnue_qaccum_push();
	nnue_qaccum_update_after_move(move, color);

	score = -negamax(state, OTHER_COLOR(color), -beta, -alpha,
			 depth - 1, 0, move);

	nnue_qaccum_pop();
	popgo();
      }

      if (state->search_aborted)
	break;

      if (score > alpha)
	alpha = score;
    }

    if (!state->search_aborted || depth == 1)
      best_score = alpha;

    if (state->search_aborted)
      break;

    if (state->node_count == prev_node_count)
      break;
  }

  return best_score;
}

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
