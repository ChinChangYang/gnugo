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

#include "nnue.h"
#include "nnue_train.h"
#include "interface.h"

void
play_train(int generations, int games_per_gen, int node_limit,
	   int deep_node_limit, const char *weights_file)
{
  fprintf(stderr, "NNUE Training Mode\n");
  fprintf(stderr, "  Generations:     %d\n", generations);
  fprintf(stderr, "  Games/gen:       %d\n", games_per_gen);
  fprintf(stderr, "  Shallow nodes:   %d\n", node_limit);
  fprintf(stderr, "  Deep nodes:      %d\n", deep_node_limit);
  fprintf(stderr, "  Weights file:    %s\n", weights_file);
  fprintf(stderr, "\n");

  nnue_train_run(generations, games_per_gen, node_limit,
		 deep_node_limit, weights_file);

  fprintf(stderr, "\nTraining complete.\n");
}

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */
