/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 * This is GNU GO, a Go program. Contact gnugo@gnu.org, or see   *
 * http://www.gnu.org/software/gnugo/ for more information.      *
 *                                                               *
 * Copyright 1999, 2000, 2001 by the Free Software Foundation.   *
 *                                                               *
 * This program is free software; you can redistribute it and/or *
 * modify it under the terms of the GNU General Public License   *
 * as published by the Free Software Foundation - version 2.     *
 *                                                               *
 * This program is distributed in the hope that it will be       *
 * useful, but WITHOUT ANY WARRANTY; without even the implied    *
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR       *
 * PURPOSE.  See the GNU General Public License in file COPYING  *
 * for more details.                                             *
 *                                                               *
 * You should have received a copy of the GNU General Public     *
 * License along with this program; if not, write to the Free    *
 * Software Foundation, Inc., 59 Temple Place - Suite 330,       *
 * Boston, MA 02111, USA.                                        *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#include "liberty.h"
#include "hash.h"
#include "cache.h"
#include "sgftree.h"

Hashtable *movehash;

static int hashtable_init(Hashtable *table, int tablesize, int num_nodes,
			  int num_results);
static Hashtable *hashtable_new(int tablesize, int num_nodes, int num_results);
static void hashtable_clear(Hashtable *table);

static Hashnode *hashtable_enter_position(Hashtable *table, Hash_data *hd);
static Hashnode *hashtable_search(Hashtable *table, Hash_data *hd);

static Read_result *hashnode_search(Hashnode *node, int routine, int komaster,
				    int kom_i, int kom_j, int i, int j);
static Read_result *hashnode_new_result(Hashtable *table, Hashnode *node, 
					int routine, 
					int komaster, int kom_i, int kom_j,
					int ri, int rj);

static void hashtable_unlink_closed_results(Hashnode *node, 
					    int exclusions, 
					    unsigned int stackplimit,
					    int statistics[][20]);
static void hashtable_partially_clear(Hashtable *table);
static int do_get_read_result(int routine, int komaster, int kom_i, int kom_j,
			      int *si, int *sj,
			      Read_result **read_result);

/*
 * Dump an ASCII representation of the contents of a Read_result onto
 * the FILE outfile. 
 */



void
read_result_dump(Read_result *result, FILE *outfile)
{
  fprintf(outfile, "Komaster %d (%d, %d) Routine %d, (%d, %d), depth: %d ",
	  ((result->compressed_data) >> 29) & 0x03,
	  (((result->compressed_data) >> 24) & 0x1f) - 1,
	  (((result->compressed_data) >> 19) & 0x1f) - 1,
	  ((result->compressed_data) >> 15) & 0x0f,
	  ((result->compressed_data) >> 10) & 0x1f,
	  ((result->compressed_data) >> 5)  & 0x1f,
	  ((result->compressed_data) >> 0)  & 0x1f);
  fprintf(outfile, "Result: %d %d, (%d, %d)\n",
	  ((result->result_ri_rj) >> 24) & 0xff,
	  ((result->result_ri_rj) >> 16) & 0xff,
	  ((result->result_ri_rj) >> 8) & 0xff,
	  ((result->result_ri_rj) >> 0) & 0xff);
}


/*
 * Dump an ASCII representation of the contents of a Hashnode onto
 * the FILE outfile. 
 */

void
hashnode_dump(Hashnode *node, FILE *outfile)
{
  Read_result  * result;

  /* Data about the node itself. */
  fprintf(outfile, "Hash value: %lx\n", (unsigned long) node->key.hashval);
  hashposition_dump(&(node->key.hashpos), outfile);

  for (result = node->results; result != NULL; result = result->next) {
    read_result_dump(result, outfile);
  }
  /* FIXME: Dump contents of data also. */
}


/*
 * Dump an ASCII representation of the contents of a Hashtable onto
 * the FILE outfile. 
 */

void
hashtable_dump(Hashtable *table, FILE *outfile)
{
  int i;
  Hashnode * hn;

  /* Data about the table itself. */
  fprintf(outfile, "Dump of hashtable\n");
  fprintf(outfile, "Total size: %d\n", table->num_nodes);
  fprintf(outfile, "Size of hash table: %d\n", table->hashtablesize);
  fprintf(outfile, "Number of positions in table: %d\n", table->free_node);

  /* Data about the contents. */
  for (i = 0; i < table->hashtablesize; ++i) {
    fprintf(outfile, "Bucket %5d: ", i);
    hn = table->hashtable[i];
    if (hn == NULL)
      fprintf(outfile, "empty");
    else
      while (hn) {
	hashnode_dump(hn, outfile);
	hn = hn->next;
      }
    fprintf(outfile, "\n");
  }
}


#if 0
/*
 * Dump an alternative representation of the contents of a Hashtable
 * onto the FILE outfile. This one is mainly useful if you have to
 * debug the hashtable implementation itself.
 */

static void
hashtable_dump2(Hashtable *table, FILE *outfile)
{
  int i;
  for (i = 0; i < table->hashtablesize; i++) {
    fprintf(outfile, "bucket %d: ", i);
    if (table->hashtable[i] == NULL)
      fprintf(outfile, "NULL\n");
    else
      fprintf(outfile, "%d\n", table->hashtable[i] - table->all_nodes);
  }

  for (i = 0; i < table->num_nodes; i++) {
    Hashnode *node = &(table->all_nodes[i]);
    if (node->results == NULL)
      continue;
    fprintf(outfile, "node %d: ", i);
    if (node->results == NULL)
      fprintf(outfile, "NULL ");
    else
      fprintf(outfile, "%d ", node->results - table->all_results);
    if (node->next == NULL)
      fprintf(outfile, "NULL\n");
    else
      fprintf(outfile, "%d\n", node->next - table->all_nodes);
  }
  
  for (i = 0; i < table->num_results; i++) {
    Read_result *result = &(table->all_results[i]);
    if (rr_get_status(*result) == 0)
      continue;
    fprintf(outfile, "result %d ", i);
    if (result->next == NULL)
      fprintf(outfile, "NULL ");
    else
      fprintf(outfile, "%d ", result->next - table->all_results);
    read_result_dump(result, outfile);
  }
}
#endif


/*
 * Initialize a hash table for a given total size and size of the
 * hash table.
 *
 * Return 0 if something went wrong.  Just now this means that there
 * wasn't enough memory available.
 */

static int
hashtable_init(Hashtable *table, 
	       int tablesize, int num_nodes, int num_results)
{
  /* Make sure the hash system is initialized. */
  hash_init();

  /* Allocate memory for the pointers in the hash table proper. */
  table->hashtablesize = tablesize;
  table->hashtable = (Hashnode **) malloc(tablesize * sizeof(Hashnode *));
  if (table->hashtable == NULL) {
    free(table);
    return 0;
  }

  /* Allocate memory for the nodes. */
  table->num_nodes = num_nodes;
  table->all_nodes = (Hashnode *) malloc(num_nodes * sizeof(Hashnode));
  if (table->all_nodes == NULL) {
    free(table->hashtable);
    free(table);
    return 0;
  }

  /* Allocate memory for the results. */
  table->num_results = num_results;
  table->all_results = (Read_result *) malloc(num_results 
					      * sizeof(Read_result));
  if (table->all_results == NULL) {
    free(table->hashtable);
    free(table->all_nodes);
    free(table);
    return 0;
  }

  /* Initialize the table and all nodes to the empty state . */
  hashtable_clear(table);

  return 1;
}


/*
 * Allocate a new hash table and return a pointer to it. 
 *
 * Return NULL if there is insufficient memory.
 */

static Hashtable *
hashtable_new(int tablesize, int num_nodes, int num_results)
{
  Hashtable  * table;

  /* Make sure the hash system is initialized. */
  hash_init();

  /* Allocate the hashtable struct. */
  table = (Hashtable *) malloc(sizeof(Hashtable));
  if (table == NULL)
    return NULL;

  /* Initialize the table. */
  if (!hashtable_init(table, tablesize, num_nodes, num_results)) {
    free(table);
    return NULL;
  }

  return table;
}


/*
 * Clear an existing hash table.  
 */

static void
hashtable_clear(Hashtable *table)
{
  int bucket;
  int i;
  
  if (!table)
    return;
  
  /* Initialize all hash buckets to the empty list. */
  for (bucket = 0; bucket < table->hashtablesize; ++bucket)
    table->hashtable[bucket] = NULL;

  /* Mark all read_results as free. */
  for (i = 0; i < table->num_results; i++)
    table->all_results[i].result_ri_rj = 0;
  
  /* Mark all nodes as free. */
  for (i = 0; i < table->num_nodes; i++)
    table->all_nodes[i].results = NULL;
  
  table->free_node = 0;
  table->free_result = 0;
}


/*
 * Unlink the closed results from the linked list of results at a node.
 */

static void
hashtable_unlink_closed_results(Hashnode *node, 
				int exclusions, unsigned int stackplimit,
				int statistics[][20])
{
  Read_result *previous_result = NULL;
  Read_result *current_result = node->results;
  
  while (current_result != NULL) {
    int stackp;

    stackp = rr_get_stackp(*current_result);
    if (stackp > 19)
      stackp = 19;
    statistics[rr_get_routine(*current_result)][stackp]++;

    if (rr_get_status(*current_result) == 2
	&& ((1 << rr_get_routine(*current_result)) & exclusions) == 0
	&& rr_get_stackp(*current_result) >= stackplimit) {
      if (previous_result == NULL)
	node->results = current_result->next;
      else
	previous_result->next = current_result->next;
      current_result->result_ri_rj = 0;
    }
    else
      previous_result = current_result;

    current_result = current_result->next;
  }
}



/*
 * Clear an existing hash table except for open nodes.
 *
 * Don't even think about compressing the node and results arrays
 * afterwards in order to simplify distribution of new nodes and
 * results. The read result pointers out in reading.c and owl.c will
 * never know that you moved them around.
 */

static void
hashtable_partially_clear(Hashtable *table)
{
  Hashnode *node;
  int bucket;
  Hashnode *previous;
  Hashnode *current;
  int k, l;
  
  int statistics[NUM_ROUTINES][20];

  DEBUG(DEBUG_READING_PERFORMANCE, "Hashtable cleared because it was full.\n");

  for (k = 0; k < NUM_ROUTINES; ++k)
    for (l = 0; l < 20; ++l)
      statistics[k][l] = 0;

  /* Walk through all_nodes. Closed nodes are unlinked from the
   * linked lists and marked as free.
   */
  for (k = 0; k < table->num_nodes; k++) {
    node = &(table->all_nodes[k]);
    bucket = node->key.hashval % table->hashtablesize;
    previous = NULL;
    current = table->hashtable[bucket];

    /* If there are no results attached, this node is not in the table. */
    if (node->results == NULL)
      continue;

    /* Remove all closed results for this node except OWL_{ATTACK,DEFEND}. */
    hashtable_unlink_closed_results(node, 
				    (1 << OWL_ATTACK | 1 << OWL_DEFEND), 3,
				    statistics);
    if (node->results != NULL)
      continue;

    /* Find the node in the linked list and unlink. */
    while (current != NULL) {
      if (current != node) {
	previous = current;
	current = current->next;
      }
      else {
	if (previous == NULL)
	  table->hashtable[bucket] = current->next;
	else
	  previous->next = current->next;
	break;
      }
    }
  }

  if (debug & DEBUG_READING_PERFORMANCE) {
    /* FIXME: These names should be where the constants are defined. */
    const char *routines[] = {
      "find_defense", "defend1",    "defend2", "defend3",
      "defend4",      "attack",     "attack2", "attack3",
      "owl_attack",   "owl_defend", "",        "",
      "",             "",           "",        "",
    };
    int total;

    fprintf(stderr, "routine        total     0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19\n");

    for (k = 0; k < NUM_ROUTINES; ++k) {
      total = 0; 
      for (l = 0; l < 20; ++l)
	total += statistics[k][l];

      if (total == 0)
	continue;

      fprintf(stderr, "%-14s%6d", routines[k], total);
      for (l = 0; l < 20; ++l)
	fprintf(stderr, "%6d", statistics[k][l]);
      fprintf(stderr, "\n");
    }
  }


  /* FIXME: This is not entirely safe although it probably works more
   * than 99.999% of all cases.  If result no 0 or node no 0 is not
   * free after the partial clearing, this will explode into our face. */
  table->free_result = 0;
  table->free_node = 0;
}


/*
 * Enter a position with a given hash value into the table.  Return 
 * a pointer to the hash node where it was stored.  If it is already
 * there, don't enter it again, but return a pointer to the old one. */

static Hashnode *
hashtable_enter_position(Hashtable *table, Hash_data *hd)
{
  Hashnode  * node;
  int         bucket;

  /* If the position is already in the table, return a pointer to it. */
  node = hashtable_search(table, hd);
  if (node != NULL) {
    return node;
  }

  /* If the next node is not free, skip until we find one which is free. */
  while (table->free_node < table->num_nodes
	 && table->all_nodes[table->free_node].results != NULL)
    table->free_node++;
  
  /* If the table is full, return NULL */
  if (table->free_node == table->num_nodes)
    return NULL;

  /* It wasn't there and there is still room.  Allocate a new node for it... */
  node = &(table->all_nodes[table->free_node++]);
  node->key = *hd;
  node->results = NULL;

  /* ...and enter it into the table. */
  bucket = hd->hashval % table->hashtablesize;
  node->next = table->hashtable[bucket];
  table->hashtable[bucket] = node;

  stats.position_entered++;
  return node;
}


/* 
 * Given a Hashposition and a Hash value, find the hashnode which contains
 * this very position with the given hash value.  
 *
 * We could compute the hash value within this functions, but later
 * when we have incremental calculation of the hash function, this 
 * would be dumb. So we demand the hash value from outside from the 
 * very beginning.
 */

static Hashnode *
hashtable_search(Hashtable *table, Hash_data *hd)
{
  Hashnode     * node;
  int            bucket;

  bucket = hd->hashval % table->hashtablesize;
  for (node = table->hashtable[bucket]; node != NULL; node = node->next) {
    if (node->key.hashval != hd->hashval)
      continue;
    if (hashposition_compare(&hd->hashpos, &node->key.hashpos) == 0)
      break;
  }
  return node;
}


/* 
 * Search the result list in a hash node for a particular result. This
 * result is from ROUTINE (e.g. readlad1) at (i, j) and reading depth
 * stackp.
 *
 * All these numbers must be unsigned, and 0 <= x <= 255).
 */

static Read_result *
hashnode_search(Hashnode *node, int routine, 
		int komaster, int kom_i, int kom_j, int i, int j)
{
  Read_result  * result;
  unsigned int   search_for;

  search_for = rr_compress_data(rr, routine, komaster, kom_i, kom_j,
				i, j, stackp);

  for (result = node->results; result != NULL; result = result->next) {
    if (result->compressed_data == search_for)
      break;
    }

  return result;
}


/*
 * Enter a new Read_result into a Hashnode.
 * We already have the node, now we just want to enter the result itself.
 * We will fill in the result itself later, so we only need the routine
 * number for now.
 */

static Read_result *
hashnode_new_result(Hashtable *table, Hashnode *node, int routine, 
		    int komaster, int kom_i, int kom_j, int i, int j)
{
  Read_result  * result;

  /* If the next result is not free, skip until we find one which is free. */
  while (table->free_result < table->num_results
	 && rr_get_status(table->all_results[table->free_result]) != 0)
    table->free_result++;
  
  /* If the table is full, return NULL */
  if (table->free_result == table->num_results)
    return NULL;

  /* There is still room. Allocate a new node for it... */
  result = &(table->all_results[table->free_result++]);

  /* ...and enter it into the table. */
  result->next = node->results;
  node->results = result;

  /* Now, put the routine number into it. */
  result->compressed_data = rr_compress_data(rr, routine, komaster,
					     kom_i, kom_j, i, j, stackp);

  /* And set the status to open. */
  result->result_ri_rj = 1 << 24;
  
  stats.read_result_entered++;
  return result;

}

/* Initialize the cache for read results, using at most the given
 * number of bytes of memory. If the memory isn't sufficient to
 * allocate a single node or if the allocation fails, the caching is
 * disabled.
 */
void
reading_cache_init(int bytes)
{
  /* Initialize hash table.
   *
   * The number 1.4 below is the quotient between the number of nodes
   * and the number of read results.  It was found in a test that this 
   * number varies between 1.15 and 1.4.  Thus we use 1.4.
   */
  float nodes = ((float) bytes
		 / (1.5 * sizeof(Hashnode *)
		    + sizeof(Hashnode)
		    + 1.4 * sizeof(Read_result)));
  /* If we get a zero size hash table, disable hashing completely. */
  if (nodes < 1.0)
    hashflags = HASH_NOTHING;
  movehash = hashtable_new((int) (1.5 * nodes),  /* table size   */
			   (int) nodes,          /* nodes        */
			   (int) (1.4 * nodes)); /* read results */
  
  if (!movehash) {
    fprintf(stderr, "Warning: failed to allocate hashtable, caching disabled.\n");
      hashflags = HASH_NOTHING;
  }
}

/* Clear the cache for read results. */
void
reading_cache_clear()
{
  hashtable_clear(movehash);
}

/*
 * Return a Read_result for the current position, routine and location.
 * For performance, the location is changed to the origin of the string.
 */

int
get_read_result(int routine, int komaster, int kom_i, int kom_j,
		int *si, int *sj, Read_result **read_result)
{
  int result;
  /* Only store the result if stackp <= depth. Above that, there
   * is no branching, so we won't gain anything.
   */
  if (stackp > depth) {
    *read_result = NULL;
    return 0;
  }
  
  /* Find the origin of the string containing (si, sj),
   * in order to make the caching of read results work better.
   */
  find_origin(*si, *sj, si, sj);
  
  result = do_get_read_result(routine, komaster, kom_i, kom_j,
			      si, sj, read_result);
  if (*read_result == NULL) {
    /* Clean up the hashtable and try once more. */
    hashtable_partially_clear(movehash);
    result = do_get_read_result(routine, komaster, kom_i, kom_j, si, sj,
				read_result);
  }
  return result;
}

static int
do_get_read_result(int routine, int komaster, int kom_i, int kom_j,
		   int *si, int *sj, Read_result **read_result)
{
  Hashnode      * hashnode;
  int             retval;

#if CHECK_HASHING
  Hash_data    key;

  /* Check the hash table to see if we have had this position before. */
  hashdata_recalc(&key, p, ko_i, ko_j);
  gg_assert(hashdata_diff_dump(&key, &hashdata) == 0);

  /* Find this position in the table.  If it wasn't found, enter it. */
  hashnode = hashtable_search(movehash, &hashdata);
  if (hashnode != NULL) {
    stats.position_hits++;
    RTRACE("We found position %H in the hash table...\n",
	   (unsigned long) hashdata.hashval);
  } else {
    hashnode = hashtable_enter_position(movehash, &hashdata);
    if (hashnode)
      RTRACE("Created position %H in the hash table...\n",
	     (unsigned long) hashdata.hashval);
  }
#else
  /* Find this position in the table.  If it wasn't found, enter it. */
  hashnode = hashtable_search(movehash, &hashdata);
  if (hashnode != NULL) {
    stats.position_hits++;
    RTRACE("We found position %H in the hash table...\n",
	   (unsigned long) hashdata.hashval);
  } else {
    hashnode = hashtable_enter_position(movehash, &hashdata);
    if (hashnode)
      RTRACE("Created position %H in the hash table...\n",
	     (unsigned long) hashdata.hashval);
  }
#endif

  retval = 0;
  if (hashnode == NULL) {
    /* No hash node, so we can't enter a result into it. */
    *read_result = NULL;

  } else {

    /* We found it!  Now see if we can find a previous result. */
    *read_result = hashnode_search(hashnode, routine, komaster, kom_i, kom_j,
				   *si, *sj);

    if (*read_result != NULL) {
      stats.read_result_hits++;
      retval = 1;
    } else {
      RTRACE("...but no previous result for routine %d and (%m)...",
	     routine, *si, *sj);

      *read_result = hashnode_new_result(movehash, hashnode, routine,
					 komaster, kom_i, kom_j, *si, *sj);
      
      if (*read_result == NULL)
	RTRACE("%o...and unfortunately there was no room for one.\n");
      else
	RTRACE("%o...so we allocate a new one.\n");
    }
  }

  return retval;
}


/* Write reading trace data to an SGF file. Normally called through the
 * macro SGFTRACE in cache.h.
 */

void
sgf_trace(const char *func, int si, int sj, int i, int j,
	  int result, const char *message)
{
  char buf[100];

  sprintf(buf, "%s %c%d: ", func, sj + 'A' + (sj>=8), board_size - si);
  
  if (result == 0)
    sprintf(buf + strlen(buf), "0");
  else if (ON_BOARD(i, j))
    sprintf(buf + strlen(buf), "%s %c%d", result_to_string(result), 
	    j + 'A' + (j>=8),
	    board_size - i);
  else if (i == -1 && j == -1)
    sprintf(buf + strlen(buf), "%s PASS", result_to_string(result));
  else
    sprintf(buf + strlen(buf), "%s [%d,%d]", result_to_string(result), i, j);

  if (message)
    sprintf(buf + strlen(buf), " (%s)", message);
  
  sgftreeAddComment(sgf_dumptree, NULL, buf);
}  

/*
 * Local Variables:
 * tab-width: 8
 * c-basic-offset: 2
 * End:
 */


