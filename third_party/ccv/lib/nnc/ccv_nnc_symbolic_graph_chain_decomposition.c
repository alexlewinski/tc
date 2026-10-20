#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

// Implement the new method for exec_dep. We use chain decomposition such that each node only needs to log which chain and at which node to be dependent on.
ccv_nnc_exec_dep_t ccv_nnc_exec_dep_new(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_visit_t* const visit)
{
	const int exec_symbol_info_size = graph->exec_symbol_info->rnum;
	int* chain_ids = ccmalloc(sizeof(int) * exec_symbol_info_size * 2);
	int* chain_pos = chain_ids + exec_symbol_info_size;
	int* buf = (int*)ccmalloc(sizeof(int) * exec_symbol_info_size);
	int* reversed_depth = buf;
	const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
	int i, j;
	// Go reverse order to generate the distance from sink.
	ccv_nnc_graph_visit_for_reversed(visit, exec_symbol_info, node, idx, term) {
		if (node->flags & CCV_NNC_GRAPH_EXEC_DEAD)
			continue;
		chain_ids[idx] = -1;
		if (!node->outgoings || node->outgoings->rnum == 0)
		{
			reversed_depth[idx] = 0;
			continue;
		}
		const int outgoing = *(int*)ccv_array_get(node->outgoings, 0);
		int depth = reversed_depth[outgoing];
		for (i = 1; i < node->outgoings->rnum; i++)
		{
			const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			depth = ccv_max(depth, reversed_depth[outgoing]);
		}
		reversed_depth[idx] = depth + 1;
	} ccv_nnc_graph_visit_endfor
	// Go in order to generate chain ids (if there are multiple exits, we use the reverse depth to break the tie).
	// Note that we cannot use depth so-far because then multiple exit nodes are equally good to "inherit" the chain selection.
	int chain_count = 0;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx, term) {
		if (node->flags & CCV_NNC_GRAPH_EXEC_DEAD)
			continue;
		int chain_id = chain_ids[idx];
		if (chain_ids[idx] < 0)
		{
			chain_id = chain_count;
			chain_ids[idx] = chain_id;
			chain_pos[idx] = 1; // The first one in this chain. 1-based index because in sparse matrix, 0 is the default value.
			chain_count += 1;
		}
		if (!node->outgoings || node->outgoings->rnum == 0)
			continue;
		int depth = -1;
		int next_idx = -1;
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			if (chain_ids[outgoing] < 0 && reversed_depth[outgoing] > depth)
				depth = reversed_depth[outgoing], next_idx = outgoing;
		}
		if (next_idx >= 0)
		{
			chain_ids[next_idx] = chain_id;
			assert(reversed_depth[idx] - depth >= 1);
			chain_pos[next_idx] = chain_pos[idx] + (reversed_depth[idx] - depth);
		}
	} ccv_nnc_graph_visit_endfor
	if (exec_symbol_info_size < chain_count * 3) // Be more conservative on RAM usage.
		buf = ccrealloc(buf, sizeof(int) * chain_count * 3);
	ccv_sparse_matrix_t* deps = ccv_sparse_matrix_new(graph->exec_symbol_info->rnum, chain_count, CCV_32S | CCV_C2, CCV_SPARSE_ROW_MAJOR, 0);
	// It logs which pos on that chain we depend on. We can simply compare that with the chain_pos for a node to know if they are ancestors.
#define for_block(x, val) \
	do { \
		if (((int32_t*)val)[0] > 0) \
		{ \
			buf[buf_size * 3] = x; \
			buf[buf_size * 3 + 1] = ((int32_t*)val)[0]; \
			buf[buf_size * 3 + 2] = ((int32_t*)val)[1] + 1; \
			++buf_size; \
		} \
	} while (0)
	int buf_size;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx, term) {
		if (node->flags & CCV_NNC_GRAPH_EXEC_DEAD)
			continue;
		buf_size = 0; /* save all its parent deps to this buffer */
		ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(deps, idx);
		if (vector)
			CCV_SPARSE_VECTOR_FOREACH(deps, vector, for_block);
		if (!node->outgoings)
			continue;
		const int chain_id = chain_ids[idx];
		const int pos = chain_pos[idx];
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			const int outgoing_chain_id = chain_ids[outgoing];
			if (outgoing_chain_id != chain_id)
			{
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(deps, outgoing, chain_id);
				/* If not found, set, if the current node is the destination node, no need 
				 * set itself as parent of subsequent nodes because its terminal nature. */
				if (!cell.i32 || cell.i32[0] == 0 || cell.i32[0] < pos)
				{
					int p[2] = { pos, 1 };
					ccv_set_sparse_matrix_cell(deps, outgoing, chain_id, &p);
				}
			}
			if (buf_size > 0)
			{
				ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(deps, outgoing);
				for (j = 0; j < buf_size; j++) /* set with all idx's dependencies as well */
				{
					if (outgoing_chain_id == buf[j * 3]) // We don't need to add as dependency for the same chain.
						continue;
					if (!vector)
					{
						ccv_set_sparse_matrix_cell(deps, outgoing, buf[j * 3], &buf[j * 3 + 1]);
						vector = ccv_get_sparse_matrix_vector(deps, outgoing);
						continue;
					}
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3]);
					/* If not found, set. Otherwise, set to the latest one only if it is later. */
					if (!cell.i32)
						ccv_set_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3], &buf[j * 3 + 1]);
					else if (cell.i32[0] == 0 || cell.i32[0] < buf[j * 3 + 1])
						ccv_set_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3], &buf[j * 3 + 1]);
					else if (cell.i32[0] == buf[j * 3 + 1]) { // If we point to the same one, use the longest.
						int p[2] = { cell.i32[0], ccv_max(buf[j * 3 + 2], cell.i32[1]) };
						ccv_set_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3], &p);
					}
				}
			}
		}
	} ccv_nnc_graph_visit_endfor
#undef for_block
	ccfree(buf);
	ccv_nnc_exec_dep_t exec_dep = {
		.chain_ids = chain_ids,
		.chain_pos = chain_pos,
		.deps = deps
	};
	return exec_dep;
}

void ccv_nnc_exec_dep_free(const ccv_nnc_exec_dep_t exec_dep)
{
	ccfree(exec_dep.chain_ids);
	ccv_matrix_free(exec_dep.deps);
}
