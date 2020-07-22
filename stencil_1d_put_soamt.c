#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>
#include <infiniband/verbs.h>

#include "shared.h"

/**
 * The dimension of dividing up the mesh is vertical.
 * In this case, each WQE represents one element.
 * The ranks will exchange data with RDMA-writes.
 * The total number of the threads on a node has to be less than 16.
 * The first half of the ranks is on rank 0 and the second half on rank 1.
 */

int init_params(void);

int read_args(int argc, char *argv[]);

static int put_halo(int direction, int num_rows, int hz_halo_dim, int element_bytes, int hz_tile_dim,
					unsigned long int local_source_addr, int local_key,
					unsigned long int remote_dest_addr, int remote_key,
					struct stencil_thread_flow_vars *flow_vars, struct stencil_thread_flow_vars *flow_vars_arr);

static int wait(struct stencil_thread_flow_vars *flow_vars, struct stencil_thread_flow_vars *flow_vars_arr);

static int init_ep_res(void);

static int connect_eps(int left_rank, int right_rank);

static int free_ep_res(void);

struct ibv_context *dev_context;
struct ibv_pd *left_pd;
struct ibv_pd *right_pd;
struct ibv_mr *left_tile_mr;
struct ibv_mr *right_tile_mr;
struct ibv_cq *cq;
struct ibv_qp *left_qp;
struct ibv_qp *right_qp;

int left_rem_tx_depth;
int right_rem_tx_depth;

int main (int argc, char *argv[])
{
	int ret = 0, provided;

	int tile_dim_x, tile_dim_y;
	int left_rank, right_rank;
	int left_rkey, right_rkey;
	int rows_per_thread;	
	int element_bytes;

	double *local_tile;

	double start_time, total_time;

	unsigned long int local_left_source_addr;
	unsigned long int local_right_source_addr;

	unsigned long int local_left_halo_addr;
	unsigned long int local_right_halo_addr;
	unsigned long int remote_left_halo_addr;
	unsigned long int remote_right_halo_addr;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE) {
		printf("Insufficient threading support\n");
		ret = EXIT_FAILURE;
		goto clean;
	}
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size % 2) {
		fprintf(stderr, "Supporting only multiples of two at the moment since we have only two nodes\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ret = init_params();
	if (ret) {
		fprintf(stderr, "Error in initializing paramaters\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}

	ret = read_args(argc, argv);
	if (ret)
		goto clean_mpi;

	if (ga_dim_x % size) {
		fprintf(stderr, "The number of ranks has to be a factor of the global horizontal dimension\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}

	if (size / 2 * num_threads > 16) {
		fprintf(stderr, "Total number of threads per node has to be less than 16, the number of cores per socket\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}

	if (ga_dim_y % num_threads) {
		fprintf(stderr, "Number of threads per rank has to be a factor of the global vertical dimension\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}

	omp_set_num_threads(num_threads);
	
	rows_per_thread = ga_dim_y / num_threads;

	tile_dim_y = ga_dim_y;
	tile_dim_x = ga_dim_x / size;

	left_rank = (rank == 0) ? size - 1 : rank - 1;
	right_rank = (rank + 1) % size;

	ret = init_ep_res();
	if (ret) {
		fprintf(stderr, "Failure in initializing EP's resources.\n");
		goto clean_mpi;
	}

	ret = connect_eps(left_rank, right_rank);
	if (ret) {
		fprintf(stderr, "Failure in connecting eps.\n");
		goto clean_mpi;
	}

	int tile_size = (tile_dim_y + 2 /* zero padding */) * (tile_dim_x + 2 /* halo */);
	local_tile = calloc(tile_size, sizeof *local_tile); 
	if (!local_tile) {
		fprintf(stderr, "Failure in allocating the tile of the process.\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}
	
	local_left_source_addr 	= (unsigned long int) &local_tile[(tile_dim_x + 2) + 1];
	local_left_halo_addr 	= (unsigned long int) &local_tile[(tile_dim_x + 2) + 0];
	local_right_source_addr	= (unsigned long int) &local_tile[(tile_dim_x + 2) + tile_dim_x];
	local_right_halo_addr 	= (unsigned long int) &local_tile[(tile_dim_x + 2) + tile_dim_x + 1]; 

	left_tile_mr = ibv_reg_mr(left_pd, local_tile, tile_size * sizeof *local_tile, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
	right_tile_mr = ibv_reg_mr(right_pd, local_tile, tile_size * sizeof *local_tile, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
	if (!left_tile_mr || !right_tile_mr) {
		fprintf(stderr, "Failure in allocating MRs for the halo regions.\n");
		ret = EXIT_FAILURE;
		goto clean_mpi;
	}

	if (rank % 2) {
		MPI_Send(&left_tile_mr->rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&left_rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		MPI_Send(&right_tile_mr->rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&right_rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Send(&local_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&remote_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Send(&local_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&remote_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	} else {
		MPI_Recv(&right_rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&right_tile_mr->rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);

		MPI_Recv(&left_rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&left_tile_mr->rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);

		MPI_Recv(&remote_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&local_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD);

		MPI_Recv(&remote_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&local_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	struct stencil_thread_flow_vars *flow_vars;

	element_bytes = sizeof(double); // TODO: keep it configurable
	
	flow_vars = calloc(num_threads, sizeof *flow_vars);

	#pragma omp parallel private(ret)
	{
		int tid;
		int p;
		
		tid = omp_get_thread_num();

		struct ibv_sge *SGE;
		struct ibv_send_wr *send_wqe;
		struct ibv_wc *WC;
		
		posix_memalign((void**) &SGE, CACHE_LINE_SIZE, postlist * sizeof(struct ibv_sge));
		posix_memalign((void**) &send_wqe, CACHE_LINE_SIZE, postlist * sizeof(struct ibv_send_wr));
		posix_memalign( (void**) &WC, CACHE_LINE_SIZE, cq_depth * sizeof(struct ibv_wc) );

		memset(SGE, 0, postlist * sizeof(struct ibv_sge));
		memset(send_wqe, 0, postlist * sizeof(struct ibv_send_wr));
		memset(WC, 0, cq_depth * sizeof(struct ibv_wc));

		for (p = 0; p < postlist; p++) {
			send_wqe[p].wr_id = tid << 3;
			send_wqe[p].next = (p == postlist-1) ? NULL: &send_wqe[p+1];
			send_wqe[p].sg_list = &SGE[p];
			send_wqe[p].num_sge = 1;
		}

		struct ibv_qp_attr attr;
		struct ibv_qp_init_attr qp_init_attr;
		ret = ibv_query_qp(left_qp, &attr, IBV_QP_CAP, &qp_init_attr);
		if (ret) {
			fprintf(stderr, "Failure in querying the left QP\n");
			exit(0);
		}
		/* Assuming the max inline data is the same for both the left and right QPs */

		flow_vars[tid].tid = tid;
		flow_vars[tid].left_post_count = 0;
		flow_vars[tid].right_post_count = 0;
		flow_vars[tid].left_comp_count = 0;
		flow_vars[tid].right_comp_count = 0;
		flow_vars[tid].left_posts = 0;
		flow_vars[tid].right_posts = 0;
		flow_vars[tid].postlist = postlist;
		flow_vars[tid].mod_comp = mod_comp;
		flow_vars[tid].tx_depth = tx_depth;
		flow_vars[tid].cq_depth = cq_depth;
		flow_vars[tid].max_inline_data = qp_init_attr.cap.max_inline_data;
		flow_vars[tid].tx_decrement_val = max(postlist, mod_comp);
		flow_vars[tid].sge = SGE;
		flow_vars[tid].wqe = send_wqe;
		flow_vars[tid].wc = WC;
		flow_vars[tid].left_qp = left_qp;
		flow_vars[tid].right_qp = right_qp;
		flow_vars[tid].my_cq = cq;
	}
	
	int bytes_per_thread = rows_per_thread * (tile_dim_x + 2) * element_bytes;
	left_rem_tx_depth = tx_depth;
	right_rem_tx_depth = tx_depth;

	MPI_Barrier(MPI_COMM_WORLD);
	start_time = MPI_Wtime();	

	/* Stencil */
	int iter;
	for (iter = 0; iter < iterations; iter++) {

		#pragma omp parallel private(ret) firstprivate(local_left_source_addr, local_right_source_addr, remote_left_halo_addr, remote_right_halo_addr, bytes_per_thread, rows_per_thread, element_bytes, tile_dim_x, left_tile_mr, right_tile_mr, left_rkey, right_rkey)
		{
			int p;
			int tid = omp_get_thread_num();
			int my_i = tid * rows_per_thread + 1;
			
			unsigned long int my_left_source_addr 	= local_left_source_addr 	+ tid * bytes_per_thread; // &local_tile[my_i * (tile_dim_x + 2) + 1]
			unsigned long int my_right_source_addr	= local_right_source_addr	+ tid * bytes_per_thread; // &local_tile[my_i * (tile_dim_x + 2) + tile_dim_x]
			unsigned long int my_left_remote_addr 	= remote_left_halo_addr 	+ tid * bytes_per_thread;
			unsigned long int my_right_remote_addr 	= remote_right_halo_addr	+ tid * bytes_per_thread;
			
			/* Remove the previous direction in the WQE IDs */
			for (p = 0; p < postlist; p++) {
				flow_vars[tid].wqe[p].wr_id = tid << 3;
			}
			
			/* Send halo region to left neighbor */
			ret = put_halo(LEFT, rows_per_thread, 1, element_bytes, tile_dim_x+2,
					 my_left_source_addr, left_tile_mr->lkey,
					 my_left_remote_addr, left_rkey,
					 &flow_vars[tid], flow_vars);
			#ifdef ERRCHK
			if (ret) {
				fprintf(stderr, "Error in putting halo to the right\n");
				exit(0);
			}
			#endif
			//printf("Done putting to the left\n");
			
			/* Remove the previous direction in the WQE IDs */
			for (p = 0; p < postlist; p++) {
				flow_vars[tid].wqe[p].wr_id = tid << 3;
			}

			/* Send halo region to right neighbor */
			ret = put_halo(RIGHT, rows_per_thread, 1, element_bytes, tile_dim_x+2,
					 my_right_source_addr, right_tile_mr->lkey,
					 my_right_remote_addr, right_rkey, 
					 &flow_vars[tid], flow_vars);
			#ifdef ERRCHK
			if (ret) {
				fprintf(stderr, "Error in putting halo to the right\n");
				exit(0);
			}
			#endif
			//printf("Done putting to the right\n");

			/* Complete */	
			ret = wait(&flow_vars[tid], flow_vars);
			#ifdef ERRCHK
			if (ret) {
				fprintf(stderr, "Error in wating for puts to complete\n");
				exit(0);
			}
			#endif
			//printf("Done putting waiting\n");
			
			/* Update grid */
			if (compute) {
				int i, j;
				for (i = my_i; i < my_i + rows_per_thread; i++) {
					for (j = 1; j < tile_dim_x + 1; j++) {
						local_tile[i * (tile_dim_x + 2) + j] += local_tile[(i - 1) * (tile_dim_x + 2) + j] + /* top */
																local_tile[(i + 1) * (tile_dim_x + 2) + j] + /* bottom */
																local_tile[i * (tile_dim_x + 2) + j - 1] + /* left */
																local_tile[i * (tile_dim_x + 2) + j + 1];  /* right */ 
					}
				}
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	total_time = MPI_Wtime() - start_time;
	if (rank == 0) {
		if (!compute) {
			int total_write_messages = 2 * tile_dim_y * iterations * size;
			double write_mr = (double) total_write_messages / total_time / 1e6;
			printf("%-10s\t%-10s\t%-10s\n", "PPN", "Threads", "Write Mmsgs/s");
			printf("%-10d\t%-10d\t%-10.2f\n", size/2, num_threads, write_mr);
		} else {
			printf("%-10s\t%-10s\t%-10s\n", "PPN", "Threads", "Time (s)");
			printf("%-10d\t%-10d\t%-10.2f\n", size/2, num_threads, total_time);
		}
	}

	ret = free_ep_res();
	if (ret)
		fprintf(stderr, "Failure in freeing resources\n");

	free(local_tile);

clean_mpi:
	MPI_Finalize();
clean:
	return ret;
}

static int put_halo(int direction, int num_rows, int hz_halo_dim, int element_bytes, int hz_tile_dim,
					unsigned long int local_source_addr, int local_key,
					unsigned long int remote_dest_addr, int remote_key,
					struct stencil_thread_flow_vars *flow_vars, struct stencil_thread_flow_vars *flow_vars_arr)
{
	int ret = 0;
	int p;
	int cqe_count;
	int send_inline;
	int cur_rem_tx_depth;
	int bytes_in_wqe, bytes_in_row;
	uint64_t local_base_addr, remote_base_addr;
	struct ibv_send_wr *bad_send_wqe;
	int cqe_i, comp_tid, comp_direction;

	/* Active variables */
	int posts = 0;
	int *rem_tx_depth = NULL;
	int *post_count = NULL;
	struct ibv_qp *qp = NULL;
	
	switch (direction) {
		case LEFT:
			flow_vars->left_posts += num_rows;
			posts = flow_vars->left_posts;
			post_count = &flow_vars->left_post_count;
			qp = flow_vars->left_qp;
			rem_tx_depth = &left_rem_tx_depth;
			break;
		case RIGHT:
			flow_vars->right_posts += num_rows;
			posts = flow_vars->right_posts;
			post_count = &flow_vars->right_post_count;
			qp = flow_vars->right_qp;
			rem_tx_depth = &right_rem_tx_depth;
			break;
		#ifdef ERRCHK
		default:
			ret = EXIT_FAILURE;
			goto exit;
		#endif
	}
	
	bytes_in_wqe = hz_halo_dim * element_bytes;
	bytes_in_row = hz_tile_dim * element_bytes;

	send_inline = (bytes_in_wqe <= flow_vars->max_inline_data) ? IBV_SEND_INLINE : 0;

	for (p = 0; p < flow_vars->postlist; p++) {
		flow_vars->sge[p].length	= bytes_in_wqe;
		flow_vars->sge[p].lkey		= local_key;

		flow_vars->wqe[p].wr_id |= direction;
		flow_vars->wqe[p].opcode = IBV_WR_RDMA_WRITE;
		flow_vars->wqe[p].send_flags = send_inline;
		flow_vars->wqe[p].wr.rdma.rkey = remote_key;
	}

	local_base_addr = (uint64_t) local_source_addr;
	remote_base_addr = (uint64_t) remote_dest_addr;
				
	while (*post_count < posts) {
		#pragma omp atomic capture
		{cur_rem_tx_depth = *rem_tx_depth; *rem_tx_depth -= flow_vars->tx_decrement_val;}
		if (cur_rem_tx_depth <= 0) {
			#pragma omp atomic
			*rem_tx_depth += flow_vars->tx_decrement_val;
			goto poll;
		}
		do {
			for (p = 0; p < flow_vars->postlist; p++) {
				flow_vars->sge[p].addr 	= local_base_addr;
				
				if (((*post_count)+p+1) % flow_vars->mod_comp == 0)
					flow_vars->wqe[p].send_flags = IBV_SEND_SIGNALED;
				else
					flow_vars->wqe[p].send_flags = 0;
				flow_vars->wqe[p].send_flags |= send_inline;
				flow_vars->wqe[p].wr.rdma.remote_addr = remote_base_addr;
				
				local_base_addr += bytes_in_row;
				remote_base_addr += bytes_in_row; 
			}
			ret = ibv_post_send(qp, &flow_vars->wqe[0], &bad_send_wqe);
			#ifdef ERRCHK
			if (ret) {
				fprintf(stderr, "Thread %d: Error %d in posting send_wqe on QP\n", flow_vars->tid, ret);
				goto exit;
			}
			#endif
			*post_count += flow_vars->postlist;
		} while (*post_count % flow_vars->mod_comp);
		if (cur_rem_tx_depth <= 0) {
poll:
			// Poll only if SQ is full
			cqe_count = ibv_poll_cq(flow_vars->my_cq, flow_vars->cq_depth, flow_vars->wc);
			#ifdef ERRCHK
			if (cqe_count < 0) {
				fprintf(stderr, "Thread %d: Failure in polling CQ: %d\n", flow_vars->tid, cqe_count);
				ret = cqe_count;
				goto exit;
			}
			#endif
			for (cqe_i = 0; cqe_i < cqe_count; cqe_i++) {
				#ifdef ERRCHK
				if (flow_vars->wc[cqe_i].status != IBV_WC_SUCCESS) {
					fprintf(stderr, "Thread %d: Failed status %s for %d; cqe_count %d\n", flow_vars->tid,
							ibv_wc_status_str(flow_vars->wc[cqe_i].status), (int) flow_vars->wc[cqe_i].wr_id, cqe_i);
					ret = EXIT_FAILURE;
					goto exit;
				}
				#endif
				comp_tid = flow_vars->wc[cqe_i].wr_id >> 3; // tid is after the first 3 bits
				comp_direction = flow_vars->wc[cqe_i].wr_id & 0x7; // extract the first 3 bits
				switch(comp_direction) {
					case LEFT:
						#pragma omp atomic
						flow_vars_arr[comp_tid].left_comp_count += flow_vars_arr[comp_tid].mod_comp;
						#pragma omp atomic
						left_rem_tx_depth += flow_vars_arr[comp_tid].mod_comp;
						break;
					case RIGHT:
						#pragma omp atomic
						flow_vars_arr[comp_tid].right_comp_count += flow_vars_arr[comp_tid].mod_comp;
						#pragma omp atomic
						right_rem_tx_depth += flow_vars_arr[comp_tid].mod_comp;
						break;
				}
			}
		}
	}

#ifdef ERRCHK
exit:
#endif
	return ret;
}

static int wait(struct stencil_thread_flow_vars *flow_vars, struct stencil_thread_flow_vars *flow_vars_arr)
{
	int ret = 0;
	int cqe_count;
	int cqe_i;
	int comp_tid, comp_direction;

	while ((flow_vars->left_comp_count < flow_vars->left_post_count) || (flow_vars->right_comp_count < flow_vars->right_post_count)) {
		cqe_count = ibv_poll_cq(flow_vars->my_cq, flow_vars->cq_depth, flow_vars->wc);
		#ifdef ERRCHK
		if (cqe_count < 0) {
			fprintf(stderr, "Thread %d: Failure in polling CQ: %d\n", flow_vars->tid, cqe_count);
			ret = cqe_count;
			goto exit; 
		}
		#endif
		for (cqe_i = 0; cqe_i < cqe_count; cqe_i++) {
			#ifdef ERRCHK
			if (flow_vars->wc[cqe_i].status != IBV_WC_SUCCESS) {
				fprintf(stderr, "Thread %d: Failed status %s for %d; cqe_count %d\n", flow_vars->tid,
						ibv_wc_status_str(flow_vars->wc[cqe_i].status), (int) flow_vars->wc[cqe_i].wr_id, cqe_i);
				ret = EXIT_FAILURE;
				goto exit;
			}
			#endif
			comp_tid = flow_vars->wc[cqe_i].wr_id >> 3; // tid is after the first 3 bits
			comp_direction = flow_vars->wc[cqe_i].wr_id & 0x7; // extract the first 3 bits
			switch(comp_direction) {
				case LEFT:
					#pragma omp atomic
					flow_vars_arr[comp_tid].left_comp_count += flow_vars_arr[comp_tid].mod_comp;
					#pragma omp atomic
					left_rem_tx_depth += flow_vars_arr[comp_tid].mod_comp;
					break;
				case RIGHT:
					#pragma omp atomic
					flow_vars_arr[comp_tid].right_comp_count += flow_vars_arr[comp_tid].mod_comp;
					#pragma omp atomic
					right_rem_tx_depth += flow_vars_arr[comp_tid].mod_comp;
					break;
			}
		}
	}

#ifdef ERRCHK
exit:
#endif
	return ret;
}

static int init_ep_res(void)
{
	int ret = 0;
	int dev_i;

	struct ibv_device **dev_list;
	struct ibv_device *dev;

	dev_list = ibv_get_device_list(NULL);
	if (!dev_list) {
		fprintf(stderr, "Failed to get IB devices list");
		ret = EXIT_FAILURE;
		goto exit;
	}

	if (!dev_name) {
		dev = *dev_list;
		if (!dev) {
			fprintf(stderr, "No IB devices found\n");
			return EXIT_FAILURE;
		}
	} else {
		for (dev_i = 0; dev_list[dev_i]; ++dev_i)
			if (!strcmp(ibv_get_device_name(dev_list[dev_i]), dev_name))
				break;
		dev = dev_list[dev_i];
		if (!dev) {
			fprintf(stderr, "IB device %s not found\n", dev_name);
			return EXIT_FAILURE;
		}
	}

	/* Acquire a Device Context */	
	dev_context = ibv_open_device(dev);
	if (!dev_context) {
		fprintf(stderr, "Couldn't get context for %s\n",
		ibv_get_device_name(dev));
		ret = EXIT_FAILURE;
		goto clean_dev_list;
	}
	
	/* Open up Protection Domains */
	left_pd = ibv_alloc_pd(dev_context);
	right_pd = ibv_alloc_pd(dev_context);
	if (!left_pd || !right_pd) {
		fprintf(stderr, "Couldn't allocate PD\n");
		ret = EXIT_FAILURE;
		goto clean_dev_context;
	}

	/* Create Completion Queues */
	cq_depth = tx_depth / mod_comp * 2; // 2 since the left and right QPs are mapped to the same CQ
	cq = ibv_create_cq(dev_context, cq_depth, NULL, NULL, 0);
	if (!cq) {
		fprintf(stderr, "Couldn't create CQs\n");
		ret = EXIT_FAILURE;
		goto clean_pd;
	}

	/* Create Queue Pairs and transition them to the INIT state */
	struct ibv_qp_init_attr qp_init_attr = {
		.send_cq = cq,
		.recv_cq = cq, // the same CQ for sending and receiving
		.cap     = {
			.max_send_wr  = tx_depth, // maximum number of outstanding WRs that can be posted to the SQ in this QP
			.max_recv_wr  = rx_depth, // maximum number of outstanding WRs that can be posted to the RQ in this QP
			.max_send_sge = 1,
			.max_recv_sge = 1,
		},
		.qp_type = IBV_QPT_RC,
		.sq_sig_all = 0, // all send_wqes posted will generate a WC
	};
	
	left_qp = ibv_create_qp(left_pd, &qp_init_attr); // this puts the QP in the RESET state
	right_qp = ibv_create_qp(right_pd, &qp_init_attr); // this puts the QP in the RESET state
	if (!left_qp || !right_qp) {
		fprintf(stderr, "Couldn't create QPs\n");
		ret = EXIT_FAILURE;
		goto clean_cq;
	}

	struct ibv_qp_attr qp_attr = {
		.qp_state = IBV_QPS_INIT,
		.pkey_index = 0, // according to examples
		.port_num = ib_port,
		.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
	};

	/* Initialize the QP to the INIT state */
	ret = ibv_modify_qp(left_qp, &qp_attr,
				IBV_QP_STATE		|
				IBV_QP_PKEY_INDEX	|
				IBV_QP_PORT 		|
				IBV_QP_ACCESS_FLAGS);
	ret = ibv_modify_qp(right_qp, &qp_attr,
				IBV_QP_STATE		|
				IBV_QP_PKEY_INDEX	|
				IBV_QP_PORT 		|
				IBV_QP_ACCESS_FLAGS);
	if (ret) {
		fprintf(stderr, "Failed to modify QPs to INIT\n");
		goto clean_qp;
	}

	goto exit;

clean_qp:
	ibv_destroy_qp(left_qp);
	ibv_destroy_qp(right_qp);

clean_cq:
	ibv_destroy_cq(cq); 

clean_pd:
	ibv_dealloc_pd(left_pd);
	ibv_dealloc_pd(right_pd);

clean_dev_context:
	ibv_close_device(dev_context);

clean_dev_list:
	ibv_free_device_list(dev_list);

exit:
	return ret;
}

static int connect_eps(int left_rank, int right_rank)
{
	int ret = 0;
	int my_lid, left_lid, right_lid;
	int left_dest_qp_index, right_dest_qp_index;;

	/* Query port to get my LID */
	struct ibv_port_attr ib_port_attr;
	ret = ibv_query_port(dev_context, ib_port, &ib_port_attr);
	if (ret) {
		fprintf(stderr, "Failed to get port info\n");
		ret = EXIT_FAILURE;
		goto exit;
	}
	my_lid = ib_port_attr.lid;

	if (rank % 2) {
		/* Exchange LIDs with right rank */
		MPI_Send(&my_lid, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&right_lid, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Exchange LIDs with left rank */
		MPI_Send(&my_lid, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&left_lid, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		/* Exchange QP indexes with right rank */
		MPI_Send(&right_qp->qp_num, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&right_dest_qp_index, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Exchange QP indexes with left rank */
		MPI_Send(&left_qp->qp_num, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
		MPI_Recv(&left_dest_qp_index, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	} else {
		/* Exchange LIDs with left rank */
		MPI_Recv(&left_lid, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&my_lid, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);

		/* Exchange LIDs with right rank */
		MPI_Recv(&right_lid, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&my_lid, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);

		/* Exchange QP indexes with left rank */
		MPI_Recv(&left_dest_qp_index, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&left_qp->qp_num, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);

		/* Exchange QP indexes with right rank */
		MPI_Recv(&right_dest_qp_index, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&right_qp->qp_num, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
	}

	/* Transition left QP to RTR state */
	struct ibv_qp_attr left_qp_attr = {
		.qp_state = IBV_QPS_RTR,
		.ah_attr = {
			.is_global = 0,
			.dlid = left_lid,
			.sl = 0, // set Service Level to 0 (relates to QoS)
			.src_path_bits = 0,
			.port_num = ib_port,
		},
		.path_mtu = IBV_MTU_4096,
		.dest_qp_num = left_dest_qp_index,
		.rq_psn = 0,
		.max_dest_rd_atomic = 16, // according to Anuj's benchmark: 16
		.min_rnr_timer = 12
	};
	struct ibv_qp_attr right_qp_attr = {
		.qp_state = IBV_QPS_RTR,
		.ah_attr = {
			.is_global = 0,
			.dlid = right_lid,
			.sl = 0, // set Service Level to 0 (relates to QoS)
			.src_path_bits = 0,
			.port_num = ib_port,
		},
		.path_mtu = IBV_MTU_4096,
		.dest_qp_num = right_dest_qp_index,
		.rq_psn = 0,
		.max_dest_rd_atomic = 16, // according to Anuj's benchmark: 16
		.min_rnr_timer = 12
	};

	ret = ibv_modify_qp(left_qp, &left_qp_attr,
				IBV_QP_STATE 				|
				IBV_QP_AV					|
				IBV_QP_PATH_MTU				|
				IBV_QP_DEST_QPN				|
				IBV_QP_RQ_PSN				|
				IBV_QP_MAX_DEST_RD_ATOMIC	|
				IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		fprintf(stderr, "Error in transitioning left QP to RTR state\n");
		ret = EXIT_FAILURE;
		goto exit;
	}
	ret = ibv_modify_qp(right_qp, &right_qp_attr,
				IBV_QP_STATE 				|
				IBV_QP_AV					|
				IBV_QP_PATH_MTU				|
				IBV_QP_DEST_QPN				|
				IBV_QP_RQ_PSN				|
				IBV_QP_MAX_DEST_RD_ATOMIC	|
				IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		fprintf(stderr, "Error in transitioning right QP to RTR state\n");
		ret = EXIT_FAILURE;
		goto exit;
	}

	/* Transition to RTS state */
	memset(&left_qp_attr, 0, sizeof left_qp_attr); // reset
	memset(&right_qp_attr, 0, sizeof right_qp_attr); // reset
	left_qp_attr.qp_state = IBV_QPS_RTS;
	left_qp_attr.sq_psn = 0;
	left_qp_attr.timeout = 14; 
	left_qp_attr.retry_cnt = 7;
	left_qp_attr.rnr_retry = 7;
	left_qp_attr.max_rd_atomic = 16; // according to Anuj's benchmark: 16
	right_qp_attr.qp_state = IBV_QPS_RTS;
	right_qp_attr.sq_psn = 0;
	right_qp_attr.timeout = 14; 
	right_qp_attr.retry_cnt = 7;
	right_qp_attr.rnr_retry = 7;
	right_qp_attr.max_rd_atomic = 16; // according to Anuj's benchmark: 16

	ret = ibv_modify_qp(left_qp, &left_qp_attr,
				IBV_QP_STATE		|
				IBV_QP_SQ_PSN		|
				IBV_QP_TIMEOUT		|
				IBV_QP_RETRY_CNT 	|
				IBV_QP_RNR_RETRY	|
				IBV_QP_MAX_QP_RD_ATOMIC );
	if (ret) {
		fprintf(stderr, "Error in transitioning left QP to RTS state\n");
		ret = EXIT_FAILURE;
		goto exit;
	}
	ret = ibv_modify_qp(right_qp, &right_qp_attr,
				IBV_QP_STATE		|
				IBV_QP_SQ_PSN		|
				IBV_QP_TIMEOUT		|
				IBV_QP_RETRY_CNT 	|
				IBV_QP_RNR_RETRY	|
				IBV_QP_MAX_QP_RD_ATOMIC );
	if (ret) {
		fprintf(stderr, "Error in transitioning right QP to RTS state\n");
		ret = EXIT_FAILURE;
		goto exit;
	}

exit: 
	return ret;
}

int free_ep_res(void)
{
	ibv_destroy_qp(left_qp);
	ibv_destroy_qp(right_qp);
	ibv_destroy_cq(cq);
	
	ibv_dereg_mr(left_tile_mr);
	ibv_dereg_mr(right_tile_mr);
	
	ibv_dealloc_pd(left_pd);
	ibv_dealloc_pd(right_pd);

	ibv_close_device(dev_context);

	return 0;
}

int read_args(int argc, char *argv[])
{
	int ret = 0, op;
	int bench_type;

	struct option long_options[] = {
		{.name = "ib-dev",			.has_arg = 1, .val = 'd'},
		{.name = "num-threads",		.has_arg = 1, .val = 't'},
		{.name = "ga-dim-x",		.has_arg = 1, .val = 'n'},
		{.name = "ga-dim-y",		.has_arg = 1, .val = 'm'},
		{.name = "iterations",		.has_arg = 1, .val = 'r'},
		{.name = "postlist",		.has_arg = 1, .val = 'p'},
		{.name = "mod-comp",		.has_arg = 1, .val = 'q'},
		{.name = "tx-depth",		.has_arg = 1, .val = 'T'},
		{.name = "rx-depth",		.has_arg = 1, .val = 'R'},
		{.name = "use-cq-ex",		.has_arg = 0, .val = 'x'},
		{.name = "compute",			.has_arg = 0, .val = 'C'},
		//{.name = "use-td",		.has_arg = 0, .val = 'o'},
		{0 , 0 , 0, 0}
	};

	while (1) {
		op = getopt_long(argc, argv, "h?d:t:n:m:r:p:q:T:R:xC", long_options, NULL);
		if (op == -1)
			break;
		
		switch (op) {
			case '?':
			case 'h':
				print_usage(argv[0], 1);
				ret = -1;
				goto exit;
			default:
				parse_args(op, optarg);
				break;
		}
	}

	bench_type = EFF_MT;
	if (optind < argc) {
		print_usage(argv[0], bench_type);
		ret = -1;
	}

exit:
	return ret;
}

int init_params(void)
{
	dev_name = NULL;
	num_threads = DEF_NUM_THREADS;
	
	ga_dim_x = DEF_GA_DIM_X;
	ga_dim_common = DEF_GA_DIM_COMMON;
	ga_dim_y = DEF_GA_DIM_Y;
	tile_dim_x = DEF_TILE_DIM_X;
	tile_dim_common = DEF_TILE_DIM_COMMON;
	tile_dim_y = DEF_TILE_DIM_Y;
	iterations = DEF_ITERATIONS;
	
	postlist = DEF_POSTLIST;
	mod_comp = DEF_MOD_COMP;
	tx_depth = DEF_TX_DEPTH;
	rx_depth = DEF_RX_DEPTH;
	use_cq_ex = DEF_USE_CQ_EX;

	compute = DEF_COMPUTE;

	ib_port = DEF_IB_PORT;

	return 0;
}
