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

static int put_halo(int direction, int num_rows, int hz_halo_dim, int element_bytes,
                    int hz_tile_dim, unsigned long int local_source_addr, int local_key,
                    unsigned long int remote_dest_addr, int remote_key,
                    struct stencil_thread_flow_vars *flow_vars);

static int wait(struct stencil_thread_flow_vars *flow_vars);

static int alloc_ep_res(void);

static int init_ep_res(void);

static int connect_eps(int left_rank, int right_rank);

static int free_ep_res(void);

struct ibv_context *dev_context;
struct ibv_context **ded_ctx;
struct ibv_pd **left_ded_pd;
struct ibv_pd **right_ded_pd;
struct ibv_pd *left_pd;
struct ibv_pd *right_pd;
struct ibv_pd **left_parent_d;
struct ibv_pd **right_parent_d;
struct ibv_td **left_td;
struct ibv_td **right_td;
struct ibv_mr *left_tile_mr;
struct ibv_mr *right_tile_mr;
struct ibv_mr **left_ded_mr;
struct ibv_mr **right_ded_mr;
struct ibv_cq **cq;
struct ibv_cq_ex **cq_ex;
struct ibv_qp **left_qp;
struct ibv_qp **right_qp;

int main(int argc, char *argv[])
{
    int ret = 0, provided;

    int tile_dim_x, tile_dim_y;
    int left_rank, right_rank;
    int left_rkey, right_rkey;
    int *left_ded_rkey, *right_ded_rkey;
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
        fprintf(stderr,
                "Supporting only multiples of two at the moment since we have only two nodes\n");
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
        fprintf(stderr,
                "The number of ranks has to be a factor of the global horizontal dimension\n");
        ret = EXIT_FAILURE;
        goto clean_mpi;
    }

    if (size / 2 * num_threads > 16) {
        fprintf(stderr,
                "Total number of threads per node has to be less than 16, the number of cores per socket\n");
        ret = EXIT_FAILURE;
        goto clean_mpi;
    }

    if (ga_dim_y % num_threads) {
        fprintf(stderr,
                "Number of threads per rank has to be a factor of the global vertical dimension\n");
        ret = EXIT_FAILURE;
        goto clean_mpi;
    }

    if (xdynamic + dynamic + sharedd + use_static > 1) {
        fprintf(stderr, "You can use only one category at a time\n");
        ret = EXIT_FAILURE;
        goto clean_mpi;
    }

    omp_set_num_threads(num_threads);

    num_qps = (xdynamic) ? 2 * num_threads : num_threads;

    rows_per_thread = ga_dim_y / num_threads;

    tile_dim_y = ga_dim_y;
    tile_dim_x = ga_dim_x / size;

    left_rank = (rank == 0) ? size - 1 : rank - 1;
    right_rank = (rank + 1) % size;

    ret = alloc_ep_res();
    if (ret) {
        fprintf(stderr, "Failure in allocating EP's resources.\n");
        goto clean_mpi;
    }

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

    local_left_source_addr = (unsigned long int) &local_tile[(tile_dim_x + 2) + 1];
    local_left_halo_addr = (unsigned long int) &local_tile[(tile_dim_x + 2) + 0];
    local_right_source_addr = (unsigned long int) &local_tile[(tile_dim_x + 2) + tile_dim_x];
    local_right_halo_addr = (unsigned long int) &local_tile[(tile_dim_x + 2) + tile_dim_x + 1];

    int qp_i;
    if (dedicated) {
        for (qp_i = 0; qp_i < num_qps; qp_i++) {
            left_ded_mr[qp_i] =
                ibv_reg_mr(left_ded_pd[qp_i], local_tile, tile_size * sizeof *local_tile,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ);
            right_ded_mr[qp_i] =
                ibv_reg_mr(right_ded_pd[qp_i], local_tile, tile_size * sizeof *local_tile,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ);
            if (!left_ded_mr[qp_i] || !right_ded_mr[qp_i]) {
                fprintf(stderr, "Failure in allocating dedicated MRs for the halo regions.\n");
                ret = EXIT_FAILURE;
                goto clean_mpi;
            }
        }
    } else {
        left_tile_mr =
            ibv_reg_mr(left_pd, local_tile, tile_size * sizeof *local_tile,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        right_tile_mr =
            ibv_reg_mr(right_pd, local_tile, tile_size * sizeof *local_tile,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!left_tile_mr || !right_tile_mr) {
            fprintf(stderr, "Failure in allocating MRs for the halo regions.\n");
            ret = EXIT_FAILURE;
            goto clean_mpi;
        }
    }

    if (dedicated) {
        left_ded_rkey = malloc(num_qps * (sizeof *left_ded_rkey));
        right_ded_rkey = malloc(num_qps * (sizeof *right_ded_rkey));
        if (!left_ded_rkey || !right_ded_rkey) {
            ret = EXIT_FAILURE;
            fprintf(stderr, "Failure in allocating arrays of rkeys\n");
            goto clean_mpi;
        }
    }

    if (rank % 2) {
        if (dedicated) {
            for (qp_i = 0; qp_i < num_qps; qp_i++) {
                MPI_Send(&left_ded_mr[qp_i]->rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
                MPI_Recv(&left_ded_rkey[qp_i], 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                MPI_Send(&right_ded_mr[qp_i]->rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
                MPI_Recv(&right_ded_rkey[qp_i], 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(&left_tile_mr->rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(&left_rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Send(&right_tile_mr->rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(&right_rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Send(&local_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&remote_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        MPI_Send(&local_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&remote_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    } else {
        if (dedicated) {
            for (qp_i = 0; qp_i < num_qps; qp_i++) {
                MPI_Recv(&right_ded_rkey[qp_i], 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(&right_ded_mr[qp_i]->rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);

                MPI_Recv(&left_ded_rkey[qp_i], 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(&left_ded_mr[qp_i]->rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(&right_rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&right_tile_mr->rkey, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);

            MPI_Recv(&left_rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&left_tile_mr->rkey, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
        }
        MPI_Recv(&remote_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&local_right_halo_addr, 1, MPI_UNSIGNED_LONG, right_rank, 0, MPI_COMM_WORLD);

        MPI_Recv(&remote_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&local_left_halo_addr, 1, MPI_UNSIGNED_LONG, left_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    struct stencil_thread_flow_vars *flow_vars;

    element_bytes = sizeof(double);     // TODO: keep it configurable

    flow_vars = calloc(num_threads, sizeof *flow_vars);

#pragma omp parallel private(ret)
    {
        int tid, qp_i;
        int p;

        tid = omp_get_thread_num();
        qp_i = (xdynamic) ? 2 * tid : tid;

        struct ibv_sge *SGE;
        struct ibv_send_wr *send_wqe;
        struct ibv_wc *WC;

        posix_memalign((void **) &SGE, CACHE_LINE_SIZE, postlist * sizeof(struct ibv_sge));
        posix_memalign((void **) &send_wqe, CACHE_LINE_SIZE, postlist * sizeof(struct ibv_send_wr));
        posix_memalign((void **) &WC, CACHE_LINE_SIZE, cq_depth * sizeof(struct ibv_wc));

        memset(SGE, 0, postlist * sizeof(struct ibv_sge));
        memset(send_wqe, 0, postlist * sizeof(struct ibv_send_wr));
        memset(WC, 0, cq_depth * sizeof(struct ibv_wc));

        for (p = 0; p < postlist; p++) {
            send_wqe[p].next = (p == postlist - 1) ? NULL : &send_wqe[p + 1];
            send_wqe[p].sg_list = &SGE[p];
            send_wqe[p].num_sge = 1;
        }

        struct ibv_qp_attr attr;
        struct ibv_qp_init_attr qp_init_attr;
        ret = ibv_query_qp(left_qp[qp_i], &attr, IBV_QP_CAP, &qp_init_attr);
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
        flow_vars[tid].sge = SGE;
        flow_vars[tid].wqe = send_wqe;
        flow_vars[tid].wc = WC;
        flow_vars[tid].left_qp = left_qp[qp_i];
        flow_vars[tid].right_qp = right_qp[qp_i];
        flow_vars[tid].my_cq = cq[qp_i];
    }

    int bytes_per_thread = rows_per_thread * (tile_dim_x + 2) * element_bytes;

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    /* Stencil */
    int iter;
    for (iter = 0; iter < iterations; iter++) {

#pragma omp parallel private(ret) firstprivate(local_left_source_addr, local_right_source_addr, remote_left_halo_addr, remote_right_halo_addr, bytes_per_thread, rows_per_thread, element_bytes, tile_dim_x, left_tile_mr, right_tile_mr, left_rkey, right_rkey)
        {
            int tid = omp_get_thread_num();
            int my_i = tid * rows_per_thread + 1;

            unsigned long int my_left_source_addr = local_left_source_addr + tid * bytes_per_thread;    // &local_tile[my_i * (tile_dim_x + 2) + 1]
            unsigned long int my_right_source_addr = local_right_source_addr + tid * bytes_per_thread;  // &local_tile[my_i * (tile_dim_x + 2) + tile_dim_x]
            unsigned long int my_left_remote_addr = remote_left_halo_addr + tid * bytes_per_thread;
            unsigned long int my_right_remote_addr =
                remote_right_halo_addr + tid * bytes_per_thread;

            /* Send halo region to left neighbor */
            ret = put_halo(LEFT, rows_per_thread, 1, element_bytes, tile_dim_x + 2,
                           my_left_source_addr,
                           (dedicated) ? left_ded_mr[tid]->lkey : left_tile_mr->lkey,
                           my_left_remote_addr, (dedicated) ? left_ded_rkey[tid] : left_rkey,
                           &flow_vars[tid]);
#ifdef ERRCHK
            if (ret) {
                fprintf(stderr, "Error in putting halo to the right\n");
                exit(0);
            }
#endif

            /* Send halo region to right neighbor */
            ret = put_halo(RIGHT, rows_per_thread, 1, element_bytes, tile_dim_x + 2,
                           my_right_source_addr,
                           (dedicated) ? right_ded_mr[tid]->lkey : right_tile_mr->lkey,
                           my_right_remote_addr, (dedicated) ? right_ded_rkey[tid] : right_rkey,
                           &flow_vars[tid]);
#ifdef ERRCHK
            if (ret) {
                fprintf(stderr, "Error in putting halo to the right\n");
                exit(0);
            }
#endif

            /* Complete */
            ret = wait(&flow_vars[tid]);
#ifdef ERRCHK
            if (ret) {
                fprintf(stderr, "Error in wating for puts to complete\n");
                exit(0);
            }
#endif

            /* Update grid */
            if (compute) {
                int i, j;
                for (i = my_i; i < my_i + rows_per_thread; i++) {
                    for (j = 1; j < tile_dim_x + 1; j++) {
                        local_tile[i * (tile_dim_x + 2) + j] += local_tile[(i - 1) * (tile_dim_x + 2) + j] +    /* top */
                            local_tile[(i + 1) * (tile_dim_x + 2) + j] +        /* bottom */
                            local_tile[i * (tile_dim_x + 2) + j - 1] +  /* left */
                            local_tile[i * (tile_dim_x + 2) + j + 1];   /* right */
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
            printf("%-10s\t%-10s\t%-10s\t%-10s\n", "PPN", "Threads", "Write Mmsgs/s", "Time (s)");
            printf("%-10d\t%-10d\t%-10.2f\t%-10.2f\n", size / 2, num_threads, write_mr, total_time);
        } else {
            printf("%-10s\t%-10s\t%-10s\n", "PPN", "Threads", "Time (s)");
            printf("%-10d\t%-10d\t%-10.2f\n", size / 2, num_threads, total_time);
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

static int put_halo(int direction, int num_rows, int hz_halo_dim, int element_bytes,
                    int hz_tile_dim, unsigned long int local_source_addr, int local_key,
                    unsigned long int remote_dest_addr, int remote_key,
                    struct stencil_thread_flow_vars *flow_vars)
{
    int ret = 0;
    int p;
    int db, doorbells;
    int cqe_count;
    int send_inline;
    int bytes_in_wqe, bytes_in_row;
    uint64_t local_base_addr, remote_base_addr;
    struct ibv_send_wr *bad_send_wqe;
    int cqe_i;

    /* Active variables */
    int posts = 0;
    int *post_count = NULL;
    int *comp_count = NULL;
    struct ibv_qp *qp = NULL;

    switch (direction) {
        case LEFT:
            flow_vars->left_posts += num_rows;
            posts = flow_vars->left_posts;
            post_count = &flow_vars->left_post_count;
            comp_count = &flow_vars->left_comp_count;
            qp = flow_vars->left_qp;
            break;
        case RIGHT:
            flow_vars->right_posts += num_rows;
            posts = flow_vars->right_posts;
            post_count = &flow_vars->right_post_count;
            comp_count = &flow_vars->right_comp_count;
            qp = flow_vars->right_qp;
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
        flow_vars->sge[p].length = bytes_in_wqe;
        flow_vars->sge[p].lkey = local_key;

        flow_vars->wqe[p].wr_id = direction;
        flow_vars->wqe[p].opcode = IBV_WR_RDMA_WRITE;
        flow_vars->wqe[p].send_flags = send_inline;
        flow_vars->wqe[p].wr.rdma.rkey = remote_key;
    }

    local_base_addr = (uint64_t) local_source_addr;
    remote_base_addr = (uint64_t) remote_dest_addr;

    while (*post_count < posts) {
        //printf("TID %d: Postcount %d\tCompcount %d\n", tid, post_count, comp_count);
        // Post
        doorbells = min((posts - *post_count),
                        (flow_vars->tx_depth - (*post_count - *comp_count))) / flow_vars->postlist;
        for (db = 0; db < doorbells; db++) {
            for (p = 0; p < flow_vars->postlist; p++) {
                flow_vars->sge[p].addr = local_base_addr;

                if (((*post_count) + p + 1) % flow_vars->mod_comp == 0)
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
                fprintf(stderr, "Thread %d: Error %d in posting send_wqe on QP\n", flow_vars->tid,
                        ret);
                goto exit;
            }
#endif
            *post_count += flow_vars->postlist;
        }
        if (!doorbells) {
            // Poll only if SQ is full
            cqe_count = ibv_poll_cq(flow_vars->my_cq, flow_vars->cq_depth, flow_vars->wc);
#ifdef ERRCHK
            if (cqe_count < 0) {
                fprintf(stderr, "Thread %d: Failure in polling CQ: %d\n", flow_vars->tid,
                        cqe_count);
                ret = cqe_count;
                goto exit;
            }
#endif
            for (cqe_i = 0; cqe_i < cqe_count; cqe_i++) {
#ifdef ERRCHK
                if (flow_vars->wc[cqe_i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "Thread %d: Failed status %s for %d; cqe_count %d\n",
                            flow_vars->tid, ibv_wc_status_str(flow_vars->wc[cqe_i].status),
                            (int) flow_vars->wc[cqe_i].wr_id, cqe_i);
                    ret = EXIT_FAILURE;
                    goto exit;
                }
#endif
                switch (flow_vars->wc[cqe_i].wr_id) {
                    case LEFT:
                        flow_vars->left_comp_count += flow_vars->mod_comp;
                        break;
                    case RIGHT:
                        flow_vars->right_comp_count += flow_vars->mod_comp;
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

static int wait(struct stencil_thread_flow_vars *flow_vars)
{
    int ret = 0;
    int cqe_count;
    int cqe_i;

    while ((flow_vars->left_comp_count < flow_vars->left_post_count) ||
           (flow_vars->right_comp_count < flow_vars->right_post_count)) {
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
                fprintf(stderr, "Thread %d: Failed status %s for %d; cqe_count %d\n",
                        flow_vars->tid, ibv_wc_status_str(flow_vars->wc[cqe_i].status),
                        (int) flow_vars->wc[cqe_i].wr_id, cqe_i);
                ret = EXIT_FAILURE;
                goto exit;
            }
#endif
            switch (flow_vars->wc[cqe_i].wr_id) {
                case LEFT:
                    flow_vars->left_comp_count += flow_vars->mod_comp;
                    break;
                case RIGHT:
                    flow_vars->right_comp_count += flow_vars->mod_comp;
                    break;
            }
        }
    }

#ifdef ERRCHK
  exit:
#endif
    return ret;
}

static int alloc_ep_res(void)
{
    int ret = 0;

    if (dedicated) {
        ded_ctx = malloc(num_qps * (sizeof *ded_ctx));
        if (!ded_ctx) {
            fprintf(stderr, "Failure in allocating dedicated device contexts\n");
            ret = EXIT_FAILURE;
            goto exit;
        }

        left_ded_pd = malloc(num_qps * (sizeof *left_ded_pd));
        if (!left_ded_pd) {
            fprintf(stderr, "Failure in allocating dedicated protection domains\n");
            ret = EXIT_FAILURE;
            goto exit;
        }
        right_ded_pd = malloc(num_qps * (sizeof *right_ded_pd));
        if (!right_ded_pd) {
            fprintf(stderr, "Failure in allocating dedicated protection domains\n");
            ret = EXIT_FAILURE;
            goto exit;
        }

        left_ded_mr = malloc(num_qps * (sizeof *left_ded_mr));
        if (!left_ded_mr) {
            fprintf(stderr, "Failure in allocating dedicated memory regions\n");
            ret = EXIT_FAILURE;
            goto exit;
        }
        right_ded_mr = malloc(num_qps * (sizeof *right_ded_mr));
        if (!right_ded_mr) {
            fprintf(stderr, "Failure in allocating dedicated memory regions\n");
            ret = EXIT_FAILURE;
            goto exit;
        }
    }

    if (dynamic || xdynamic || sharedd) {
        left_parent_d = malloc(num_qps * (sizeof *left_parent_d));
        right_parent_d = malloc(num_qps * (sizeof *right_parent_d));
        if (!left_parent_d || !right_parent_d) {
            fprintf(stderr, "Failure in allocating parent domains\n");
            ret = EXIT_FAILURE;
            goto exit;
        }

        left_td = malloc(num_qps * (sizeof *left_td));
        right_td = malloc(num_qps * (sizeof *right_td));
        if (!left_td || !right_td) {
            fprintf(stderr, "Failure in allocating thread domains\n");
            return EXIT_FAILURE;
            goto exit;
        }
    }

    if (use_cq_ex) {
        cq_ex = malloc(num_qps * (sizeof *cq_ex));
        if (!cq_ex) {
            fprintf(stderr, "Failure in allocating extended completion queues\n");
            ret = EXIT_FAILURE;
            goto exit;
        }
    } else {
        cq = malloc(num_qps * (sizeof *cq));
        if (!cq) {
            fprintf(stderr, "Failure in allocating completion queues\n");
            ret = EXIT_FAILURE;
            goto exit;
        }
    }

    left_qp = malloc(num_qps * (sizeof *left_qp));
    right_qp = malloc(num_qps * (sizeof *right_qp));
    if (!left_qp || !right_qp) {
        fprintf(stderr, "Failure in allocating queue pairs\n");
        ret = EXIT_FAILURE;
        goto exit;
    }

  exit:
    return ret;
}

static int init_ep_res(void)
{
    int ret = 0;
    int dev_i, cq_i, qp_i;

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
    if (dedicated) {
        for (qp_i = 0; qp_i < num_qps; qp_i++) {
            ded_ctx[qp_i] = ibv_open_device(dev);
            if (!ded_ctx[qp_i]) {
                fprintf(stderr, "Couldn't get dedicated context for %s\n",
                        ibv_get_device_name(dev));
                ret = EXIT_FAILURE;
                goto clean_dev_list;
            }
        }
    } else {
        dev_context = ibv_open_device(dev);
        if (!dev_context) {
            fprintf(stderr, "Couldn't get context for %s\n", ibv_get_device_name(dev));
            ret = EXIT_FAILURE;
            goto clean_dev_list;
        }
    }

    /* Open up Protection Domains */
    if (dedicated) {
        for (qp_i = 0; qp_i < num_qps; qp_i++) {
            left_ded_pd[qp_i] = ibv_alloc_pd(ded_ctx[qp_i]);
            right_ded_pd[qp_i] = ibv_alloc_pd(ded_ctx[qp_i]);
            if (!left_ded_pd[qp_i] || !right_ded_pd[qp_i]) {
                fprintf(stderr, "Couldn't allocate dedicated PDs\n");
                ret = EXIT_FAILURE;
                goto clean_dev_context;
            }
        }
    } else {
        left_pd = ibv_alloc_pd(dev_context);
        right_pd = ibv_alloc_pd(dev_context);
        if (!left_pd || !right_pd) {
            fprintf(stderr, "Couldn't allocate PD\n");
            ret = EXIT_FAILURE;
            goto clean_dev_context;
        }
    }

    if (xdynamic || dynamic || sharedd) {
        for (qp_i = 0; qp_i < num_qps; qp_i++) {
            /* Open up Thread Domains */
            struct ibv_td_init_attr td_init;
            memset(&td_init, 0, sizeof td_init);
            td_init.comp_mask = 0;
            td_init.independent = (sharedd) ? 0 : 1;
            left_td[qp_i] = ibv_alloc_td(dev_context, &td_init);
            if (!left_td[qp_i]) {
                fprintf(stderr, "Couldn't allocate TDs\n");
                ret = EXIT_FAILURE;
                goto clean_pd;
            }

            /* Open up Parent Domains */
            struct ibv_parent_domain_init_attr left_pd_init;
            memset(&left_pd_init, 0, sizeof left_pd_init);
            left_pd_init.pd = left_pd;
            left_pd_init.td = left_td[qp_i];
            left_pd_init.comp_mask = 0;
            left_parent_d[qp_i] = ibv_alloc_parent_domain(dev_context, &left_pd_init);
            if (!left_parent_d[qp_i]) {
                fprintf(stderr, "Couldn't allocate Parent Ds\n");
                ret = EXIT_FAILURE;
                goto clean_td;
            }
        }

        for (qp_i = 0; qp_i < num_qps; qp_i++) {
            /* Open up Thread Domains */
            struct ibv_td_init_attr td_init;
            memset(&td_init, 0, sizeof td_init);
            td_init.comp_mask = 0;
            td_init.independent = (sharedd) ? 0 : 1;
            right_td[qp_i] = ibv_alloc_td(dev_context, &td_init);
            if (!right_td[qp_i]) {
                fprintf(stderr, "Couldn't allocate TDs\n");
                ret = EXIT_FAILURE;
                goto clean_pd;
            }

            /* Open up Parent Domains */
            struct ibv_parent_domain_init_attr right_pd_init;
            memset(&right_pd_init, 0, sizeof right_pd_init);
            right_pd_init.pd = right_pd;
            right_pd_init.td = right_td[qp_i];
            right_pd_init.comp_mask = 0;
            right_parent_d[qp_i] = ibv_alloc_parent_domain(dev_context, &right_pd_init);
            if (!right_parent_d[qp_i]) {
                fprintf(stderr, "Couldn't allocate Parent Ds\n");
                ret = EXIT_FAILURE;
                goto clean_td;
            }
        }
    }

    /* Create Completion Queues */
    cq_depth = tx_depth / mod_comp * 2; // 2 since the left and right QPs are mapped to the same CQ
    for (cq_i = 0; cq_i < num_qps; cq_i++) {
        if (use_cq_ex) {
            struct ibv_cq_init_attr_ex cq_ex_attr = {
                .cqe = cq_depth,
                .cq_context = NULL,
                .channel = NULL,
                .comp_vector = 0,
                .wc_flags = 0,
                .comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS,
                .flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED,
            };
            cq_ex[cq_i] = ibv_create_cq_ex(dev_context, &cq_ex_attr);
            if (!cq_ex[cq_i]) {
                fprintf(stderr, "Couldn't create extended CQs\n");
                ret = EXIT_FAILURE;
                goto clean_parent_d;
            }
        } else {
            if (dedicated)
                cq[cq_i] = ibv_create_cq(ded_ctx[cq_i], cq_depth, NULL, NULL, 0);
            else
                cq[cq_i] = ibv_create_cq(dev_context, cq_depth, NULL, NULL, 0);
            if (!cq[cq_i]) {
                fprintf(stderr, "Couldn't create CQs\n");
                ret = EXIT_FAILURE;
                goto clean_parent_d;
            }
        }
    }

    /* Create left Queue Pairs and transition them to the INIT state */
    for (qp_i = 0; qp_i < num_qps; qp_i++) {
        cq_i = qp_i;
        struct ibv_qp_init_attr qp_init_attr = {
            .send_cq = (use_cq_ex) ? ibv_cq_ex_to_cq(cq_ex[cq_i]) : cq[cq_i],
            .recv_cq = (use_cq_ex) ? ibv_cq_ex_to_cq(cq_ex[cq_i]) : cq[cq_i],   // the same CQ for sending and receiving
            .cap = {
                    .max_send_wr = tx_depth,    // maximum number of outstanding WRs that can be posted to the SQ in this QP
                    .max_recv_wr = rx_depth,    // maximum number of outstanding WRs that can be posted to the RQ in this QP
                    .max_send_sge = 1,
                    .max_recv_sge = 1,
                    },
            .qp_type = IBV_QPT_RC,
            .sq_sig_all = 0,    // all send_wqes posted will generate a WC
        };
        if (xdynamic || dynamic || sharedd)
            left_qp[qp_i] = ibv_create_qp(left_parent_d[qp_i], &qp_init_attr);  // this puts the QP in the RESET state
        else if (dedicated)
            left_qp[qp_i] = ibv_create_qp(left_ded_pd[qp_i], &qp_init_attr);    // this puts the QP in the RESET state
        else
            left_qp[qp_i] = ibv_create_qp(left_pd, &qp_init_attr);      // this puts the QP in the RESET state
        if (!left_qp[qp_i]) {
            fprintf(stderr, "Couldn't create QPs\n");
            ret = EXIT_FAILURE;
            goto clean_cq;
        }

        struct ibv_qp_attr qp_attr = {
            .qp_state = IBV_QPS_INIT,
            .pkey_index = 0,    // according to examples
            .port_num = ib_port,
            .qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
        };

        /* Initialize the QP to the INIT state */
        ret = ibv_modify_qp(left_qp[qp_i], &qp_attr,
                            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (ret) {
            fprintf(stderr, "Failed to modify QPs to INIT\n");
            goto clean_qp;
        }
    }

    /* Create right Queue Pairs and transition them to the INIT state */
    for (qp_i = 0; qp_i < num_qps; qp_i++) {
        cq_i = qp_i;
        struct ibv_qp_init_attr qp_init_attr = {
            .send_cq = (use_cq_ex) ? ibv_cq_ex_to_cq(cq_ex[cq_i]) : cq[cq_i],
            .recv_cq = (use_cq_ex) ? ibv_cq_ex_to_cq(cq_ex[cq_i]) : cq[cq_i],   // the same CQ for sending and receiving
            .cap = {
                    .max_send_wr = tx_depth,    // maximum number of outstanding WRs that can be posted to the SQ in this QP
                    .max_recv_wr = rx_depth,    // maximum number of outstanding WRs that can be posted to the RQ in this QP
                    .max_send_sge = 1,
                    .max_recv_sge = 1,
                    },
            .qp_type = IBV_QPT_RC,
            .sq_sig_all = 0,    // all send_wqes posted will generate a WC
        };
        if (xdynamic || dynamic || sharedd)
            right_qp[qp_i] = ibv_create_qp(right_parent_d[qp_i], &qp_init_attr);        // this puts the QP in the RESET state
        else if (dedicated)
            right_qp[qp_i] = ibv_create_qp(right_ded_pd[qp_i], &qp_init_attr);  // this puts the QP in the RESET state
        else
            right_qp[qp_i] = ibv_create_qp(right_pd, &qp_init_attr);    // this puts the QP in the RESET state
        if (!right_qp[qp_i]) {
            fprintf(stderr, "Couldn't create QPs\n");
            ret = EXIT_FAILURE;
            goto clean_cq;
        }

        struct ibv_qp_attr qp_attr = {
            .qp_state = IBV_QPS_INIT,
            .pkey_index = 0,    // according to examples
            .port_num = ib_port,
            .qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
        };

        /* Initialize the QP to the INIT state */
        ret = ibv_modify_qp(right_qp[qp_i], &qp_attr,
                            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (ret) {
            fprintf(stderr, "Failed to modify QPs to INIT\n");
            goto clean_qp;
        }
    }

    goto exit;

  clean_qp:
    for (qp_i = 0; qp_i < num_qps; qp_i++) {
        ibv_destroy_qp(left_qp[qp_i]);
        ibv_destroy_qp(right_qp[qp_i]);
    }

  clean_cq:
    for (cq_i = 0; cq_i < num_qps; cq_i++) {
        ibv_destroy_cq((use_cq_ex) ? ibv_cq_ex_to_cq(cq_ex[cq_i]) : cq[cq_i]);
    }

  clean_parent_d:
    for (qp_i = 0; qp_i < num_qps; qp_i++) {
        ibv_dealloc_pd(left_parent_d[qp_i]);
        ibv_dealloc_pd(right_parent_d[qp_i]);
    }

  clean_td:
    for (qp_i = 0; qp_i < num_qps; qp_i++) {
        ibv_dealloc_td(left_td[qp_i]);
        ibv_dealloc_td(right_td[qp_i]);
    }

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
    int tid;
    int my_lid, left_lid, right_lid;
    int *left_dest_qp_index, *right_dest_qp_index;;

    left_dest_qp_index = calloc(num_qps, sizeof(int));
    right_dest_qp_index = calloc(num_qps, sizeof(int));
    if (!left_dest_qp_index || !right_dest_qp_index) {
        fprintf(stderr, "Error in allocating array of QP indexes\n");
        ret = EXIT_FAILURE;
        goto exit;
    }

    /* Query port to get my LID */
    struct ibv_port_attr ib_port_attr;
    if (dedicated)
        ret = ibv_query_port(ded_ctx[0], ib_port, &ib_port_attr);
    else
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
        for (tid = 0; tid < num_qps; tid++) {
            MPI_Send(&right_qp[tid]->qp_num, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(&right_dest_qp_index[tid], 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        /* Exchange QP indexes with left rank */
        for (tid = 0; tid < num_qps; tid++) {
            MPI_Send(&left_qp[tid]->qp_num, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(&left_dest_qp_index[tid], 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    } else {
        /* Exchange LIDs with left rank */
        MPI_Recv(&left_lid, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&my_lid, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);

        /* Exchange LIDs with right rank */
        MPI_Recv(&right_lid, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&my_lid, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);

        /* Exchange QP indexes with left rank */
        for (tid = 0; tid < num_qps; tid++) {
            MPI_Recv(&left_dest_qp_index[tid], 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(&left_qp[tid]->qp_num, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD);
        }

        /* Exchange QP indexes with right rank */
        for (tid = 0; tid < num_qps; tid++) {
            MPI_Recv(&right_dest_qp_index[tid], 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(&right_qp[tid]->qp_num, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD);
        }
    }

    for (tid = 0; tid < num_qps; tid++) {

        /* Transition left QP to RTR state */
        struct ibv_qp_attr left_qp_attr = {
            .qp_state = IBV_QPS_RTR,
            .ah_attr = {
                        .is_global = 0,
                        .dlid = left_lid,
                        .sl = 0,        // set Service Level to 0 (relates to QoS)
                        .src_path_bits = 0,
                        .port_num = ib_port,
                        },
            .path_mtu = IBV_MTU_4096,
            .dest_qp_num = left_dest_qp_index[tid],
            .rq_psn = 0,
            .max_dest_rd_atomic = 16,   // according to Anuj's benchmark: 16
            .min_rnr_timer = 12
        };
        struct ibv_qp_attr right_qp_attr = {
            .qp_state = IBV_QPS_RTR,
            .ah_attr = {
                        .is_global = 0,
                        .dlid = right_lid,
                        .sl = 0,        // set Service Level to 0 (relates to QoS)
                        .src_path_bits = 0,
                        .port_num = ib_port,
                        },
            .path_mtu = IBV_MTU_4096,
            .dest_qp_num = right_dest_qp_index[tid],
            .rq_psn = 0,
            .max_dest_rd_atomic = 16,   // according to Anuj's benchmark: 16
            .min_rnr_timer = 12
        };

        ret = ibv_modify_qp(left_qp[tid], &left_qp_attr,
                            IBV_QP_STATE |
                            IBV_QP_AV |
                            IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN |
                            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (ret) {
            fprintf(stderr, "Error in transitioning left QP %d to RTR state\n", tid);
            ret = EXIT_FAILURE;
            goto exit;
        }
        ret = ibv_modify_qp(right_qp[tid], &right_qp_attr,
                            IBV_QP_STATE |
                            IBV_QP_AV |
                            IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN |
                            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (ret) {
            fprintf(stderr, "Error in transitioning right QP %d to RTR state\n", tid);
            ret = EXIT_FAILURE;
            goto exit;
        }

        /* Transition to RTS state */
        memset(&left_qp_attr, 0, sizeof left_qp_attr);  // reset
        memset(&right_qp_attr, 0, sizeof right_qp_attr);        // reset
        left_qp_attr.qp_state = IBV_QPS_RTS;
        left_qp_attr.sq_psn = 0;
        left_qp_attr.timeout = 14;
        left_qp_attr.retry_cnt = 7;
        left_qp_attr.rnr_retry = 7;
        left_qp_attr.max_rd_atomic = 16;        // according to Anuj's benchmark: 16
        right_qp_attr.qp_state = IBV_QPS_RTS;
        right_qp_attr.sq_psn = 0;
        right_qp_attr.timeout = 14;
        right_qp_attr.retry_cnt = 7;
        right_qp_attr.rnr_retry = 7;
        right_qp_attr.max_rd_atomic = 16;       // according to Anuj's benchmark: 16

        ret = ibv_modify_qp(left_qp[tid], &left_qp_attr,
                            IBV_QP_STATE |
                            IBV_QP_SQ_PSN |
                            IBV_QP_TIMEOUT |
                            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
        if (ret) {
            fprintf(stderr, "Error in transitioning left QP %d to RTS state\n", tid);
            ret = EXIT_FAILURE;
            goto exit;
        }
        ret = ibv_modify_qp(right_qp[tid], &right_qp_attr,
                            IBV_QP_STATE |
                            IBV_QP_SQ_PSN |
                            IBV_QP_TIMEOUT |
                            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
        if (ret) {
            fprintf(stderr, "Error in transitioning right QP %d to RTS state\n", tid);
            ret = EXIT_FAILURE;
            goto exit;
        }
    }

  exit:
    return ret;
}

int free_ep_res(void)
{
    int tid;

    for (tid = 0; tid < num_qps; tid++) {
        ibv_destroy_qp(left_qp[tid]);
        ibv_destroy_qp(right_qp[tid]);
    }
    if (xdynamic || dynamic || sharedd) {
        for (tid = 0; tid < num_qps; tid++) {
            ibv_dealloc_td(left_td[tid]);
            ibv_dealloc_td(right_td[tid]);
        }
        for (tid = 0; tid < num_qps; tid++) {
            ibv_dealloc_pd(left_parent_d[tid]);
            ibv_dealloc_pd(right_parent_d[tid]);
        }
    }
    for (tid = 0; tid < num_qps; tid++) {
        ibv_destroy_cq((use_cq_ex) ? ibv_cq_ex_to_cq(cq_ex[tid]) : cq[tid]);
    }

    if (dedicated) {
        for (tid = 0; tid < num_qps; tid++) {
            ibv_dereg_mr(left_ded_mr[tid]);
            ibv_dereg_mr(right_ded_mr[tid]);
        }
        for (tid = 0; tid < num_qps; tid++) {
            ibv_dealloc_pd(left_ded_pd[tid]);
            ibv_dealloc_pd(right_ded_pd[tid]);
        }
        for (tid = 0; tid < num_qps; tid++) {
            ibv_close_device(ded_ctx[tid]);
        }
    } else {
        ibv_dereg_mr(left_tile_mr);
        ibv_dereg_mr(right_tile_mr);

        ibv_dealloc_pd(left_pd);
        ibv_dealloc_pd(right_pd);

        ibv_close_device(dev_context);
    }

    free(left_qp);
    free(right_qp);
    if (xdynamic || dynamic || sharedd) {
        free(left_td);
        free(right_td);
        free(left_parent_d);
        free(right_parent_d);
    }
    free(cq);
    if (use_cq_ex)
        free(cq_ex);
    if (dedicated) {
        free(left_ded_mr);
        free(right_ded_mr);
        free(left_ded_pd);
        free(right_ded_pd);
        free(ded_ctx);
    }

    return 0;
}

int read_args(int argc, char *argv[])
{
    int ret = 0, op;
    int bench_type;

    struct option long_options[] = {
        {.name = "ib-dev",.has_arg = 1,.val = 'd'},
        {.name = "num-threads",.has_arg = 1,.val = 't'},
        {.name = "ga-dim-x",.has_arg = 1,.val = 'n'},
        {.name = "ga-dim-y",.has_arg = 1,.val = 'm'},
        {.name = "iterations",.has_arg = 1,.val = 'r'},
        {.name = "postlist",.has_arg = 1,.val = 'p'},
        {.name = "mod-comp",.has_arg = 1,.val = 'q'},
        {.name = "tx-depth",.has_arg = 1,.val = 'T'},
        {.name = "rx-depth",.has_arg = 1,.val = 'R'},
        {.name = "use-cq-ex",.has_arg = 0,.val = 'x'},
        {.name = "compute",.has_arg = 0,.val = 'C'},
        {.name = "dedicated",.has_arg = 0,.val = 'e'},
        {.name = "xdynamic",.has_arg = 0,.val = 'u'},
        {.name = "dynamic",.has_arg = 0,.val = 'v'},
        {.name = "sharedd",.has_arg = 0,.val = 'o'},
        {.name = "use-static",.has_arg = 0,.val = 'w'},
        //{.name = "use-td",            .has_arg = 0, .val = 'o'},
        {0, 0, 0, 0}
    };

    while (1) {
        op = getopt_long(argc, argv, "h?d:t:n:m:r:p:q:T:R:xCeuvow", long_options, NULL);
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
    dedicated = DEF_DEDICATED;
    xdynamic = DEF_XDYNAMIC;
    dynamic = DEF_DYNAMIC;
    sharedd = DEF_SHAREDD;
    use_static = DEF_USE_STATIC;

    ib_port = DEF_IB_PORT;

    return 0;
}
