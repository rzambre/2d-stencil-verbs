#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "shared.h"

char *dev_name;
int num_threads;
int ga_dim_x;
int ga_dim_common;
int ga_dim_y;
int tile_dim_x;
int tile_dim_common;
int tile_dim_y;
int iterations;

int postlist;
int mod_comp;
int tx_depth;
int rx_depth;
int use_cq_ex;

int compute;
int type;
int dedicated;
int xdynamic;
int dynamic;
int sharedd;
int use_static;

void print_usage(const char *argv0, int type)
{
	switch (type) {
		case SOA_MPROC:
			printf("Global array kernel using State-of-the-art Multi-process network configuration\n");
			break;
		case SOA_MT:
			printf("Global array kernel using State-of-the-art Multi-thread network configuration\n");
			break;
		case EFF_MT:
			printf("Global array kernel using Efficient Multi-thread network configuration\n");
			break;
		default:
			break;
	}
	printf("\n");
	printf("Usage:\n");
	//printf("  OMP_PLACES=cores OMP_PROC_BIND=close mpiexec -n 2 -ppn 1 -bind-to core:<num-threads> -hosts <sender>,<receiver> %s <options>\n", argv0);
	printf("Options:\n");
	printf("  -d, --ib-dev <>     		use IB device <dev> (default first device found)\n");
	printf("  -t, --num-threads <>		number of threads to use (default 1)\n");
	printf("  -n, --ga-dim-x <>   		length of horizontal dimension of the global array (default 1024)\n");
	printf("  -c, --ga-dim-comm <> 		length of common dimension of A and B (default 1024)\n");
	printf("  -m, --ga-dim-y <>   		length of vertical dimension of the global array (default 1024)\n");
	printf("  -j, --tile-dim-x <> 		length of horizontal dimension of the tile (default 2)\n");
	printf("  -k, --tile-dim-comm <>	length of common dimension of the tile (default 2)\n");
	printf("  -i, --tile-dim-y <>    	length of vertical dimension of the tile (default 2)\n");
	printf("  -y, --iterations <>    	number of iterations (default 10)\n");
	printf("  -p, --postlist <>			number of WQEs to post per ibv_post_send (default 32)\n");
	printf("  -q, --mod-comp <>			number of WQEs per signaled completion (default 64)\n");
	printf("  -T, --tx-depth <>			depth of the SQ in the QP (default 128)\n");
	printf("  -R, --rx-depth <>			depth of the RQ in the QP (default 512)\n");
	printf("  -x, --use-cq-ex <>		use an extended CQ (default NO)\n");
	printf("  -C, --compute <>			flag to turn on compute (default OFF)\n");
}

void parse_args(int op, char *optarg)
{
	switch (op) {
		case 'd':
			dev_name = strdup(optarg);
			break;
		case 't':
			num_threads = atoi(optarg);
			break;
		case 'n':
			ga_dim_x = atoi(optarg);
			break;
		case 'c':
			ga_dim_common = atoi(optarg);
			break;
		case 'm':
			ga_dim_y = atoi(optarg);
			break;
		case 'j':
			tile_dim_x = atoi(optarg);
			break;
		case 'k':
			tile_dim_common = atoi(optarg);
			break;
		case 'i':
			tile_dim_y = atoi(optarg);
			break;
		case 'y':
			iterations = atoi(optarg);
			break;
		case 'p':
			postlist = atoi(optarg);
			break;
		case 'q':
			mod_comp = atoi(optarg);
			break;
		case 'T':
			tx_depth = atoi(optarg);
			break;
		case 'R':
			rx_depth = atoi(optarg);
			break;
		case 'x':
			use_cq_ex = 1;
			break;
		case 'C':
			compute = 1;
			break;
		case 'e':
			dedicated = 1;
			xdynamic = 0;
			dynamic = 0;
			sharedd = 0;
			use_static = 0;
			break;
		case 'u':
			dedicated = 0;
			xdynamic = 1;
			dynamic = 0;
			sharedd = 0;
			use_static = 0;
			break;
		case 'v':
			dedicated = 0;
			xdynamic = 0;
			dynamic = 1;
			sharedd = 0;
			use_static = 0;
			break;
		case 'o':
			dedicated = 0;
			xdynamic = 0;
			dynamic = 0;
			sharedd = 1;
			use_static = 0;
			break;
		case 'w':
			dedicated = 0;
			xdynamic = 0;
			dynamic = 0;
			sharedd = 0;
			use_static = 1;
			break;
	}
}

void show_perf(int *read_messages, int *write_messages,
		int *read_a_tile_counter, int *read_b_tile_counter, int *write_c_tile_counter, 
		int tile_a_bytes, int tile_b_bytes, int tile_c_bytes,
		double *read_time, double *write_time, int num_threads)
{
	int tid;
	double read_mr, write_mr, tot_read_mr, tot_write_mr;
   	double read_bw, write_bw, tot_read_bw, tot_write_bw;
	
	printf("%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\n",
			"Thread", "Read s", "Read Mmsgs/s", "Read MB/s",
			"Write s", "Write Mmsgs/s", "Write MB/s");

	tot_read_mr = 0;
	tot_write_mr = 0;
	tot_read_bw = 0;
	tot_write_bw = 0;
	for (tid = 0; tid < num_threads; tid++) {
		read_mr = (double) read_messages[tid] / read_time[tid] / 1e6;
		write_mr = (double) write_messages[tid] / write_time[tid] / 1e6;

		read_bw = ((double) read_a_tile_counter[tid] * tile_a_bytes
				+ (double) read_b_tile_counter[tid] * tile_b_bytes) / read_time[tid] / 1e6;
		write_bw = (double) write_c_tile_counter[tid] * tile_c_bytes / write_time[tid] / 1e6;
		

		printf("%-10d\t", tid);
		printf("%-10.2f\t", read_time[tid]);
		printf("%-10.2f\t", read_mr);
		printf("%-10.2f\t", read_bw);
		printf("%-10.2f\t", write_time[tid]);
		printf("%-10.2f\t", write_mr);
		printf("%-10.2f\n", write_bw);
		
		tot_read_mr 	+= read_mr;
		tot_write_mr 	+= write_mr;
		tot_read_bw 	+= read_bw;
		tot_write_bw	+= write_bw;
	}

	printf("\n");

	printf("%-10s\t%-10s\t%-10s\t%-10s\t%-10s\n",
			"Threads", "Read Mmsgs/s", "Read MB/s",
			"Write Mmsgs/s", "Write MB/s");

	printf("%-10d\t", num_threads);
	printf("%-10.2f\t%-10.2f\t%-10.2f\t%-10.2f\n", tot_read_mr, tot_read_bw, tot_write_mr, tot_write_bw);
}

void dgemm(double *local_a, double *local_b, double *local_c, int vertical, int common, int horizontal)
{
	int i, j, k;

	memset(local_c, 0, vertical * horizontal * sizeof *local_c);

	for (i = 0; i < vertical; i++) {
		for (j = 0; j < horizontal; j++) {
			for (k = 0; k < common; k++) {
				local_c[i*horizontal + j] += local_a[i * common + k] * local_b[k * horizontal + j];
			}
		}
	}
}

/* Recursive function to return gcd of a and b */
int gcd(int a, int b)
{
	// base case
	if (a == b)
		return a;
	
	// a is greater
	if (a > b)
		return gcd(a-b, b);
	return gcd(a, b-a);
}

/* LCM of two numbers */
int lcm(int a, int b)
{
	return (a*b)/gcd(a,b);
}
