#include <inttypes.h>

#define CACHE_LINE_SIZE             64

#define DEF_IB_PORT					1
#define DEF_NUM_THREADS				1
#define DEF_GA_DIM_X				512
#define DEF_GA_DIM_COMMON			512
#define DEF_GA_DIM_Y				512
#define DEF_TILE_DIM_X				4
#define DEF_TILE_DIM_COMMON			4
#define DEF_TILE_DIM_Y				4
#define DEF_ITERATIONS				20480
#define DEF_POSTLIST 				2
#define DEF_MOD_COMP 				2
#define DEF_TX_DEPTH				128
#define DEF_RX_DEPTH				512
#define DEF_USE_CQ_EX				0
#define DEF_COMPUTE					0

#define DEF_DEDICATED				0
#define DEF_XDYNAMIC				0
#define DEF_DYNAMIC					1
#define DEF_SHAREDD					0
#define DEF_USE_STATIC				0

enum {
	EFF_MT = 1,
	SOA_MT = 2,
	SOA_MPROC = 3,
};

enum {
	LEFT = 1,
	RIGHT = 2,
	TOP = 3,
	BOTTOM = 4, 
};

struct thread_flow_vars {
	char pre_padding[CACHE_LINE_SIZE];
	int tid;
	int post_count;
	int comp_count;
	int posts;
	int postlist;
	int mod_comp;
	int tx_depth;
	int cq_depth;
	int tx_decrement_val;
	int max_inline_data;
	struct ibv_sge *sge;
	struct ibv_send_wr *wqe;
	struct ibv_wc *wc;
	struct ibv_qp *my_qp;
	struct ibv_cq *my_cq;
	char post_padding[CACHE_LINE_SIZE];
};

struct stencil_thread_flow_vars {
	char pre_padding[CACHE_LINE_SIZE];
	int tid;
	int left_post_count;
	int right_post_count;
	int left_comp_count;
	int right_comp_count;
	int left_posts;
	int right_posts;
	int postlist;
	int mod_comp;
	int tx_depth;
	int cq_depth;
	int tx_decrement_val;
	int max_inline_data;
	struct ibv_sge *sge;
	struct ibv_send_wr *wqe;
	struct ibv_wc *wc;
	struct ibv_qp *left_qp;
	struct ibv_qp *right_qp;
	struct ibv_cq *my_cq;
	char post_padding[CACHE_LINE_SIZE];
};

struct tile {
	double *tile_arr;
	struct ibv_mr *mr;
	char padding[CACHE_LINE_SIZE - sizeof(double*) - sizeof(struct ibv_mr*)];
};

extern char* dev_name;
extern int num_threads;
extern int ga_dim_x;
extern int ga_dim_common;
extern int ga_dim_y;
extern int tile_dim_x;
extern int tile_dim_common;
extern int tile_dim_y;
extern int iterations;

extern int postlist;
extern int mod_comp;
extern int tx_depth;
extern int rx_depth;
extern int use_cq_ex;

extern int compute;
extern int type;
extern int dedicated;
extern int xdynamic;
extern int dynamic;
extern int sharedd;
extern int use_static;

int num_ctxs;
int num_pds;
int num_cqs;
int num_qps;

int cq_depth;

int ib_port;

int rank, size;

static inline int64_t min(int64_t a, int64_t b);
static inline int64_t max(int64_t a, int64_t b);
static inline uint64_t get_cycles();

void parse_args(int op, char *optarg);
void print_usage(const char *argv0, int type);
void show_perf(int *read_messages, int *write_messages,
				int *read_a_tile_counter, int *read_b_tile_counter, int *write_c_tile_counter,
				int tile_a_bytes, int tile_b_bytes, int tile_c_bytes,
				double *read_time, double *write_time, int num_threads);
void dgemm(double *local_a, double *local_b, double *local_c, int vertical, int common, int horizontal);
int lcm(int a, int b);
int gcd(int a, int b);

static inline int64_t max(int64_t a, int64_t b)
{
	return (a > b) ? a : b;
}

static inline int64_t min(int64_t a, int64_t b)
{
	return (a < b) ? a : b;
}

static inline uint64_t get_cycles()
{
	unsigned hi, lo;
	/* rdtscp modifies $ecx (=$rcx) register. */
	__asm__ __volatile__ ("rdtscp" : "=a"(lo), "=d"(hi) : : "rcx");
	uint64_t cycle = ((uint64_t)lo) | (((int64_t)hi) << 32);
	return cycle;
}
