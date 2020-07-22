CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O3# -g3
MACROS = #-DERRCHK
IFLAGS = -I/home/rzambre/base-rdma-core/build/include
LFLAGS = -L/home/rzambre/base-rdma-core/build/lib
LNAME = -libverbs
OMPFLAGS = -fopenmp
DEPS = shared.c

TARGETS = stencil_1d_put_effmt stencil_1d_put_soamt

stencil_1d_put_effmt: stencil_1d_put_effmt.c $(DEPS)
	$(MPICC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@ $(IFLAGS) $(LFLAGS) $(LNAME)

stencil_1d_put_soamt: stencil_1d_put_soamt.c $(DEPS)
	$(MPICC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@ $(IFLAGS) $(LFLAGS) $(LNAME)

clean:
	rm -f $(TARGETS)
