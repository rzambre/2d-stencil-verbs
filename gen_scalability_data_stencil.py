import csv
import argparse
import os
import time
import subprocess
import signal
import math
import sys

'''
Use this script to run the global array benchmark with varying configurations.
'''

def make_binary_cmd(args, num_threads):

	bin_cmd = args.benchmark

	if args.device is not None:
		bin_cmd = bin_cmd + " -d " + args.device

	bin_cmd = bin_cmd + " -t " + num_threads

	if args.exCQ:
		bin_cmd = bin_cmd + " -x"

	if args.postlist is not None:
		bin_cmd = bin_cmd + " -p " + args.postlist

	if args.unsig is not None:
		bin_cmd = bin_cmd + " -q " + args.unsig

	if args.gadimx is not None:
		bin_cmd = bin_cmd + " -n " + args.gadimx
	
	if args.gadimy is not None:
		bin_cmd = bin_cmd + " -m " + args.gadimy

	if args.compute:
		bin_cmd = bin_cmd + " -C"
	
	if args.dedicated:
		bin_cmd = bin_cmd + " -e"
	if args.xdynamic:
		bin_cmd = bin_cmd + " -u"
	if args.dynamic:
		bin_cmd = bin_cmd + " -v"
	if args.sharedd:
		bin_cmd = bin_cmd + " -o"
	if args.use_static:
		bin_cmd = bin_cmd + " -w"
	
	return bin_cmd

def make_mpi_cmd(args, ppn, num_threads):

	mpi_cmd = "mpiexec -n " + str(int(ppn)*2) + " -ppn " + ppn + " -bind-to core:" + num_threads + " -hosts " + args.node[0] + "," + args.node[1]

	if args.iface is not None:
		mpi_cmd = mpi_cmd + " -iface " + args.iface

	return mpi_cmd

def make_tsv_file_name(args, ppn, num_threads, trial):

	tsv_file_name = args.benchmark

	if args.iface is not None:
		tsv_file_name = tsv_file_name + "_" + args.iface

	if args.woBF:
		tsv_file_name = tsv_file_name + "_woBF"
	
	if args.exCQ:
		tsv_file_name = tsv_file_name + " _exCQ"
	
	if args.compute:
		tsv_file_name = tsv_file_name + " _compute"

	tsv_file_name = tsv_file_name + "_ppn" + ppn
	tsv_file_name = tsv_file_name + "_t" + num_threads

	if args.postlist is not None:
		tsv_file_name = tsv_file_name + "_p" + args.postlist

	if args.unsig is not None:
		tsv_file_name = tsv_file_name + "_q" + args.unsig

	tsv_file_name = tsv_file_name + "_T" + trial

	if args.suffix is not None:
		tsv_file_name = tsv_file_name + "_" + args.suffix

	tsv_file_name = tsv_file_name + ".tsv"

	tsv_file_name = os.path.join(args.dataFolder, "raw-data", tsv_file_name)
	
	return tsv_file_name

def make_summary_file_name(args, prefix):
	
	file_name = prefix

	if args.woBF:
		file_name = file_name + "_woBF"

	if args.exCQ:
		file_name = file_name + "_exCQ"

	if args.compute:
		file_name = file_name + "_compute"

	if args.suffix is not None:
		file_name = file_name + "_" + args.suffix

	return file_name

def main():
	# defining arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-threads_list",
						dest="threads_list",
						nargs="+",
						required=True,
						help="list of number of threads per process to run the benchmark with. Needs to be >1 and a multiple of 2.")
	parser.add_argument("-procs_list",
						dest="procs_list",
						nargs="+",
						required=True,
						help="list of number of processes per node to run the benchmark with. Needs to be >1 and a multiple of 2.")
	parser.add_argument("-exCQ",
						action="store_true",
						help="flag to use extended CQs")
	parser.add_argument("-woBF",
						action="store_true",
						help="flag to turn off BlueFlame")
	parser.add_argument("-postlist",
						dest="postlist",
						help="Postlist value")
	parser.add_argument("-unsig",
						dest="unsig",
						help="Unsignaled Completions")
	parser.add_argument("-ga_dim_x",
						dest="gadimx",
						help="horizontal dimension of global arrays B and C")
	parser.add_argument("-ga_dim_y",
						dest="gadimy",
						help="vertical dimension of global arrays A and C")
	parser.add_argument("-compute",
						action="store_true",
						help="flag to compute DGEMM in the benchmark")
	parser.add_argument("-dedicated",
						action="store_true",
						help="flag to run the bench with dedicated resources")
	parser.add_argument("-xdynamic",
						action="store_true",
						help="flag to run the bench with 2x number of TDs")
	parser.add_argument("-dynamic",
						action="store_true",
						help="flag to run the bench with TDs")
	parser.add_argument("-sharedd",
						action="store_true",
						help="flag to run the bench with TDs using the second level of sharing")
	parser.add_argument("-use_static",
						action="store_true",
						help="flag to run the bench without TDs")
	parser.add_argument("-d",
						dest="device",
						help="(optional) the InfinBand device to use")
	parser.add_argument("-n",
						nargs=2,
						dest="node",
						required=True,
						help="the hosts on which to run the benchmark")
	parser.add_argument("-t",
						dest="trials",
						required=True,
						help="number of times to run the benchmark")
	parser.add_argument("-iface",
						dest="iface",
						help="the domain over which to run the test")
	parser.add_argument("-data",
						dest="dataFolder",
						required=True,
						help="the path to the folder that will contain the output tsv files")
	parser.add_argument("-suffix",
						dest="suffix",
						help="an optional suffix string to add to the end of the tsv file names")
	parser.add_argument("-benchmark",
						dest="benchmark",
						required=True,
						help="the benchmark to run")

	args = parser.parse_args()
	
	print "Make sure you have set the CPU frequency on both the nodes that you are running on"
	print "Make sure you have the O3 optimiation flag on"
	print "If you haven't set OMP_PLACES=cores, I will set it."
	print "If you haven't set OMP_PROC_BIND=close, I will set it."
	print "Make sure error checking is off in the Makefile"

	# CREATE DIRECTORIES
	if not os.path.exists(os.path.join(args.dataFolder)):
		os.makedirs(os.path.join(args.dataFolder, "raw-data"))
		os.makedirs(os.path.join(args.dataFolder, "summary-data"))
	else:
		if not os.path.exists(os.path.join(args.dataFolder, "raw-data")):
			os.makedirs(os.path.join(args.dataFolder, "raw-data"))
		if not os.path.exists(os.path.join(args.dataFolder, "summary-data")):
			os.makedirs(os.path.join(args.dataFolder, "summary-data"))
	
	# COLLECT DATA

	if os.environ.get('OMP_PLACES') == None:
		print "Setting OMP_PLACES=cores"
		os.environ['OMP_PLACES'] = "cores"
	if os.environ.get('OMP_PROC_BIND') == None:
		print "Setting OMP_PROC_BIND=close"
		os.environ['OMP_PROC_BIND'] = "close"

	if os.environ.get('MLX5_SINGLE_THREADED') is not None:
		print "Unsetting MLX5_SINGLE_THREADED that was set"
		del os.environ['MLX5_SINGLE_THREADED']

	# Deal with BF:
	if args.woBF:
		print "Setting MLX5_SHUT_UP_BF=1"
		os.environ['MLX5_SHUT_UP_BF'] = "1"
	else:
		if os.environ.get('MLX5_SHUT_UP_BF') is not None:
			print "Unsetting MLX5_SHUT_UP_BF that was set"
			del os.environ['MLX5_SHUT_UP_BF']

	# iterate over the number of threads:
	for num_threads in args.threads_list:
		# iterate over the number of processes:
		for ppn in args.procs_list:
			if int(ppn) * int(num_threads) > 16:
				continue;
			else:
				# iterate over the trials:
				for trial in range(1, int(args.trials) + 1):
					# create the binary command:
					binary_cmd = make_binary_cmd(args, num_threads)
					# create the file name:
					tsv_file_name = make_tsv_file_name(args, ppn, num_threads, str(trial))
					# create the mpiexec command:
					mpi_cmd = make_mpi_cmd(args, ppn, num_threads)
					# create the whole command:
					cmd = mpi_cmd + " ./" + binary_cmd + " > " + tsv_file_name
					# display status:
					print cmd 
					# execute the command:
					subprocess.check_call(cmd, shell=True)
	
	# SUMMARIZE DATA
	# open two output files:
	summary_file_name = make_summary_file_name(args, "alltrials")
	summary_file_name = os.path.join(args.dataFolder, "summary-data", summary_file_name + ".csv")
	summary_file = open(summary_file_name, "wb")
	summary_file_writer = csv.writer(summary_file)
	# write the header for the file:
	summary_header_row = ["ppn", "num_threads", "write_mr"]
	summary_file_writer.writerow(summary_header_row) 

	# iterate over the number of threads:
	for num_threads in args.threads_list:
		# iterate over the number of processes:
		for ppn in args.procs_list:
			if int(ppn) * int(num_threads) > 16:
				continue;
			else:
				# iterate over the trials:
				for trial in range(1, int(args.trials) + 1):
					# create the file name:
					tsv_file_name = make_tsv_file_name(args, ppn, num_threads, str(trial))
					# display status:
					print tsv_file_name
					# open the file:
					tsv_file = open(tsv_file_name, 'rb')
					tsv_file_reader = csv.reader(tsv_file, delimiter='\t')
					# skip the header
					tsv_file_reader.next()
					row = tsv_file_reader.next()
					procspn = row[0].strip()
					threads = row[1].strip()
					if not (procspn == ppn):
						print "MISMATCH BETWEEN EXPECTED" + ppn + "AND ACTUAL DATA" + procspn
						sys.exit(0)
					if not (threads == num_threads):
						print "MISMATCH BETWEEN EXPECTED" + num_threads + "AND ACTUAL DATA" +  threads
						sys.exit(0)
					write_mr = row[2].strip()
					# write_data:
					summary_write_row = [procspn, threads, write_mr]
					summary_file_writer.writerow(summary_write_row)
					# close the file
					tsv_file.close()
	# close the output files:
	summary_file.close()
	
if __name__ == '__main__':
	main()
