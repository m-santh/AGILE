#NVCC_FLAG :=  -w -Xptxas -v \
 				-arch=native \
 --default-stream per-thread -gencode=arch=compute_86,code=compute_86 \
-G -lineinfo

NVCC_FLAG := -O2 -lineinfo --default-stream per-thread \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86

INCLUDE_PATH := -I./include \
				-I./driver/gdrcopy/include \
				-I./common 
				
LIB_PATH := -L./driver/gdrcopy/src

LIBS := -lcuda -lgdrapi -lcublas -lcufile

bench-write:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/bench-write/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/bench-write

bench-read:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/bench-read/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/bench-read

api-graph-bfs:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/api-graph-bfs/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/api-graph-bfs

app-dlrm-criteo-sync-dram:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/app-dlrm-criteo-sync-dram/main.cu benchmarks/app-dlrm-criteo-sync-dram/runner.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/app-dlrm-criteo-sync-dram

app-dlrm-criteo-async-dram:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/app-dlrm-criteo-async-dram/main.cu benchmarks/app-dlrm-criteo-async-dram/runner.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/app-dlrm-criteo-async-dram

app-dlrm-criteo-async-profile:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/app-dlrm-criteo-async-profile/main.cu benchmarks/app-dlrm-criteo-async-profile/runner.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/app-dlrm-criteo-async-profile

app-dlrm-criteo-test:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/app-dlrm-criteo-test/main.cu benchmarks/app-dlrm-criteo-test/runner.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/app-dlrm-criteo-test


app-dlrm-criteo-sync:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/app-dlrm-criteo-sync/main.cu benchmarks/app-dlrm-criteo-sync/runner.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/app-dlrm-criteo-sync


app-dlrm-criteo-async:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/app-dlrm-criteo-async/main.cu benchmarks/app-dlrm-criteo-async/runner.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/app-dlrm-criteo-async


block-read:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/block-read-ssd2file/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/block-read

block-write:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/block-write-file2ssd/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/block-write

bfs:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/graph-bfs/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/bfs

pr:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/graph-pr/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/pr

ctc-block:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/async-block-ctc/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/ctc-block

app-bench:
	clear
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) benchmarks/app-bench/main.cu common/common.cpp $(LIB_PATH) $(LIBS) -o ./bin/app-bench

