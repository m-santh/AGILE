# 1. Define the parameters to match your 96GB dataset
cols_file="/home/ub-12-3/san/bam/AGILE/benchmarks/data/app-bench/bfs-rows.bin"
ss_bytes=512
sbn=201326592
queue_num=16
cache_page_size=131072
nvme_bar="/dev/AGILE-NVMe-0000:03:00.0"

# 2. Run the block-write command to initialize the file
sudo ./scripts/run.sh bin/block-write -i "$cols_file" -ss "$ss_bytes" -bn "$sbn" -qn "$queue_num" -gsn "$cache_page_size" -bar "$nvme_bar"
