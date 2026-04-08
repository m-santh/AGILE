#!/bin/bash
set -euo pipefail

data_path=/home/ub-12-3/san/bam/AGILE/benchmarks/data
#datasets=( "$data_path/road" "$data_path/twitter" "$data_path/web" "$data_path/kron" "$data_path/urand" )
datasets=( "$data_path/road") #"$data_path/kron" "$data_path/urand" )

# One start node per dataset (MUST match datasets[] order)
#start_node=(5721937 26787574 7337451 74405778 39693157)
start_node=(5721937)

bam=baseline/bam/build/bin

nvme_bar="/dev/AGILE-NVMe-0000:03:00.0"
nvme_blk_size=1                      # in 512B units
ss_bytes=$((nvme_blk_size * 512))    # byte size per SSD block (e.g., 8*512=4096)

queue_num=16
queue_depth=256

# cache_page_size is in "pages" measured in (nvme_blk_size * 512) bytes
cache_page_size=$((128*1024 / nvme_blk_size))

save_path=./results/bfs/${nvme_blk_size}
repeat=1
max_ssd_page=$((2*1024*1024 / ${nvme_blk_size}))

# ---------- helpers ----------
ceil_div() { echo $(( ( $1 + $2 - 1 ) / $2 )); }

next_pow2() {
  local x=$1
  local p=1
  while (( p < x )); do p=$((p<<1)); done
  echo $p
}

compute_sbn_for_cols() {
  local cols_file="$1"
  local ss="$2"

  local cols_bytes
  cols_bytes=$(stat -c%s "$cols_file")

  local need_blocks
  need_blocks=$(ceil_div "$cols_bytes" "$ss")

  # add headroom: +5% and +1024 blocks to avoid off-by-one/alignment surprises
  local blocks_headroom=$(( need_blocks + need_blocks/20 + 1024 ))

  # round up to next power-of-two (nice for cache/table sizing)
  echo "$(next_pow2 "$blocks_headroom")"
}

# Optional: check degree(start_node) > 0 to avoid BFS finishing at level 0
degree_of_node() {
  local rows_file="$1"
  local node="$2"
  python3 - <<'PY' "$rows_file" "$node"
import sys, struct
rows, node = sys.argv[1], int(sys.argv[2])
u32 = struct.Struct("<I")
with open(rows, "rb") as f:
    f.seek(node*4)
    a = u32.unpack(f.read(4))[0]
    b = u32.unpack(f.read(4))[0]
print(b-a)
PY
}
# ----------------------------

mkdir -p "$save_path"

# sanity: start_node array length
if (( ${#start_node[@]} != ${#datasets[@]} )); then
  echo "ERROR: start_node count (${#start_node[@]}) != datasets count (${#datasets[@]})"
  exit 1
fi

for idx in "${!datasets[@]}"; do
  i="${datasets[$idx]}"
  u="$(basename "$i")"
  s_node="${start_node[$idx]}"

  rows_file="${i}/bfs-rows.bin"
  cols_file="${i}/bfs-cols.bin"
  info_file="${i}/bfs-info.txt"

  echo ""
  echo "=============================="
  echo "Dataset: $u"
  echo "Path:    $i"
  echo "Start:   $s_node"
  echo "=============================="

  # file checks
  [[ -f "$rows_file" ]] || { echo "ERROR: missing $rows_file"; exit 1; }
  [[ -f "$cols_file" ]] || { echo "ERROR: missing $cols_file"; exit 1; }
  [[ -f "$info_file" ]] || { echo "ERROR: missing $info_file"; exit 1; }

  info=($(cat "$info_file"))
  nn="${info[4]}"
  en="${info[7]}"

  echo "node $nn edge $en ($u)"

  # compute per-dataset sbn/bn from cols size
  sbn="$(compute_sbn_for_cols "$cols_file" "$ss_bytes")"
  cols_bytes=$(stat -c%s "$cols_file")
  need_blocks=$(( (cols_bytes + ss_bytes - 1) / ss_bytes ))
  echo "ss_bytes=$ss_bytes cols_bytes=$cols_bytes need_blocks=$need_blocks -> sbn=$sbn blocks"

  # optional degree check to avoid immediate BFS termination
  deg=$(degree_of_node "$rows_file" "$s_node" || echo "0")
  if [[ "$deg" -le 0 ]]; then
    echo "WARNING: start_node=$s_node degree=$deg for $u. BFS may terminate early (level 0)."
    echo "         Consider choosing a higher-degree root for this dataset."
  else
    echo "start_node degree=$deg"
  fi

  # Stage cols to SSD (use sbn as bn)
  echo "./scripts/run.sh bin/block-write -i $cols_file -ss $ss_bytes -bn $sbn -qn $queue_num -gsn $cache_page_size -bar $nvme_bar"
  #./scripts/run.sh bin/block-write -i "$cols_file" -ss "$ss_bytes" -bn "$sbn" -qn "$queue_num" -gsn "$cache_page_size" -bar "$nvme_bar"

  mkdir -p "${save_path}/${u}/bam/"
  mkdir -p "${save_path}/${u}/agile/"

  for ((r=0; r<repeat; r++)); do
    echo "./scripts/run.sh bin/bfs -nn $nn -sn $s_node -en $en -i $rows_file -gsn $cache_page_size -qn $queue_num -qd $queue_depth -bd 128 -ss $ss_bytes -sbn $sbn -bar $nvme_bar"

    ./scripts/run.sh bin/bfs \
      -nn "$nn" -sn "$s_node" -en "$en" \
      -i "$rows_file" \
      -gsn "$cache_page_size" \
      -qn "$queue_num" -qd "$queue_depth" \
      -bd 128 \
      -ss "$ss_bytes" -sbn "$sbn" \
      -bar "$nvme_bar" | tee log

    if [[ -f res-bfs.bin ]]; then
      md5sum res-bfs.bin >> log
    fi

    sudo mv log "${save_path}/${u}/agile/log-${r}"
  done
done
