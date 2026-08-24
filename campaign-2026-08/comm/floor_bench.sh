#!/bin/bash
# floor_bench.sh - RDMA all-gather floor at DECODE-COLLECTIVE SHAPES, both rails.
# Shapes from the production decode step (kit docs/COMM_FLOOR + decode profile):
#   37 KB  ~ TP all-reduce msg @ C1 (compared vs NCCL AR 37KB)
#   55 KB  ~ q-head all-gather per rank
#   96 KB  ~ indexer candidate gather
#   150 KB ~ TP all-reduce msg @ C4
# Runs ag4_proto (RDMA WRITE-with-imm full-mesh AG) 4 ranks, 3000 iters,
# median of last 2500. Both rails. Output: /tmp/floor_results.txt on gx10-1.
set -u
IPS_A=(192.168.100.11 192.168.100.12 192.168.100.13 192.168.100.14)   # rail A
IPS_B=(192.168.101.11 192.168.101.12 192.168.101.13 192.168.101.14)   # rail B
NODES=(gx10-1 gx10-2 gx10-3 gx10-4)
OUT=/tmp/floor_results.txt
ITERS=3000

echo "shape_bytes rail median_us best_us" > $OUT

run_sweep() {
  local rail=$1; shift
  local ips=("$@")
  for chunk in 37888 56320 98304 153600; do   # 37K 55K 96K 150K
    [ $chunk != 37888 ] && sleep 70   # let rdma listen sockets exit TIME_WAIT
    # launch ranks 1..3 detached, rank 0 in foreground
    for i in 1 2 3; do
      ssh -o StrictHostKeyChecking=no ${NODES[$i]} \
        "pkill -f ag4_proto 2>/dev/null; sleep 1; /tmp/ag4_proto $i 4 $chunk $ITERS ${ips[0]} ${ips[1]} ${ips[2]} ${ips[3]} > /tmp/ag4_r$i.log 2>&1"
    done
    sleep 2
    ssh ${NODES[0]} "/tmp/ag4_proto 0 4 $chunk $ITERS ${ips[0]} ${ips[1]} ${ips[2]} ${ips[3]} 2>&1 | tail -1" >> $OUT
    # collect peer lines
    for i in 1 2 3; do ssh ${NODES[$i]} "tail -1 /tmp/ag4_r$i.log" >> $OUT; done
    echo "---" >> $OUT
  done
}

echo "== building ==" 
for n in "${NODES[@]}"; do
  scp -q "/Users/gigachad/Desktop/GLM Collab/glm52-spark-kit/kernels/rdma/ag4_proto.c" $n:/tmp/ 2>/dev/null || true
  ssh $n 'gcc -O2 -o /tmp/ag4_proto /tmp/ag4_proto.c -libverbs -lrdmacm && echo built'
done

ssh ${NODES[0]} "pkill -f ag4_proto 2>/dev/null"; echo "== rail A ==" >> $OUT
run_sweep A "${IPS_A[@]}"
ssh ${NODES[0]} "pkill -f ag4_proto 2>/dev/null"; echo "== rail B ==" >> $OUT
run_sweep B "${IPS_B[@]}"

cat $OUT
