#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# The previous release against this version, in the 3:1 SWA:SDM setting:
#   equivalence  benchmarks/equivalence.py on identical weights and inputs (exits non-zero
#                unless the verdict is EQUIVALENT), and this version's cuda_graph option
#                against its eager path (a separate verdict, VERIFIED or NOT VERIFIED)
#   warm         untimed runs that compile what the timing needs (the SWA kernels of the
#                six model shapes, each version's SDM kernels, the decode step) and write the
#                decode weights, on GPUS
#   timing       refuses to run without an EQUIVALENT verdict for the same two commits and
#                the same harness files (the cells of this version's cuda_graph option run
#                only if the option's verdict is VERIFIED). The cells:
#                  32K tokens per step: B x T = 1 x 32K, 2 x 16K and 4 x 8K for
#                  bench_sdm.py, one SDM layer (C = 64): eager and as a whole-step CUDA graph,
#                  in separate processes, both versions; and eager with this version's
#                  cuda_graph option;
#                  bench_sdm_gdn.py, 6 x [SWA, SWA, SWA, SDM] with a fused AdamW step at SWA
#                  window 128: the same three ways;
#                  bench_decode.py, sdm_flagship.yaml's model (one set of random weights for
#                  both versions) at SWA window 128: the previous release decoding eagerly,
#                  this version eagerly and with its CUDA-graph decode step.
#                The warm pass runs first (WARM=0 skips it). Then REPEATS rounds of fresh
#                processes in two phases:
#                  graph cells (whole-step CUDA graphs, GPU-bound): one lane per GPU of GPUS,
#                  each pinned to its share of its GPU's CPUs;
#                  then the host-bound cells (eager, the cuda_graph option, decoding): one
#                  lane per CPU socket (NUMA node), on the first GPU of GPUS on that socket,
#                  pinned to all of the socket's CPUs and to its memory, so that nothing else
#                  runs on the socket.
#                In a round, both versions of a cell run back to back on one lane (the first
#                one alternating), and the cells rotate over the lanes between rounds.
#
# Usage: benchmarks/run_release_compare.sh OLD NEW OUT {equivalence|warm|timing}
#   OLD, NEW  checkouts whose SDM code is compared (e.g. a worktree of the previous release
#             and this one); the scripts always come from this checkout's benchmarks/
#   OUT       results directory; each version gets its own TORCH_EXTENSIONS_DIR and
#             TRITON_CACHE_DIR under ${CACHE_ROOT:-$OUT/cache}/{old,new}
# Env: VERDICT (timing; default $OUT/equivalence/verdict.json), GPUS (default: all visible),
#      REPEATS (6), ONLY (timing: a shell pattern of the cell names to run, e.g. '*opt_*'),
#      WARM (timing: 0 skips the warm pass), DECODE_CONFIG (default: NEW's
#      apps/main/configs/sdm_flagship.yaml), PY (python).
set -u
OLD=$(cd "$1" && pwd); NEW=$(cd "$2" && pwd); OUT=$3; WHAT=$4
BENCH=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
REPEATS=${REPEATS:-6}
ONLY=${ONLY:-*}
CACHE_ROOT=${CACHE_ROOT:-$OUT/cache}
mkdir -p "$OUT/logs"
export PYTHONDONTWRITEBYTECODE=1

# with_version old|new CMD...: CMD with that version's checkout first on PYTHONPATH and its caches
with_version() {
  local v=$1; shift
  local root=$OLD; [ "$v" = new ] && root=$NEW
  mkdir -p "$CACHE_ROOT/$v/torch_ext" "$CACHE_ROOT/$v/triton"
  PYTHONPATH=$root TORCH_EXTENSIONS_DIR=$CACHE_ROOT/$v/torch_ext \
    TRITON_CACHE_DIR=$CACHE_ROOT/$v/triton "$@"
}

if [ "$WHAT" = equivalence ]; then
  E=$OUT/equivalence; mkdir -p "$E"
  set -e
  with_version new $PY "$BENCH/equivalence.py" make --out "$E/inputs.pt"
  with_version old $PY "$BENCH/equivalence.py" run --inputs "$E/inputs.pt" --out "$E/old.pt"
  with_version new $PY "$BENCH/equivalence.py" run --inputs "$E/inputs.pt" --out "$E/new.pt"
  $PY "$BENCH/equivalence.py" oracle --inputs "$E/inputs.pt" --out "$E/oracle.pt"
  with_version old $PY "$BENCH/equivalence.py" run-model --inputs "$E/inputs.pt" --out "$E/old_model.pt"
  with_version new $PY "$BENCH/equivalence.py" run-model --inputs "$E/inputs.pt" --out "$E/new_model.pt"
  for dt in bf16 fp32; do  # one process per file: the model's graphs start from a fresh process
    for mode in free frozen; do
      with_version new $PY "$BENCH/equivalence.py" run-cudagraph --inputs "$E/inputs.pt" --out "$E" \
        --cg-dtype $dt --cg-mode $mode
    done
  done
  with_version new $PY "$BENCH/equivalence.py" compare --dir "$E" --a old --b new --out "$E/verdict.json"
  exit 0
fi
case $WHAT in warm|timing) ;; *) echo "usage: $0 OLD NEW OUT {equivalence|warm|timing}"; exit 2 ;; esac

# The paper's SDM layer at the README's 1.4B scale: dim 2048, 2 heads of 1024 (65536 slots
# per head, product keys 2 x 256), W = R = 64, C = 64; bf16, TF32 off.
LAYER="--mode train --dim 2048 --hd 1024 --W 64 --R 64 --sph 65536 --C 64"
MODEL="--mixer sdm --n-blocks 6 --dim 2048 --sdm-heads 2 --slots-per-head 65536 --writes 64 --reads 64 --chunk 64 --optimizer adamw"
SHAPES=("1 32768" "2 16384" "4 8192")  # B x T of the layer and model cells: 32K tokens per step
SWA=128  # the model's SWA window: that of the shipped configs (the paper's)
DECODE_CONFIG=${DECODE_CONFIG:-$NEW/apps/main/configs/sdm_flagship.yaml}
DECODE_WEIGHTS=$CACHE_ROOT/decode_weights.pt  # written by the first decode run, loaded by the others
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd' ')}
read -r -a G <<< "$GPUS"
n=${#G[@]}

warm_pass() {  # untimed runs, spread over GPUS, so that no timed process compiles
  local W=$OUT/logs/warm k j v script args bt jobs=()
  for bt in "${SHAPES[@]}"; do
    set -- $bt
    jobs+=("new bench_sdm_gdn.py $MODEL --swa-window $SWA --batch $1 --seqlen $2 --mode graph --warmup 1 --iters 1"
           "old bench_sdm_gdn.py $MODEL --swa-window $SWA --batch $1 --seqlen $2 --mode eager --warmup 1 --iters 1"
           "old bench_sdm.py $LAYER --B $1 --T $2 --warmup 1 --iters 1"
           "new bench_sdm.py $LAYER --B $1 --T $2 --warmup 1 --iters 1")
  done
  jobs+=("new bench_sdm.py $LAYER --B 4 --T 8192 --sdm-cuda-graph --warmup 1 --iters 1"
         "new bench_sdm_gdn.py $MODEL --swa-window $SWA --batch 4 --seqlen 8192 --mode eager --sdm-cuda-graph --warmup 1 --iters 1"
         "old bench_decode.py --config $DECODE_CONFIG --swa-window $SWA --weights $DECODE_WEIGHTS --contexts 1024 --gen 8 --warm 2 --repeats 1")
  mkdir -p "$W"
  # First, alone: this version's decode run, which writes the decode weights both versions load.
  [ -f "$DECODE_WEIGHTS" ] || CUDA_VISIBLE_DEVICES=${G[0]} with_version new $PY "$BENCH/bench_decode.py" \
    --config "$DECODE_CONFIG" --swa-window 128 --weights "$DECODE_WEIGHTS" --contexts 1024 --gen 8 \
    --warm 2 --repeats 1 --allow-busy --out "$W/warm_weights.json" > "$W/warm_weights.log" 2>&1 \
    || echo "warm decode weights failed (see $W/warm_weights.log)"
  for k in $(seq 0 $((n - 1))); do
    ( for j in $(seq "$k" "$n" $((${#jobs[@]} - 1))); do
        read -r v script args <<< "${jobs[$j]}"
        CUDA_VISIBLE_DEVICES=${G[$k]} with_version "$v" $PY "$BENCH/$script" $args --allow-busy \
          --out "$W/warm_$j.json" > "$W/warm_$j.log" 2>&1 || echo "warm job $j failed (see $W/warm_$j.log)"
      done ) &
  done
  wait
}
if [ "$WHAT" = warm ]; then warm_pass; exit 0; fi

VERDICT=${VERDICT:-$OUT/equivalence/verdict.json}
# The verdict must be EQUIVALENT and for the same two commits and harness files; the cells of
# the cuda_graph option run only if its separate verdict is VERIFIED.
OPTION_OK=$(BENCH_DIR=$BENCH with_version new $PY - "$VERDICT" "$(git -C "$OLD" rev-parse HEAD)" \
    "$(git -C "$NEW" rev-parse HEAD)" <<'EOF'
import json, os, sys
sys.path.insert(0, os.environ["BENCH_DIR"])
from provenance import harness_sha256
v = json.load(open(sys.argv[1]))
bad = [m for m, c in (("verdict", v.get("verdict") == "EQUIVALENT"),
                      ("old commit", v["a"]["commit"] == sys.argv[2]),
                      ("new commit", v["b"]["commit"] == sys.argv[3]),
                      ("harness", v.get("harness_sha256") == harness_sha256())) if not c]
if bad:
    sys.exit(f"no usable equivalence verdict in {sys.argv[1]}: {bad}")
print(1 if v.get("cuda_graph_option", {}).get("verdict") == "VERIFIED" else 0)
EOF
) || exit 3
echo "equivalence verdict OK: $VERDICT (cuda_graph option verified: $OPTION_OK)"

# name|versions|kind|script and arguments. Kind "graph": timed in the parallel phase; kind
# "eager": host-bound, timed one process per CPU socket.
CELLS=()
for bt in "${SHAPES[@]}"; do
  set -- $bt
  l="bench_sdm.py $LAYER --B $1 --T $2"
  m="bench_sdm_gdn.py $MODEL --swa-window $SWA --batch $1 --seqlen $2"
  CELLS+=("layer_B$1_T$2_C64_eager|old new|eager|$l --no-graph"
          "layer_B$1_T$2_C64_graph|old new|graph|$l --no-eager"
          "layeropt_B$1_T$2_C64|new|eager|$l --sdm-cuda-graph"
          "model_W${SWA}_B$1_T$2_eager|old new|eager|$m --mode eager"
          "model_W${SWA}_B$1_T$2_graph|old new|graph|$m --mode graph"
          "modelopt_W${SWA}_B$1_T$2|new|eager|$m --mode eager --sdm-cuda-graph")
done
CELLS+=("decode_W$SWA|old new|eager|bench_decode.py --config $DECODE_CONFIG --swa-window $SWA --weights $DECODE_WEIGHTS")

# Per GPU: its PCI bus id, local CPUs and NUMA node.
declare -A BUS CPUS NODE
for k in "${!G[@]}"; do
  BUS[$k]=$(nvidia-smi -i "${G[$k]}" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-Z' 'a-z' | sed 's/^0000\(....\)/\1/')
  CPUS[$k]=$(cat "/sys/bus/pci/devices/${BUS[$k]}/local_cpulist" 2>/dev/null || echo "")
  NODE[$k]=$(cat "/sys/bus/pci/devices/${BUS[$k]}/numa_node" 2>/dev/null || echo -1)
done
cpus_for() {  # k -> a CPU list of its own (shared local lists are split by GPU order)
  local k=$1 list=${CPUS[$1]} peers=() i=0 me=0
  [ -z "$list" ] && { echo ""; return; }
  for j in "${!G[@]}"; do [ "${CPUS[$j]}" = "$list" ] && { [ "$j" = "$k" ] && me=$i; peers+=("$j"); i=$((i + 1)); }; done
  $PY - "$list" "${#peers[@]}" "$me" <<'EOF'
import sys
spec, parts, me = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cpus = []
for r in spec.split(","):
    a, _, b = r.partition("-")
    cpus += list(range(int(a), int(b or a) + 1))
per = len(cpus) // parts
print(",".join(map(str, cpus[me * per:(me + 1) * per])))
EOF
}
# The host-bound lanes: the first GPU of GPUS on each socket, with all of its CPUs and memory.
EG=(); EPIN=()
declare -A SEEN
for k in "${!G[@]}"; do
  key=${NODE[$k]}; [ "$key" -ge 0 ] 2>/dev/null || key="cpus ${CPUS[$k]}"
  [ -n "${SEEN[$key]:-}" ] && continue
  SEEN[$key]=1
  EG+=("${G[$k]}")
  if [ -n "${CPUS[$k]}" ] && [ "${NODE[$k]}" -ge 0 ] 2>/dev/null && command -v numactl > /dev/null; then
    EPIN+=("numactl --physcpubind=${CPUS[$k]} --membind=${NODE[$k]}")
  elif [ -n "${CPUS[$k]}" ]; then
    EPIN+=("taskset -c ${CPUS[$k]}")
  else
    EPIN+=("")
  fi
done

# rounds KIND LIST LANES: one line per process, "<lane> <name> <version> <script and args>".
# Round r puts the i-th cell of KIND on lane (i + r) mod LANES, both versions back to back,
# the first version alternating with r + i.
rounds() {
  local kind=$1 list=$2 nl=$3 r i cell name versions k cmd order v
  : > "$list"
  for r in $(seq 0 $((REPEATS - 1))); do
    i=0
    for cell in "${CELLS[@]}"; do
      IFS='|' read -r name versions k cmd <<< "$cell"
      [ "$k" = "$kind" ] || continue
      case $name in *opt_*) [ "$OPTION_OK" = 1 ] || continue ;; esac
      # shellcheck disable=SC2254
      case $name in $ONLY) ;; *) continue ;; esac
      if [ "$versions" = "new" ]; then order="new"
      elif [ $(((r + i) % 2)) -eq 0 ]; then order="old new"; else order="new old"; fi
      for v in $order; do echo "$(((i + r) % nl)) ${name}_r$r $v $cmd" >> "$list"; done
      i=$((i + 1))
    done
  done
}
FAILED=$OUT/logs/failed.txt; : > "$FAILED"
lane() {  # lane LIST K GPU [PIN...]: the lines of LIST for lane K, in order, on GPU
  local list=$1 k=$2 gpu=$3 lk name v script args; shift 3
  echo "lane $k: gpu $gpu, ${*:-unpinned}"
  while read -r lk name v script args; do
    [ "$lk" -eq "$k" ] || continue
    echo "$(date -u +%FT%TZ) gpu $gpu $name $v"
    CUDA_VISIBLE_DEVICES=$gpu with_version "$v" "$@" $PY "$BENCH/$script" $args \
      --out "$OUT/$v/$name.json" > "$OUT/logs/${name}_$v.log" 2>&1 \
      || { echo "  failed: $name $v"; echo "$name $v" >> "$FAILED"; }
  done < "$list"
}

[ "${WARM:-1}" = 0 ] || warm_pass
lanes_ok=1
GLIST=$OUT/logs/graph_commands.txt; rounds graph "$GLIST" "$n"
echo "graph phase $(date -u +%FT%TZ): $(wc -l < "$GLIST") processes over GPUs ${G[*]}"
pids=()
for k in $(seq 0 $((n - 1))); do
  cpus=$(cpus_for "$k")
  if [ -n "$cpus" ]; then
    lane "$GLIST" "$k" "${G[$k]}" taskset -c "$cpus" > "$OUT/logs/graph_lane$k.log" 2>&1 &
  else
    lane "$GLIST" "$k" "${G[$k]}" > "$OUT/logs/graph_lane$k.log" 2>&1 &
  fi
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || lanes_ok=0; done
ne=${#EG[@]}
ELIST=$OUT/logs/eager_commands.txt; rounds eager "$ELIST" "$ne"
echo "host-bound phase $(date -u +%FT%TZ): $(wc -l < "$ELIST") processes over GPUs ${EG[*]} (one per CPU socket)"
pids=()
for k in $(seq 0 $((ne - 1))); do
  # shellcheck disable=SC2086
  lane "$ELIST" "$k" "${EG[$k]}" ${EPIN[$k]} > "$OUT/logs/eager_lane$k.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || lanes_ok=0; done
echo "timing done $(date -u +%FT%TZ); failed processes: $(wc -l < "$FAILED"); lanes ok: $lanes_ok"
[ "$lanes_ok" = 1 ] && [ ! -s "$FAILED" ] || exit 1
