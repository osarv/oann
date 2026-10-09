# Two builds of bench/lm.olang run in alternation - A then B, then B then A, ... - and the medians (with the range) of
# each one's step, attention forward and backward, products and the rest of the graph, in ms a step: an A/B that
# a machine shared with other work does not tilt toward whichever build ran during a quiet spell.
# python3 -I bench/abstep.py ROUNDS STEPS THREADS BUILD_A BUILD_B [the model: layers width heads context sequences]
# A build may carry arguments of its own after commas, put after the model's: build/bench_lm,bf16
import re
import statistics
import subprocess
import sys

rounds, steps, threads = int(sys.argv[1]), sys.argv[2], sys.argv[3]
builds = sys.argv[4:6]
model = sys.argv[6:]
keys = ["step", "attention forward", "attention backward", "products", "rest"]
seen = {b: {k: [] for k in keys} for b in builds}
for r in range(rounds):
    for b in (builds if r % 2 == 0 else builds[::-1]):
        path, *own = b.split(",")
        out = subprocess.run([path, steps, threads] + model + own, capture_output=True, text=True, check=True).stdout
        d = seen[b]
        d["step"].append(float(re.search(r"a step ([\d.]+) ms", out).group(1)))
        m = re.search(r"^Attention\s+([\d.]+)\s+([\d.]+)", out, re.M)
        d["attention forward"].append(float(m.group(1)))
        d["attention backward"].append(float(m.group(2)))
        m = re.search(r"products ([\d.]+) ms, attention [\d.]+ ms, the rest of the graph ([\d.]+) ms", out)
        d["products"].append(float(m.group(1)))
        d["rest"].append(float(m.group(2)))
for b in builds:
    print(b + ": " + ", ".join(f"{k} {statistics.median(v):.1f} ({min(v):.1f}-{max(v):.1f})" for k, v in seen[b].items()))
