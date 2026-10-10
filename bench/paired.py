# Paired comparison of two runs of the same lives (examples/bandit_settle.olang or examples/nights.olang with
# perlife=1): every "life K x1 x2 ..." line of each, matched by K, and for each column the mean of A, of B and of B - A
# life by life, each with the half-width of its 95% interval (1.96 standard errors). Columns are the lives' block
# regrets, then (nights.olang) the test's answers and recalls on seen, new, noisy and partial observations.
# python3 -I bench/paired.py A.txt B.txt [block] [columns] [odd|even]  - block (250) labels the block columns by their
# last moment; columns, as first:last (Python's slice; ":" for all), picks some; odd or even keeps those lives only
import math
import sys


def lives(path):
    out = {}
    for line in open(path):
        if line.startswith("life "):
            parts = line.split()
            out[int(parts[1])] = [float(x) for x in parts[2:]]
    return out


def interval(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


a, b = lives(sys.argv[1]), lives(sys.argv[2])
block = int(sys.argv[3]) if len(sys.argv) > 3 else 250
keys = sorted(set(a) & set(b))
if len(sys.argv) > 5:
    keys = [k for k in keys if k % 2 == (1 if sys.argv[5] == "odd" else 0)]
width = min(len(a[k]) for k in keys)
cols = range(width)
if len(sys.argv) > 4:
    lo, hi = sys.argv[4].split(":")
    cols = range(width)[slice(int(lo) if lo else None, int(hi) if hi else None)]
print(f"{len(keys)} lives paired")
print(f"{'column':>10}  {'A':>16}  {'B':>16}  {'B - A':>16}")
for c in cols:
    xa = [a[k][c] for k in keys]
    xb = [b[k][c] for k in keys]
    d = [y - x for x, y in zip(xa, xb)]
    ma, ha = interval(xa)
    mb, hb = interval(xb)
    md, hd = interval(d)
    print(f"{c:>3} {(c + 1) * block:>6}  {ma:7.3f} +- {ha:5.3f}  {mb:7.3f} +- {hb:5.3f}  {md:+7.3f} +- {hd:5.3f}")
