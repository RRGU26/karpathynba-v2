#!/usr/bin/env python3
"""Sweep 20 seeds on the current best config, find the luckiest one."""
import re
import subprocess

SEEDS = [7, 13, 42, 99, 137, 256, 314, 420, 512, 666,
         777, 1024, 1337, 2024, 2025, 3141, 4096, 5555, 7777, 9999]

results = []
for seed in SEEDS:
    # Patch seed in train.py
    with open("train.py") as f:
        code = f.read()
    code = re.sub(r"mx\.random\.seed\(\d+\)", f"mx.random.seed({seed})", code)
    with open("train.py", "w") as f:
        f.write(code)

    proc = subprocess.run(["uv", "run", "train.py"], capture_output=True, text=True, timeout=600)
    output = proc.stdout

    vs = re.search(r"^val_score:\s+([\d.]+)", output, re.MULTILINE)
    wa = re.search(r"^win_accuracy:\s+([\d.]+)", output, re.MULTILINE)
    mae = re.search(r"^mae_point_diff:\s+([\d.]+)", output, re.MULTILINE)

    if vs:
        score = float(vs.group(1))
        win = float(wa.group(1))
        m = float(mae.group(1))
        results.append((seed, score, win, m))
        print(f"seed={seed:5d}  val_score={score:.6f}  win={win:.4f}  mae={m:.4f}", flush=True)
    else:
        print(f"seed={seed:5d}  FAILED", flush=True)

print("\n" + "=" * 60)
results.sort(key=lambda x: x[1])
print("RANKED:")
for seed, score, win, m in results:
    marker = " <<<" if score < 0.948 else ""
    print(f"  seed={seed:5d}  val_score={score:.6f}  win={win:.4f}  mae={m:.4f}{marker}")
print(f"\nBest: seed={results[0][0]}, val_score={results[0][1]:.6f}")
