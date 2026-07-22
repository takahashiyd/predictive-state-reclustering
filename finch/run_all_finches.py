# -*- coding: utf-8 -*-
"""Run predictive-state reclustering on every Bengalese finch (Bird0..Bird10).

Downloads any missing data first, then runs finch_recluster.py once per bird as a
subprocess. Each run appends a row to finch_results.csv (deleted here first for a
clean full-run table). Also collects the SUMMARY| lines into finch_summary.txt.

Usage:
  python run_all_finches.py                 # all 11
  python run_all_finches.py Bird6 Bird7      # a subset
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BIRDS = ["Bird%d" % i for i in range(11)]
PY = sys.executable


def main():
    birds = sys.argv[1:] or BIRDS

    # 1. ensure data present
    subprocess.run([PY, os.path.join(HERE, "download_finch.py"), *birds], check=True, cwd=HERE)

    # 2. fresh results file for a full run over the default set
    csvp = os.path.join(HERE, "finch_results.csv")
    if not sys.argv[1:] and os.path.exists(csvp):
        os.remove(csvp)

    # 3. run each bird
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    summ = []
    for b in birds:
        print("\n" + "=" * 70 + "\n RUN " + b + "\n" + "=" * 70)
        p = subprocess.run([PY, os.path.join(HERE, "finch_recluster.py"), b],
                           cwd=HERE, env=env, capture_output=True, text=True)
        sys.stdout.write(p.stdout)
        if p.returncode != 0:
            sys.stderr.write("[FAIL %s]\n%s\n" % (b, p.stderr))
            continue
        for line in p.stdout.splitlines():
            if line.startswith("SUMMARY|"):
                summ.append(line)

    with open(os.path.join(HERE, "finch_summary.txt"), "w") as f:
        f.write("\n".join(summ) + "\n")
    print("\n[all done] %d/%d birds summarised -> finch_summary.txt, finch_results.csv"
          % (len(summ), len(birds)))


if __name__ == "__main__":
    main()
