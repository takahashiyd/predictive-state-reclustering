# -*- coding: utf-8 -*-
"""Download + extract the Koumura & Okanoya (2016) BirdsongRecognition corpus.
figshare article 3470165, 11 per-bird zips (~1.49 GB total), tokenless.

Lays data out as  <this folder>/data/BirdN/{Annotation.xml, Wave/*.wav}  -- exactly what
finch_recluster.py expects. Idempotent: a bird already extracted is skipped. Zips are
deleted after extraction to save space. Stdlib only (urllib + zipfile).

Usage:
  python download_finch.py            # all 11 birds
  python download_finch.py Bird6 Bird7 ...   # only these
"""
import os, sys, zipfile, urllib.request, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ZIPS = os.path.join(DATA, "_zips")

# figshare file ids (from api.figshare.com/v2/articles/3470165)
FILE_ID = {
    "Bird0": 5463221, "Bird1": 5463224, "Bird2": 5463227, "Bird3": 5463230,
    "Bird4": 5463233, "Bird5": 5463236, "Bird6": 5463239, "Bird7": 5463242,
    "Bird8": 5463245, "Bird9": 5463248, "Bird10": 5463251,
}


def have(bird):
    return os.path.isfile(os.path.join(DATA, bird, "Annotation.xml"))


def fetch(bird):
    os.makedirs(ZIPS, exist_ok=True)
    fid = FILE_ID[bird]
    url = "https://ndownloader.figshare.com/files/%d" % fid
    zpath = os.path.join(ZIPS, bird + ".zip")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        with open(zpath, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); got += len(chunk)
                if total:
                    sys.stdout.write("\r  %s  %5.1f / %5.1f MB" % (bird, got / 1e6, total / 1e6))
                    sys.stdout.flush()
    print()
    # extract into DATA (zip root is already 'BirdN/...')
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    os.remove(zpath)
    if not have(bird):
        raise RuntimeError("%s extracted but Annotation.xml missing" % bird)


def main():
    birds = sys.argv[1:] or list(FILE_ID)
    os.makedirs(DATA, exist_ok=True)
    for b in birds:
        if b not in FILE_ID:
            print("skip unknown", b); continue
        if have(b):
            print("[have]    ", b); continue
        print("[download]", b, "...")
        fetch(b)
        print("[ok]      ", b)
    # tidy empty zip dir
    if os.path.isdir(ZIPS) and not os.listdir(ZIPS):
        shutil.rmtree(ZIPS)
    print("[done] data in", DATA)


if __name__ == "__main__":
    main()
