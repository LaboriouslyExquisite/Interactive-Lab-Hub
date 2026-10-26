"""
tram_train.py: teach a small image classifier "tram in view" vs "no tram", semi-supervised.

    python tram_train.py                     # 2 rounds (labels, then labels + pseudo-labels)
    python tram_train.py --rounds 1          # supervised only, to compare
    python tram_train.py --epochs 30         # train longer

How it learns (no color rules: the network learns what a tram looks like from the pictures):
  1. Your labels (tram_labels.csv) are split per class: 80% to train, 20% held back to test.
  2. Round 1: a YOLO classifier pretrained on ImageNet (yolo26n-cls) is fine-tuned on your
     training labels. Pretraining means it already knows edges, shapes and objects, so a few
     hundred labels go a long way.
  3. Pseudo-labeling (the semi-supervised part): the round-1 model looks at the unlabeled crops
     and keeps the ones it is very sure about (>= --conf) as extra training data.
  4. Round 2: fine-tune again on your labels + those pseudo-labels.
  5. Every score is measured on the held-back images YOU labeled, never on pseudo-labels,
     so the numbers are honest. The best model is copied to tram_model.pt.

Training on the Pi 5 CPU takes a while (roughly 10-30 min for a few thousand crops); a laptop
is faster: copy tram_data/ and tram_labels.csv over, run this there, copy tram_model.pt back.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from tram_common import DATA, MODEL
from tram_label import all_images, load_labels

DS = Path("tram_dataset")          # rebuilt every run: train/{tram,no_tram}, val/{tram,no_tram}
RUNS = Path("tram_runs")
CLASSES = {0: "no_tram", 1: "tram"}


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except OSError:                  # e.g. a filesystem without symlinks
        shutil.copy(src, dst)


def flat(name):
    return name.replace("/", "_")


def build_dataset(train, val, labels, pseudo=None):
    shutil.rmtree(DS, ignore_errors=True)
    for n in train:
        link(DATA / n, DS / "train" / CLASSES[labels[n]] / flat(n))
    for n, y in (pseudo or {}).items():
        link(DATA / n, DS / "train" / CLASSES[y] / ("pseudo_" + flat(n)))
    for n in val:
        link(DATA / n, DS / "val" / CLASSES[labels[n]] / flat(n))


def tram_probs(model, names, imgsz, batch=64):
    """P(tram) for each image name."""
    idx = [k for k, v in model.names.items() if v == "tram"][0]
    out = []
    for i in range(0, len(names), batch):
        chunk = [str(DATA / n) for n in names[i:i + batch]]
        for r in model.predict(chunk, imgsz=imgsz, verbose=False):
            out.append(float(r.probs.data[idx]))
    return np.array(out)


def score(model, val, labels, imgsz):
    p = tram_probs(model, val, imgsz)
    y = np.array([labels[n] for n in val])
    pred = p >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"accuracy": (tp + tn) / len(y), "recall": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp), "missed": fn, "false_alarms": fp}


def report(name, s):
    print(f"\n{name}, on the held-back images you labeled:")
    print(f"  accuracy  {s['accuracy']:.1%}")
    print(f"  recall    {s['recall']:.1%}   (of the trams really there, how many it caught)")
    print(f"  precision {s['precision']:.1%}   (of its 'tram!' calls, how many were right)")
    print(f"  missed trams: {s['missed']}   false alarms: {s['false_alarms']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2, help="1 = supervised only")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=160, help="image size the network sees")
    parser.add_argument("--base", default="yolo26n-cls.pt", help="pretrained starting weights")
    parser.add_argument("--conf", type=float, default=0.95, help="pseudo-label confidence needed")
    parser.add_argument("--pool", type=int, default=4000, help="unlabeled images to pseudo-label")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("needs ultralytics (it is in the Lab 5 requirements)")

    labels = load_labels()
    n_tram = sum(labels.values())
    if n_tram < 15 or len(labels) - n_tram < 15:
        raise SystemExit(f"need at least 15 of each: have {n_tram} tram, {len(labels) - n_tram} "
                         "no tram. Keep labeling (tram_label.py --find-trams helps)")

    rng = random.Random(0)                        # stratified split: 20% of each class held back
    train, val = [], []
    for cls in (0, 1):
        members = sorted(n for n, y in labels.items() if y == cls)
        rng.shuffle(members)
        k = max(3, len(members) // 5)
        val += members[:k]
        train += members[k:]
    print(f"{len(labels)} labeled ({n_tram} tram). training on {len(train)}, testing on {len(val)}",
          flush=True)

    weights, pseudo, best, history = args.base, {}, None, []
    for rnd in range(1, args.rounds + 1):
        build_dataset(train, val, labels, pseudo)
        print(f"\n=== round {rnd}: {len(train)} your labels + {len(pseudo)} pseudo-labels ===",
              flush=True)
        model = YOLO(weights)
        model.train(data=str(DS), epochs=args.epochs, imgsz=args.imgsz, batch=32,
                    device=args.device, workers=2, project=str(RUNS.resolve()),
                    name=f"round{rnd}", exist_ok=True, patience=6, verbose=False, plots=False)
        trained = RUNS / f"round{rnd}" / "weights" / "best.pt"
        model = YOLO(str(trained))
        s = score(model, val, labels, args.imgsz)
        report(f"round {rnd}", s)
        history.append({"round": rnd, "pseudo": len(pseudo), **s})
        if best is None or s["accuracy"] >= best[1]["accuracy"]:
            best = (trained, s)
        weights = str(trained)

        if rnd < args.rounds:                     # pseudo-label for the next round
            pool = [n for n in all_images() if n not in labels]
            pool = rng.sample(pool, min(len(pool), args.pool))
            p = tram_probs(model, pool, args.imgsz)
            sure_tram = [n for n, q in zip(pool, p) if q >= args.conf]
            sure_none = [n for n, q in zip(pool, p) if q <= 1 - args.conf]
            # keep the classes from getting lopsided: at most 3 empty skies per tram image
            cap = 3 * max(len(sure_tram), sum(labels[n] for n in train))
            sure_none = rng.sample(sure_none, min(len(sure_none), cap))
            pseudo = {**{n: 1 for n in sure_tram}, **{n: 0 for n in sure_none}}
            print(f"pseudo-labeled {len(sure_tram)} tram + {len(sure_none)} no-tram images "
                  f"(of {len(pool)} unlabeled; {len(pool) - len(sure_tram) - len(sure_none)} "
                  f"too unsure to use)", flush=True)

    shutil.copy(best[0], MODEL)
    with open(MODEL.with_suffix(".json"), "w") as f:
        json.dump({"imgsz": args.imgsz, "trained_on": len(train), "history": history}, f, indent=1)
    print(f"\nsaved {MODEL} (from {best[0].parent.parent.name}). Next: "
          "`python tram_label.py --uncertain`, then train again.")


if __name__ == "__main__":
    main()
