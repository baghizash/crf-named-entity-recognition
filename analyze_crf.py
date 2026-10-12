# -*- coding: utf-8 -*-
"""Named Entity Recognition alamat memakai Conditional Random Field (CRF).

Langkah:
1. Muat 234 alamat AS dari data/list_of_real_usa_addresses.json.
2. Tokenisasi tiap field (address, city, state, zip) per kata, beri label
   field pada tiap token.
3. Ekstrak fitur sederhana per token: kata, is_digit, panjang, is_upper.
4. Latih CRF (lbfgs, c1=0.1, c2=0.1, 100 iterasi) pada 80 persen data,
   uji pada 20 persen.
5. Evaluasi akurasi flat dan metrik per label, lalu visualkan.

Jalankan: python analyze_crf.py
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "list_of_real_usa_addresses.json")
CHARTS = os.path.join(BASE, "charts")
FINDINGS = os.path.join(BASE, "findings.json")
os.makedirs(CHARTS, exist_ok=True)

FIELDS = ["address", "city", "state", "zip"]


def tokenize(entry):
    tokens, labels = [], []
    for field in FIELDS:
        parts = entry[field].split()
        tokens.extend(parts)
        labels.extend([field] * len(parts))
    return tokens, labels


def token_features(token):
    return {
        "kata": token,
        "is_digit": token.isdigit(),
        "panjang": len(token),
        "is_upper": token.isupper(),
    }


def main():
    with open(DATA, "r") as f:
        addresses = json.load(f)

    X_words, y = [], []
    for entry in addresses:
        toks, labs = tokenize(entry)
        X_words.append(toks)
        y.append(labs)

    X = [[token_features(t) for t in seq] for seq in X_words]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.1, c2=0.1,
        max_iterations=100, all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)
    acc = metrics.flat_accuracy_score(y_test, y_pred)

    # Distribusi label pada seluruh data
    label_counts = Counter(l for seq in y for l in seq)

    # Metrik per label pada data uji
    y_true_flat = [l for seq in y_test for l in seq]
    y_pred_flat = [l for seq in y_pred for l in seq]
    per_label = {}
    for lab in FIELDS:
        tp = sum(1 for t, p in zip(y_true_flat, y_pred_flat)
                 if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true_flat, y_pred_flat)
                 if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true_flat, y_pred_flat)
                 if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": tp + fn}

    # Chart 1: distribusi label
    plt.figure(figsize=(7, 4.5))
    labs = list(label_counts.keys())
    plt.bar(labs, [label_counts[l] for l in labs], color="#1D4ED8")
    plt.title("Distribusi label token pada seluruh data")
    plt.xlabel("Label field")
    plt.ylabel("Jumlah token")
    for x, v in zip(labs, [label_counts[l] for l in labs]):
        plt.text(x, v + 8, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "01_distribusi_label.png"), dpi=120)
    plt.close()

    # Chart 2: F1 per label
    plt.figure(figsize=(7, 4.5))
    f1s = [per_label[l]["f1"] for l in FIELDS]
    plt.bar(FIELDS, f1s, color="#16A34A")
    plt.title("Skor F1 per label pada data uji")
    plt.xlabel("Label field")
    plt.ylabel("F1")
    plt.ylim(0, 1.05)
    for x, v in zip(FIELDS, f1s):
        plt.text(x, v + 0.02, "{:.3f}".format(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "02_f1_per_label.png"), dpi=120)
    plt.close()

    # Chart 3: bobot transisi terbesar dan terkecil
    trans = [((a, b), w) for (a, b), w in crf.transition_features_.items()]
    trans.sort(key=lambda t: t[1])
    low = trans[:6]
    high = trans[-6:]
    sel = low + high
    plt.figure(figsize=(10, 5))
    names = ["{} ke {}".format(a, b) for (a, b), _ in sel]
    weights = [w for _, w in sel]
    colors = ["#DC2626" if w < 0 else "#1D4ED8" for w in weights]
    plt.barh(names, weights, color=colors)
    plt.title("Bobot transisi CRF terkuat")
    plt.xlabel("Bobot")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "03_bobot_transisi.png"), dpi=120)
    plt.close()

    # Chart 4: fitur state terkuat
    feats = [(k, v) for k, v in crf.state_features_.items()]
    feats.sort(key=lambda t: t[1])
    sel_f = feats[:5] + feats[-5:]
    plt.figure(figsize=(10, 5))
    fnames = ["{} : {}".format(attr, lab) for (attr, lab), _ in sel_f]
    fweights = [w for _, w in sel_f]
    fcolors = ["#DC2626" if w < 0 else "#16A34A" for w in fweights]
    plt.barh(fnames, fweights, color=fcolors)
    plt.title("Bobot fitur state CRF terkuat")
    plt.xlabel("Bobot")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "04_bobot_fitur.png"), dpi=120)
    plt.close()

    findings = {
        "jumlah_alamat": len(addresses),
        "total_token": sum(len(s) for s in y),
        "akurasi_flat_data_uji": round(float(acc), 6),
        "distribusi_label": dict(label_counts),
        "metrik_per_label": {
            lab: {k: round(float(v), 4) if isinstance(v, float) else v
                  for k, v in m.items()}
            for lab, m in per_label.items()
        },
        "konfigurasi_crf": "lbfgs, c1 0.1, c2 0.1, 100 iterasi, split 80 banding 20",
    }
    with open(FINDINGS, "w") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    print("akurasi:", round(float(acc), 6))
    print("distribusi:", dict(label_counts))


if __name__ == "__main__":
    main()
