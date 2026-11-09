"""Corpus prep for stage 2: Pride and Prejudice (Project Gutenberg #1342).

Gutenberg header/footer removed, lower-cased, all whitespace runs collapsed to
a single space, characters outside the kept set mapped to '#'.
Split: first 90% = train (used for all counts and for Adam), last 10% = test.
"""
import re
import numpy as np

KEEP = "abcdefghijklmnopqrstuvwxyz .,;:!?'\"-()_#"
BOS_CH = "^"
CHARS = KEEP + BOS_CH          # BOS never appears in text
STOI = {c: i for i, c in enumerate(CHARS)}
V = len(CHARS)
BOS = STOI[BOS_CH]


def load(path="data/pg1342.txt"):
    raw = open(path, encoding="utf-8").read()
    a = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    a = raw.index("\n", a) + 1
    b = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    t = raw[a:b].lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("—", "-").replace("–", "-")
    t = re.sub(r"\s+", " ", t).strip()
    t = "".join(c if c in STOI and c != BOS_CH else "#" for c in t)
    ids = np.array([STOI[c] for c in t], dtype=np.int64)
    cut = int(len(ids) * 0.9)
    return ids[:cut], ids[cut:]


def decode(ids):
    return "".join(CHARS[i] for i in ids)


if __name__ == "__main__":
    tr, te = load()
    print("V", V, "train", len(tr), "test", len(te))
    print(repr(decode(tr[:300])))
    print(repr(decode(te[:300])))
    print("counts", {CHARS[i]: int(c) for i, c in enumerate(np.bincount(tr, minlength=V))})
