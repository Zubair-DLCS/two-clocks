"""
Text processing and account-level language representations.

- Twitter-style normalisation, matching the GloVe-Twitter preprocessing
  (<url>, <user>, <number>, <hashtag> tokens).
- SIF-weighted averages of pretrained GloVe-Twitter vectors (50-d, trained on
  2B tweets; Pennington et al. 2014). The vectors are downloaded from the
  gensim-data GitHub release and filtered to the corpus vocabulary.
- Plain TF-IDF documents (one document per account).
"""
import gzip, os, re, subprocess
from collections import Counter
import numpy as np
from common import DATA

GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-twitter-50/glove-twitter-50.gz")
GLOVE = os.path.join(DATA, "glove", "glove-twitter-50.gz")

URL = re.compile(r"https?://\S+|www\.\S+")
USER = re.compile(r"@\w+")
NUM = re.compile(r"(?<![\w#])[-+]?\d[\d,.:]*")
HASH = re.compile(r"#(\w+)")
TOK = re.compile(r"<\w+>|#?\w+(?:'\w+)?|[!?]+|[^\w\s]")


def clean(text):
    """Normalised text for TF-IDF: URLs/mentions collapsed, hashtags kept."""
    t = URL.sub(" <url> ", str(text).lower())
    t = USER.sub(" <user> ", t)
    return NUM.sub(" <number> ", t)


def glove_tokens(text):
    t = HASH.sub(r" <hashtag> \1 ", clean(text))
    return TOK.findall(t)


def load_glove(vocab):
    if not os.path.exists(GLOVE):
        os.makedirs(os.path.dirname(GLOVE), exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", GLOVE, GLOVE_URL], check=True)
    vec = {}
    with gzip.open(GLOVE, "rt", encoding="utf8", errors="ignore") as f:
        next(f)                                  # header: n_words dim
        for line in f:
            w, rest = line.split(" ", 1)
            if w in vocab:
                vec[w] = np.array(rest.split(), dtype=np.float32)
    return vec


class SIFEmbedder:
    """Smooth-inverse-frequency sentence embeddings (Arora et al., 2017)."""

    def __init__(self, token_lists, a=1e-3):
        counts = Counter(t for toks in token_lists for t in toks)
        total = sum(counts.values())
        self.vec = load_glove(set(counts))
        self.w = {t: a / (a + c / total) for t, c in counts.items() if t in self.vec}
        self.dim = 50
        self.coverage = sum(c for t, c in counts.items() if t in self.vec) / total

    def embed(self, toks):
        ws = [(self.w[t], self.vec[t]) for t in toks if t in self.w]
        if not ws:
            return np.zeros(self.dim, np.float32)
        wt = np.array([w for w, _ in ws])[:, None]
        return (wt * np.stack([v for _, v in ws])).sum(0) / wt.sum()
