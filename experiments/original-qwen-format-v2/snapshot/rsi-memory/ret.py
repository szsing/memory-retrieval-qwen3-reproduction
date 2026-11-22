"""Small embedder (BGE-small, the submission's retriever) and cross-encoder, with a disk cache."""
import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = "BAAI/bge-small-en-v1.5"
QPRE = "Represent this sentence for searching relevant passages: "
_m = {}
_cache = {}
CDIR = os.path.join(HERE, "cache")


def _model():
    if "e" not in _m:
        import torch
        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer
        _m["e"] = SentenceTransformer(EMB, device="cpu")
    return _m["e"]


def _path():
    return os.path.join(CDIR, "emb.npz")


def _load():
    if not _cache and os.path.exists(_path()):
        z = np.load(_path(), allow_pickle=False)
        for k, v in zip(z["k"], z["v"]):
            _cache[str(k)] = v


def save():
    if _cache:
        os.makedirs(CDIR, exist_ok=True)
        ks = list(_cache)
        np.savez(_path(), k=np.array(ks), v=np.stack([_cache[k] for k in ks]))


def embed(texts, query=False):
    """Unit vectors; query=True adds BGE's query instruction."""
    _load()
    texts = [(QPRE + t) if query else t for t in texts]
    keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in _cache]
    if todo:
        V = _model().encode([texts[i] for i in todo], batch_size=64, normalize_embeddings=True)
        for i, v in zip(todo, V):
            _cache[keys[i]] = v.astype(np.float32)
        save()
    return np.stack([_cache[k] for k in keys])


def cross(pairs):
    if "c" not in _m:
        from sentence_transformers import CrossEncoder
        _m["c"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return np.asarray(_m["c"].predict(pairs, batch_size=64))


def topk(qv, M, k=6):
    s = M @ qv
    return list(np.argsort(-s)[:k])
