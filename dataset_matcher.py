"""
Content-based matching between dataset images (Roboflow export, 640x640 stretch)
and raw files (CR2 / JPG).

Why content-based: the Roboflow file names no longer identify the raw source
unambiguously (e.g. raw IMG_0463.JPG vs IMG_0463.CR2 share a stem but are
different photos; dataset 'IMG_0468_jpg' is actually the pixels of
'IMG_0468(1).CR2').  Pixel content is therefore used to decide which raw file
is the source of every dataset image, and file names are only used to break
ties between byte-identical / near-identical files.
"""
import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np

S = 32
FEATURE_CACHE = '_features.pkl'


def _arr(v):
    return np.frombuffer(v, dtype=np.uint8).astype(np.float32).reshape(S, S)


def _variants(a):
    out = []
    for f in (False, True):
        b = a[:, ::-1] if f else a
        for k in range(4):
            out.append(np.rot90(b, k))
    return out


def _zn(x):
    x = x.reshape(x.shape[0], -1) if x.ndim == 3 else x.reshape(1, -1)
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def _dataset_norm(fn):
    s = re.sub(r'\.rf\.[a-f0-9]+\.jpg$', '', fn, flags=re.I)
    s = re.sub(r'_(jpg|cr2|jpeg)$', '', s, flags=re.I)
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _raw_norm(fn):
    return re.sub(r'[^a-z0-9]', '', os.path.splitext(fn)[0].lower())


def _refine(ds_path, cand_paths, cand_idx, fine=96, tol=0.002):
    """Among candidate raw files, keep those whose 96x96 signature best matches the dataset image."""
    from content_features import raw_sig, ds_sig
    _, dv = ds_sig(ds_path, fine)
    D = _zn(np.stack(_variants(np.frombuffer(dv, dtype=np.uint8).astype(np.float32).reshape(fine, fine))))
    scores = []
    for p in cand_paths:
        _, rv = raw_sig(p, fine)
        r = _zn(np.frombuffer(rv, dtype=np.uint8).astype(np.float32).reshape(fine, fine))
        scores.append(float((r @ D.T).max()))
    top = max(scores)
    return [i for i, s in zip(cand_idx, scores) if s >= top - tol]


def load_features():
    if not os.path.exists(FEATURE_CACHE):
        subprocess.check_call([sys.executable, 'content_features.py'])
    F = pickle.load(open(FEATURE_CACHE, 'rb'))
    raw = {os.path.normpath(k): _arr(v) for k, v in F['raw'].items()}
    ds = {os.path.normpath(k): _arr(v) for k, v in F['ds'].items()}
    return raw, ds


def match_species(species_key, raw, ds, tol=0.003):
    """
    species_key: 'alvarezii' | 'striatum'
    Returns (assignment, report)
      assignment: {normalized raw path: split}
      report: dict with diagnostics
    """
    rkeys = [k for k in raw if k.startswith(f'Kappaphycus_{species_key}_raw')]
    R = _zn(np.stack([raw[k] for k in rkeys]))
    rcls = [k.split(os.sep)[1] for k in rkeys]
    rname = [_raw_norm(os.path.basename(k)) for k in rkeys]
    dkeys = sorted(k for k in ds if k.startswith(f'Kappaphycus_{species_key}_datasets'))

    items = []          # (dataset key, split, class, name, candidate raw indices, best sim)
    for k in dkeys:
        p = k.split(os.sep)
        split, cls, fn = p[1], p[2], p[3]
        V = _zn(np.stack(_variants(ds[k])))
        sim = (R @ V.T).max(axis=1)
        top = sim.max()
        cand = [i for i in np.where(sim >= top - tol)[0] if rcls[i].lower() == cls.lower()]
        if not cand:   # fall back: best candidate inside the dataset class
            same = [i for i in range(len(rkeys)) if rcls[i].lower() == cls.lower()]
            best = max(same, key=lambda i: sim[i])
            cand = [best]
        if len(cand) > 1:
            cand = _refine(k, [rkeys[i] for i in cand], cand)
        items.append((k, split, cls, _dataset_norm(fn), cand, float(top)))

    assignment = {}
    claimed = set()
    unresolved = []
    # pass 1: unique candidates, pass 2: name preference, pass 3: any free candidate
    order = sorted(items, key=lambda it: len(it[4]))
    for k, split, cls, name, cand, top in order:
        free = [i for i in cand if i not in claimed]
        pick = None
        if len(free) == 1:
            pick = free[0]
        elif len(free) > 1:
            byname = [i for i in free if rname[i] == name]
            pick = byname[0] if byname else sorted(free, key=lambda i: rkeys[i])[0]
        else:
            unresolved.append((k, [rkeys[i] for i in cand]))
            continue
        claimed.add(pick)
        assignment[os.path.normpath(rkeys[pick])] = (split, k)
    report = {'dataset_images': len(dkeys), 'assigned': len(assignment),
              'unresolved': unresolved,
              'low_similarity': [(it[0], it[5]) for it in items if it[5] < 0.97]}
    return assignment, report
