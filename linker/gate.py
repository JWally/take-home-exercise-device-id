"""Kruskal single-linkage with a learned cluster-merge gate.

Edges are visited in descending pair score. Each union attempt (A, B) is described by
cluster-level features (cross-edge evidence, attribute conflicts, temporal overlap,
version monotonicity) and accepted iff gate(features) >= thr.

Training events come from a fold-respecting oracle replay: labeled clusters stay pure
(labeled-vs-unlabeled merges are guaranteed negatives), unlabeled-only merges follow
s >= unl_thr. The gate for an event uses models that excluded the folds of its labeled
sessions (mean of all models when none / lockbox only).
"""

import numpy as np
import pandas as pd

ATTRS = [
    "gpu_renderer",
    "device_model",
    "timezone",
    "languages",
    "screen_width",
    "screen_height",
    "screen_pixel_ratio",
    "device_cores",
    "device_memory",
    "fonts",
    "emoji_hashes",
    "text_shaping_hash",
    "audio_fingerprint",
    "media_speech_voice_hash",
    "user_agent",
    "browser_major",
    "os_ver",
    "asn",
    "country",
    "ip24",
    "ip",
    "platform_version",
    "gpu_parameters_hash",
    "media_rtc_video_capabilities_hash",
    "math_fingerprint_hash",
    "viewport_width",
    "viewport_height",
    "privacy_incognito",
]


def session_table(df: pd.DataFrame) -> dict:
    d = df.copy()
    ua = d.user_agent.fillna("")
    d["browser_major"] = ua.str.extract(
        r"(?:Chrome|Firefox|Version|Edg|SamsungBrowser|OPR|CriOS|FxiOS)/(\d+)"
    )[0]
    d["os_ver"] = ua.str.extract(r"(?:OS |Android |Mac OS X |Windows NT )(\d+(?:[._]\d+)*)")[0]
    d["ip24"] = d.ip.str.rsplit(".", n=1).str[0]
    T = {}
    for c in ATTRS:
        s = d[c].astype(object).where(d[c].notna(), None)
        s = s.where(~s.isin(["unreadable", "", "-1", -1, -1.0, "[]", '""']), None)
        T[c] = pd.factorize(s, use_na_sentinel=True)[0]
    T["t"] = d.ts.astype("int64").to_numpy() / 1e6 / 3600.0
    T["bmaj"] = pd.to_numeric(d.browser_major, errors="coerce").to_numpy(float)
    osv = d.os_ver.str.replace("_", ".").str.split(".")
    T["osv"] = (
        pd.to_numeric(osv.str[0], errors="coerce").fillna(-1) * 1000
        + pd.to_numeric(osv.str[1], errors="coerce").fillna(0)
    ).to_numpy(float)
    T["bf"] = pd.factorize(d.browser_family)[0]
    return T


class Clusters:
    def __init__(self, n, T, a, b, s):
        self.T = T
        self.parent = np.arange(n)
        self.members = [[i] for i in range(n)]
        self.vals = [{c: {T[c][i]} - {-1} for c in ATTRS} for i in range(n)]
        nbr = [[] for _ in range(n)]
        for i, j, x in zip(a.tolist(), b.tolist(), s.tolist(), strict=False):
            nbr[i].append((j, x))
            nbr[j].append((i, x))
        self.nbr = nbr

    def find(self, x):
        p = self.parent
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, ra, rb):
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] += self.members[rb]
        self.members[rb] = []
        for c in ATTRS:
            self.vals[ra][c] |= self.vals[rb][c]
        self.vals[rb] = None
        return ra

    def features(self, ra, rb, s_edge, ei=None, ej=None):
        A, B = self.members[ra], self.members[rb]
        if len(A) > len(B):
            A, B, ra, rb = B, A, rb, ra
        nA, nB = len(A), len(B)
        cross = [x for i in A for (j, x) in self.nbr[i] if self.find(j) == rb]
        cross = np.array(cross) if cross else np.zeros(0)
        lg = np.log(np.clip(cross, 1e-4, 1 - 1e-4) / (1 - np.clip(cross, 1e-4, 1 - 1e-4)))
        f = dict(
            s=s_edge,
            nA=nA,
            nB=nB,
            n_tot=nA + nB,
            n_cross=len(cross),
            frac_scored=len(cross) / (nA * nB),
            n_ge5=(cross >= 0.5).sum(),
            n_ge2=(cross >= 0.2).sum(),
            n_lt05=(cross < 0.05).sum(),
            mean_cross=cross.mean() if len(cross) else 0.0,
            sum_logit=lg.sum(),
            mean_logit=lg.mean() if len(cross) else 0.0,
            second=np.sort(cross)[-2] if len(cross) > 1 else 0.0,
        )
        va, vb = self.vals[ra], self.vals[rb]
        for c in ATTRS:
            x, y = va[c], vb[c]
            if not x or not y:
                f["i_" + c], f["new_" + c] = -1, -1
            else:
                f["i_" + c] = int(bool(x & y))
                f["new_" + c] = len(x | y) - max(len(x), len(y))
        T = self.T
        ta, tb = T["t"][A], T["t"][B]
        f["gap"] = np.abs(ta[:, None] - tb[None, :]).min()
        f["overlap"] = min(ta.max(), tb.max()) - max(ta.min(), tb.min())
        f["span"] = max(ta.max(), tb.max()) - min(ta.min(), tb.min())
        # near-concurrent cross sessions on different /24
        dtm = np.abs(ta[:, None] - tb[None, :])
        ipd = T["ip24"][A][:, None] != T["ip24"][B][None, :]
        f["conc_diffip"] = int(((dtm < 1 / 6) & ipd).sum())
        f["conc_1h_diffip"] = int(((dtm < 1) & ipd).sum())
        # version downgrades introduced by merging
        idx = np.array(A + B)
        order = idx[np.argsort(T["t"][idx], kind="stable")]

        def downs(ix, key):
            v = T[key][ix]
            v = v[~np.isnan(v) & (v >= 0)]
            return int((np.diff(v) < 0).sum()) if len(v) > 1 else 0

        def own(key):
            return downs(np.array(A)[np.argsort(ta)], key) + downs(np.array(B)[np.argsort(tb)], key)

        f["down_b"] = downs(order, "bmaj") - own("bmaj")
        f["down_os"] = downs(order, "osv") - own("osv")
        f["bf"] = T["bf"][A[0]]
        if ei is not None:
            for nm, v in (("i", ei), ("j", ej)):
                out = [x for (k, x) in self.nbr[v] if self.find(k) not in (ra, rb)]
                f["rival_" + nm] = max(out) if out else 0.0
                f["n_rival_ge_" + nm] = sum(x >= s_edge for x in out)
            f["rival_max"] = max(f["rival_i"], f["rival_j"])
            # deg of endpoints overall
            f["deg_i"], f["deg_j"] = len(self.nbr[ei]), len(self.nbr[ej])
        return f


def replay(n, a, b, s, T, decide, floor=0.1, record=None):
    """Kruskal over edges s >= floor, descending. decide(ra, rb, feats, members) -> bool."""
    keep = s >= floor
    eid = np.flatnonzero(keep)
    a, b, s = a[keep], b[keep], s[keep]
    order = np.argsort(-s, kind="stable")
    C = Clusters(n, T, a, b, s)
    for e in order:
        ra, rb = C.find(a[e]), C.find(b[e])
        if ra == rb:
            continue
        f = C.features(ra, rb, float(s[e]), int(a[e]), int(b[e]))
        f["edge"] = int(eid[e])
        ok = decide(ra, rb, f, C)
        if record is not None:
            record.append((f, ra, rb, ok))
        if ok:
            C.union(ra, rb)
    return np.array([C.find(i) for i in range(n)])
