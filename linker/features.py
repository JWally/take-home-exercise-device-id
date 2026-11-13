"""Session normalization and vectorized pair features (no cookie / label / pool inputs)."""

import re
from typing import Any

import numpy as np
import pandas as pd

CAT = [
    "country",
    "asn",
    "gpu_vendor",
    "gpu_renderer",
    "gpu_parameters_hash",
    "device_model",
    "user_agent",
    "platform_version",
    "languages",
    "timezone",
    "emoji_hashes",
    "text_shaping_hash",
    "math_fingerprint_hash",
    "audio_fingerprint",
    "fonts",
    "media_speech_voice_hash",
    "media_rtc_video_capabilities_hash",
    "ip",
    "ip24",
    "ip16",
    "ua_noversion",
    "browser_major",
    "hw",
    "fp_soft",
    "browser_ver",
    "os_ver",
    "ua_app",
    "k_core",
    "k_mid",
    "k_full",
]
NUM = [
    "device_cores",
    "device_memory",
    "screen_width",
    "screen_height",
    "screen_pixel_ratio",
    "screen_color_depth",
    "viewport_width",
    "viewport_height",
    "touch_max_points",
    "privacy_incognito",
]
HW = [
    "device_model",
    "browser_family",
    "screen_width",
    "screen_height",
    "screen_pixel_ratio",
    "gpu_renderer",
    "device_cores",
    "device_memory",
    "touch_max_points",
]
# Whole-fingerprint keys at three granularities: a rare full match is strong evidence even when the
# network changes (mobile wifi <-> cellular is the dominant confident-miss pattern).
K_CORE = HW + [
    "fonts",
    "emoji_hashes",
    "text_shaping_hash",
    "math_fingerprint_hash",
    "gpu_parameters_hash",
    "media_rtc_video_capabilities_hash",
    "timezone",
    "languages",
    "screen_color_depth",
]
K_MID = K_CORE + ["ua_noversion", "audio_fingerprint", "privacy_incognito"]
K_FULL = K_MID + [
    "user_agent",
    "platform_version",
    "media_speech_voice_hash",
    "viewport_width",
    "viewport_height",
]


def _codes(s: pd.Series) -> np.ndarray:
    """Integer codes with -1 for missing/unavailable."""
    s = s.astype(object).where(s.notna(), None)
    s = s.where(~s.isin(["unreadable", "", "-1", -1, -1.0, "[]", '""']), None)
    c, _ = pd.factorize(s, use_na_sentinel=True)
    return c


def prepare(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ip24"] = df.ip.str.rsplit(".", n=1).str[0]
    df["ip16"] = df.ip.str.split(".").str[:2].str.join(".")
    ua = df.user_agent.fillna("")
    df["ua_noversion"] = ua.str.replace(r"\d+(?:[._]\d+)*", "#", regex=True)
    df["browser_major"] = ua.str.extract(
        r"(?:Chrome|Firefox|Version|Edg|SamsungBrowser|OPR|CriOS|FxiOS)/(\d+)"
    )[0]
    df["os_ver"] = ua.str.extract(r"(?:OS |Android |Mac OS X |Windows NT )(\d+(?:[._]\d+)*)")[0]
    df["browser_ver"] = ua.str.extract(
        r"(?:Chrome|Firefox|Version|Edg|SamsungBrowser|OPR|CriOS|FxiOS|EdgiOS|GSA)/([\d.]+)"
    )[0]
    df["ua_app"] = ua.str.extract(
        r"((?:CriOS|FxiOS|EdgiOS|GSA|Ddg|Brave|Ecosia|OPT|FBAN|FB_IAB|Instagram"
        r"|SamsungBrowser|OPR|Edg|YaBrowser|Mint|wv)\b)"
    )[0].fillna("none")
    df["hw"] = df.groupby(HW, dropna=False, sort=False).ngroup()
    df["fp_soft"] = df.groupby(
        [
            "fonts",
            "emoji_hashes",
            "text_shaping_hash",
            "math_fingerprint_hash",
            "media_rtc_video_capabilities_hash",
        ],
        dropna=False,
        sort=False,
    ).ngroup()
    for name, cols in (("k_core", K_CORE), ("k_mid", K_MID), ("k_full", K_FULL)):
        df[name] = df.groupby(cols, dropna=False, sort=False).ngroup()
    P: dict[str, Any] = {"n": len(df)}
    for name in ("k_core", "k_mid", "hw"):
        g = df.groupby(name)
        P[name + "_nasn"] = g.asn.transform("nunique").to_numpy(np.float32)
        P[name + "_nip24"] = g.ip24.transform("nunique").to_numpy(np.float32)
    for c in CAT:
        code = _codes(df[c])
        P[c] = code
        cnt = (
            np.bincount(code[code >= 0], minlength=code.max() + 1)
            if (code >= 0).any()
            else np.zeros(1)
        )
        P[c + "_freq"] = np.where(code >= 0, cnt[np.maximum(code, 0)], 0).astype(np.float32)
    for c in NUM:
        v = df[c].to_numpy(float)
        P[c] = np.where(v < 0, np.nan, v)
    P["battery_level"] = np.where(df.battery_level < 0, np.nan, df.battery_level).astype(float)
    P["battery_charging"] = np.where(df.battery_charging < 0, np.nan, df.battery_charging).astype(
        float
    )
    P["ts"] = df.ts.astype("int64").to_numpy() / 1e6 / 3600.0  # hours
    P["hour"] = df.ts.dt.hour.to_numpy() + df.ts.dt.minute.to_numpy() / 60
    P["browser_family"] = _codes(df.browser_family)
    P["browser_major_num"] = pd.to_numeric(df.browser_major, errors="coerce").to_numpy(float)
    P["os_major"] = pd.to_numeric(df.os_ver.str.split(r"[._]").str[0], errors="coerce").to_numpy(
        float
    )
    P["os_minor"] = pd.to_numeric(df.os_ver.str.split(r"[._]").str[1], errors="coerce").to_numpy(
        float
    )
    P["os_patch"] = pd.to_numeric(df.os_ver.str.split(r"[._]").str[2], errors="coerce").to_numpy(
        float
    )
    bv = df.browser_ver.str.split(".")
    P["browser_build"] = pd.to_numeric(bv.str[2], errors="coerce").to_numpy(float)
    P["browser_patch"] = pd.to_numeric(bv.str[3], errors="coerce").to_numpy(float)
    ipo = df.ip.str.split(".", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    P["ipint"] = ipo
    P["fontset"] = [
        frozenset(x.strip('"').split(",")) if isinstance(x, str) else frozenset() for x in df.fonts
    ]
    P["langlist"] = [
        tuple(x.strip('"').split(",")) if isinstance(x, str) else () for x in df.languages
    ]
    P["emoji"] = np.array(
        [
            [int(t) for t in re.findall(r"\d+", x)][:6] + [0] * (6 - len(re.findall(r"\d+", x)[:6]))
            if isinstance(x, str)
            else [0] * 6
            for x in df.emoji_hashes
        ],
        dtype=np.int64,
    )
    return P


def pair_features(P: dict, a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    F: dict[str, Any] = {}
    for c in CAT:
        x, y = P[c][a], P[c][b]
        miss = (x < 0) | (y < 0)
        eq = (x == y) & ~miss
        F["eq_" + c] = np.where(miss, -1, eq.astype(np.int8)).astype(np.int8)
        # rarity of the shared value (log count), or of the rarer of the two
        fa, fb = P[c + "_freq"][a], P[c + "_freq"][b]
        F["lf_" + c] = np.where(
            miss, np.nan, np.log1p(np.where(eq, fa, np.minimum(fa, fb)))
        ).astype(np.float32)
    for c in NUM:
        x, y = P[c][a], P[c][b]
        F["d_" + c] = (
            (x - y).astype(np.float32)
            if c in ("viewport_width", "viewport_height")
            else np.abs(x - y).astype(np.float32)
        )
        F["v_" + c] = np.fmin(x, y).astype(np.float32)
    F["browser_family"] = P["browser_family"][a].astype(np.int16)
    F["same_browser"] = (P["browser_family"][a] == P["browser_family"][b]).astype(np.int8)
    ta, tb = P["ts"][a], P["ts"][b]
    first_is_a = ta <= tb
    dt = np.abs(ta - tb)
    F["dt_h"] = dt.astype(np.float32)
    F["log_dt"] = np.log1p(dt * 60).astype(np.float32)
    F["hour_diff"] = np.minimum(
        np.abs(P["hour"][a] - P["hour"][b]), 24 - np.abs(P["hour"][a] - P["hour"][b])
    ).astype(np.float32)
    # battery dynamics: signed change from earlier to later session, rate per hour
    la, lb = P["battery_level"][a], P["battery_level"][b]
    first_l = np.where(first_is_a, la, lb)
    later_l = np.where(first_is_a, lb, la)
    first_c = np.where(first_is_a, P["battery_charging"][a], P["battery_charging"][b])
    later_c = np.where(first_is_a, P["battery_charging"][b], P["battery_charging"][a])
    dl = later_l - first_l
    F["bat_dl"] = dl.astype(np.float32)
    F["bat_rate"] = (dl / np.maximum(dt, 1 / 60)).astype(np.float32)
    F["bat_first"] = first_l.astype(np.float32)
    F["bat_first_c"] = first_c.astype(np.float32)
    F["bat_later_c"] = later_c.astype(np.float32)
    # versions: signed change earlier -> later (downgrades are suspicious)
    for c in (
        "browser_major_num",
        "os_major",
        "os_minor",
        "os_patch",
        "browser_build",
        "browser_patch",
    ):
        x, y = P[c][a], P[c][b]
        F["dv_" + c] = np.where(first_is_a, y - x, x - y).astype(np.float32)
    # viewport/screen geometry: rotation, browser chrome (screen - viewport) deltas
    vwa, vha, vwb, vhb = (
        P["viewport_width"][a],
        P["viewport_height"][a],
        P["viewport_width"][b],
        P["viewport_height"][b],
    )
    F["vp_rotated"] = (
        (np.abs(vwa - vhb) <= 2) & (np.abs(vha - vwb) <= 2) & (np.abs(vwa - vwb) > 2)
    ).astype(np.int8)
    swa, sha, swb, shb = (
        P["screen_width"][a],
        P["screen_height"][a],
        P["screen_width"][b],
        P["screen_height"][b],
    )
    F["scr_rotated"] = ((swa == shb) & (sha == swb) & (swa != swb)).astype(np.int8)
    F["d_chrome_h"] = np.abs(
        (np.fmax(sha, swa) - np.fmax(vha, vwa)) - (np.fmax(shb, swb) - np.fmax(vhb, vwb))
    ).astype(np.float32)
    F["d_vp_area"] = (np.abs(vwa * vha - vwb * vhb) / np.fmax(vwa * vha, 1)).astype(np.float32)
    # battery plausibility: level rose without charging at either end / implied drain speed
    rose = dl > 0.005
    F["bat_rose_uncharged"] = np.where(
        np.isnan(dl), -1, (rose & (first_c == 0) & (later_c == 0)).astype(np.int8)
    ).astype(np.int8)
    F["bat_fell_charging"] = np.where(
        np.isnan(dl), -1, ((dl < -0.005) & (first_c == 1) & (later_c == 1)).astype(np.int8)
    ).astype(np.int8)
    F["bat_abs_rate"] = (np.abs(dl) / np.maximum(dt, 1 / 60)).astype(np.float32)
    F["tz_country_both_eq"] = ((F["eq_timezone"] == 1) & (F["eq_country"] == 1)).astype(np.int8)
    F["v_os_major"] = np.fmin(P["os_major"][a], P["os_major"][b]).astype(np.float32)
    ia, ib = P["ipint"][a], P["ipint"][b]
    same = np.cumprod(ia == ib, axis=1).sum(axis=1)
    F["ip_prefix_octets"] = np.where(np.isnan(ia).any(1) | np.isnan(ib).any(1), -1, same).astype(
        np.int8
    )
    F["ip_o3_diff"] = np.abs(ia[:, 2] - ib[:, 2]).astype(np.float32)
    fs = P["fontset"]
    inter = np.fromiter(
        (len(fs[i] & fs[j]) for i, j in zip(a, b, strict=False)), np.float32, len(a)
    )
    uni = np.fromiter((len(fs[i] | fs[j]) for i, j in zip(a, b, strict=False)), np.float32, len(a))
    F["font_jac"] = np.where(uni > 0, inter / np.maximum(uni, 1), np.nan).astype(np.float32)
    F["font_n"] = uni
    ll = P["langlist"]
    F["lang_first_eq"] = np.fromiter(
        ((ll[i][:1] == ll[j][:1]) for i, j in zip(a, b, strict=False)), np.int8, len(a)
    )
    F["lang_overlap"] = np.fromiter(
        (
            len(set(ll[i]) & set(ll[j])) / max(len(set(ll[i]) | set(ll[j])), 1)
            for i, j in zip(a, b, strict=False)
        ),
        np.float32,
        len(a),
    )
    F["emoji_eq_n"] = (P["emoji"][a] == P["emoji"][b]).sum(1).astype(np.int8)
    for name in ("k_core", "k_mid", "hw"):
        F[name + "_nasn"] = np.where(F["eq_" + name] == 1, P[name + "_nasn"][a], -1).astype(
            np.float32
        )
        F[name + "_nip24"] = np.where(F["eq_" + name] == 1, P[name + "_nip24"][a], -1).astype(
            np.float32
        )
    F["hw_freq"] = np.log1p(P["hw_freq"][a]).astype(np.float32)
    F["n_cat_eq"] = sum((F["eq_" + c] == 1).astype(np.int16) for c in CAT)
    F["n_cat_ne"] = sum((F["eq_" + c] == 0).astype(np.int16) for c in CAT)
    return pd.DataFrame(F)
