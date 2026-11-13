"""Candidate pair generation via multiple blocking keys, with large blocks sub-keyed."""

import numpy as np
import pandas as pd

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
KEY = 1 << 32  # pair code = a * KEY + b (a < b); same sort order as any multiplier > n


def decode(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (c // KEY).astype(np.int64), (c % KEY).astype(np.int64)


def _pairs_from_groups(codes: np.ndarray, cap: int) -> np.ndarray:
    out = []
    order = np.argsort(codes, kind="stable")
    c = codes[order]
    bounds = np.flatnonzero(np.diff(c)) + 1
    for g in np.split(order, bounds):
        if len(g) < 2 or len(g) > cap or codes[g[0]] < 0:
            continue
        i, j = np.triu_indices(len(g), 1)
        a, b = g[i].astype(np.int64), g[j].astype(np.int64)
        out.append(np.minimum(a, b) * KEY + np.maximum(a, b))
    return np.concatenate(out) if out else np.empty(0, np.int64)


def schemes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = df.copy()
    df["ip24"] = df.ip.str.rsplit(".", n=1).str[0]
    return df, {
        "hw_ip24": HW + ["ip24"],
        "hw_asn_tz": HW + ["asn", "timezone"],
        "hw_audio": HW + ["audio_fingerprint"],
        "hw_voice": HW + ["media_speech_voice_hash"],
        "hw_lang_tz_fonts": HW + ["languages", "timezone", "fonts", "emoji_hashes"],
        "hw_text_lang": HW + ["text_shaping_hash", "languages", "timezone"],
        "ip": ["ip"],
        "audio_text": ["audio_fingerprint", "text_shaping_hash", "browser_family"],
        "model_ua_lang_tz": ["device_model", "user_agent", "languages", "timezone"],
        "hw_battery_tz": HW + ["battery_level", "timezone"],
    }


# Device-model-free schemes: device_model drifts in ~27% of the true pairs the HW schemes miss.
EXTRA = {
    "nomodel_fp": (
        [
            "browser_family",
            "screen_width",
            "screen_height",
            "screen_pixel_ratio",
            "gpu_renderer",
            "fonts",
            "emoji_hashes",
            "timezone",
            "languages",
        ],
        400,
    ),
    "nomodel_audio": (["browser_family", "audio_fingerprint", "timezone", "gpu_renderer"], 200),
    "ip24_br_tz": (["ip24", "browser_family", "timezone"], 400),
}


def candidates(df: pd.DataFrame, cap: int = 400, extra: bool = True) -> np.ndarray:
    df, sch = schemes(df)
    plan = [(cols, cap) for cols in sch.values()]
    if extra:
        plan += list(EXTRA.values())
    keys = []
    for cols, c in plan:
        codes = df.groupby(cols, dropna=False, sort=False).ngroup().to_numpy()
        # treat rows with missing ip etc as fine; groupby dropna=False keeps them
        keys.append(_pairs_from_groups(codes, c))
    return np.unique(np.concatenate(keys))
