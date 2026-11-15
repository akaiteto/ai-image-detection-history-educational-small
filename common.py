# -*- coding: utf-8 -*-
"""共通の小道具（numpy だけ）。
説明用の小さな画像（32x32、白黒）で、時代ごとの検出の考え方を比べるための部品。
論文は 256x256 などのカラー写真を使う。ここでは仕組みが見えるように小さくしている。
"""
import numpy as np

SIZE = 32  # 説明用の例（論文は 256x256 など）


def blur(x, k=1):
    """まわり (2k+1)x(2k+1) マスの平均でならす（細かい模様を消す）。"""
    p = np.pad(x, ((0, 0), (k, k), (k, k)), mode="reflect")
    out = np.zeros_like(x)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            out += p[:, dy:dy + x.shape[1], dx:dx + x.shape[2]]
    return out / (2 * k + 1) ** 2


def down(x, f):
    """f x f マスごとの平均で縮める（オートエンコーダの「圧縮」の役）。"""
    n, h, w = x.shape
    return x.reshape(n, h // f, f, w // f, f).mean(axis=(2, 4))


def up_bilinear(x, f):
    """なめらかに f 倍へ広げる（オートエンコーダの「復元」の役）。"""
    n, h, w = x.shape
    ys = (np.arange(h * f) + 0.5) / f - 0.5
    xs = (np.arange(w * f) + 0.5) / f - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0, 1)[None, :, None]; wx = np.clip(xs - x0, 0, 1)[None, None, :]
    a = x[:, y0][:, :, x0]; b = x[:, y0][:, :, x1]; c = x[:, y1][:, :, x0]; d = x[:, y1][:, :, x1]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def auc(score_real, score_fake):
    """AUC：偽物のほうが「偽物らしさの点数」が高くなる確率。0.5＝当てずっぽう、1.0＝完全に見抜ける。"""
    s = np.concatenate([score_real, score_fake])
    r = s.argsort().argsort().astype(float) + 1
    n1, n2 = len(score_real), len(score_fake)
    return (r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)
