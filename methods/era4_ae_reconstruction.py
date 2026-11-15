# -*- coding: utf-8 -*-
"""図6-A AEROBLADE（Ricker ら 2024, arXiv 2401.17879）の考え方。
論文：Stable Diffusion などは、画像を小さな情報（潜在）に縮めて戻すオートエンコーダを使う。
      生成画像は、そのオートエンコーダで縮めて戻しても、本物よりずれにくい。学習なしで判定できる。
このソース：オートエンコーダの代わりに「4x4の平均で縮めて、なめらかに戻す」処理を使う。
"""
import numpy as np
from common import down, up_bilinear


def score(x, f=4):
    err = np.abs(x - up_bilinear(down(x, f), f)).mean(axis=(1, 2))
    return -np.log(err + 1e-8)  # 縮めて戻してもずれない ＝ その部品で作られた画像らしい
