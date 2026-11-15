# -*- coding: utf-8 -*-
"""図5-A DIRE（Wang ら 2023, arXiv 2303.09295）の考え方。
論文：学習済みの拡散モデルで画像を作り直す。拡散モデルが作った画像はほぼ作り直せるが、本物の写真は作り直せない。
      その差（再構成誤差＝DIRE）を、分類器に入れて判定する。
このソース：拡散モデルの代わりに「ならす処理を数回かける」作り直しで、誤差の大きさだけを点数にする。
"""
import numpy as np
from common import blur


def reconstruct(x, steps=2):
    y = x.copy()
    for _ in range(steps):
        y = blur(y, 1) * 0.5 + y * 0.5
    return y


def score(x):
    err = np.abs(x - reconstruct(x)).mean(axis=(1, 2))
    return -np.log(err + 1e-8)  # 誤差が小さい（作り直せる）ほど偽物らしい
