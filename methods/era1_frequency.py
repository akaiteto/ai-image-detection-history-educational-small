# -*- coding: utf-8 -*-
"""図3-B 周波数のクセ（Dzanic ら 2019, arXiv 1911.06465）の考え方。
論文：生成画像は、細かい模様（高い周波数）の量の分布が本物とずれる。
このソース：2マス周期（いちばん細かい周期）の強さを、まわりの周波数と比べるだけ（学習なし）。
"""
import numpy as np


def score(x):
    # 図3-B 画像を「どの細かさの模様がどれだけあるか」に分ける（フーリエ変換）
    mag = np.abs(np.fft.fft2(x - x.mean(axis=(1, 2), keepdims=True)))
    h = x.shape[1] // 2
    peak = mag[:, h, h] + mag[:, 0, h] + mag[:, h, 0]
    around = mag[:, h - 2:h + 3, h - 2:h + 3].mean(axis=(1, 2)) + 1e-8
    return np.log(peak / around + 1e-8)  # 規則的なクセがあるほど大きい
