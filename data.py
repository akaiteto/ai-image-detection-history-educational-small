# -*- coding: utf-8 -*-
"""本物の写真と、3種類の生成画像の「代わり」を作る（説明用のおもちゃのデータ）。
本物の写真や生成AIはダウンロードせず、作り方の違いだけをまねた小さな画像を使う。
- real      : なめらかな景色＋カメラのセンサーの細かいノイズ（本物の写真の代わり）
- gan       : 小さく作ってから「転置畳み込み」ふうに2倍へ広げる → 2マスおきの規則的なクセが残る（図3-A・3-B）
- diffusion : ノイズから少しずつならしていく。最後に細かいノイズも描くが、となりの画素と少し似たノイズになる（図4-A）
- ldm       : 小さな「潜在」を作り、オートエンコーダの復元部品で広げる（図6-A）
"""
import numpy as np
from common import SIZE, blur, up_bilinear


def _field(rng, n, size, smooth=3):
    x = rng.normal(size=(n, size, size))
    for _ in range(smooth):
        x = blur(x, 1)
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
    return x * 0.2 + 0.5


def make_real(rng, n):
    return _field(rng, n, SIZE) + rng.normal(scale=0.03, size=(n, SIZE, SIZE))


def make_gan(rng, n):
    small = _field(rng, n, SIZE // 2, smooth=2)
    big = np.repeat(np.repeat(small, 2, axis=1), 2, axis=2)
    kernel = np.array([[1.04, 0.98], [0.98, 1.00]])  # 転置畳み込みの重みが均一でない → 2マス周期の模様
    return big * np.tile(kernel, (SIZE // 2, SIZE // 2))[None] + rng.normal(scale=0.004, size=big.shape)


def make_diffusion(rng, n, steps=12):
    x = rng.normal(size=(n, SIZE, SIZE))
    for t in range(steps):  # 1ステップずつ、ノイズを減らしてならす（拡散モデルの「逆向き」のまね）
        x = blur(x, 1) * 1.15
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
    noise = blur(rng.normal(size=x.shape), 1) * 0.5 + rng.normal(size=x.shape) * 0.5  # となりと少し似たノイズ
    noise = noise / noise.std() * 0.03  # 大きさは本物のセンサーのノイズと同じ
    return x * 0.2 + 0.5 + noise


def make_ldm(rng, n, f=4):
    latent = _field(rng, n, SIZE // f, smooth=1)
    return up_bilinear(latent, f) + rng.normal(scale=0.003, size=(n, SIZE, SIZE))


def make_all(seed=0, n=200):
    rng = np.random.default_rng(seed)
    return {"real": make_real(rng, n), "gan": make_gan(rng, n),
            "diffusion": make_diffusion(rng, n), "ldm": make_ldm(rng, n)}
