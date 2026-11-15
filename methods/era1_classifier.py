# -*- coding: utf-8 -*-
"""図3-A CNNDetection（Wang ら 2019, arXiv 1912.11035）の考え方。
論文：ProGAN 1種類の本物/偽物で、ふつうの画像分類器（ResNet-50）を学習 → ほかのGANも見抜けた。
このソース：ResNet の代わりに、細かい模様（画像 − ならした画像）の特徴8個＋ロジスティック回帰（線形の分類器）。
"""
import numpy as np
from common import blur


def features(x):
    # 図3-A 細かい模様（残差）を取り出す。生成AIのクセは、この細かい所に出やすい
    r = x - blur(x, 1)
    f = np.fft.fft2(r)
    mag = np.abs(f)
    h = x.shape[1] // 2
    return np.stack([
        r.std(axis=(1, 2)),
        np.abs(r).mean(axis=(1, 2)),
        (r[:, :, 1:] * r[:, :, :-1]).mean(axis=(1, 2)) / (r.var(axis=(1, 2)) + 1e-8),  # となりの画素との似かた
        (r[:, 1:] * r[:, :-1]).mean(axis=(1, 2)) / (r.var(axis=(1, 2)) + 1e-8),
        np.log(mag[:, h, h] + 1e-6),  # 2マス周期の強さ（ナイキスト）
        np.log(mag[:, 0, h] + 1e-6),
        np.log(mag[:, h, 0] + 1e-6),
        np.log(mag.mean(axis=(1, 2)) + 1e-6),
    ], axis=1)


class Detector:
    """本物＝0、偽物＝1 として、特徴の重みを勾配で少しずつ直して学習する。"""
    def fit(self, real, fake, steps=800, lr=0.1):
        X = np.concatenate([features(real), features(fake)])
        y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        X = (X - self.mu) / self.sd
        self.w = np.zeros(X.shape[1]); self.b = 0.0
        for _ in range(steps):  # 損失（当たらなさ）が減る向きへ、重みを動かす
            p = 1 / (1 + np.exp(-(X @ self.w + self.b)))
            self.w -= lr * X.T @ (p - y) / len(y); self.b -= lr * (p - y).mean()
        return self

    def score(self, x):  # 大きいほど「偽物らしい」
        return ((features(x) - self.mu) / self.sd) @ self.w + self.b
