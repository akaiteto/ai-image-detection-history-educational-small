# -*- coding: utf-8 -*-
"""図5-B UnivFD（Ojha ら 2023, arXiv 2302.10174）の考え方。
論文：本物/偽物を見分ける学習をしていない大きなモデル（CLIP）の特徴を使い、近い例を探す（最近傍）だけで判定。
      偽物だけを覚えた分類器より、見たことのない生成AIに強い。
このソース：CLIP の重みは使わない（ダウンロードできないため）。代わりに、乱数で作って固定した小さな窓の束を
          「本物/偽物の学習をしていない特徴」として使い、最近傍で判定する。
"""
import numpy as np


class Detector:
    def __init__(self, n_filters=48, seed=123):
        rng = np.random.default_rng(seed)
        self.k = rng.normal(size=(n_filters, 3, 3))  # 固定して学習しない

    def features(self, x):
        n, h, w = x.shape
        p = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="reflect")
        patches = np.stack([p[:, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=-1)
        resp = patches @ self.k.reshape(len(self.k), 9).T  # 各窓の反応の地図
        f = np.concatenate([np.abs(resp).mean(axis=(1, 2)), resp.std(axis=(1, 2))], axis=1)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

    def fit(self, real, fake):
        self.bank = np.concatenate([self.features(real), self.features(fake)])
        self.lab = np.r_[np.zeros(len(real)), np.ones(len(fake))]
        return self

    def score(self, x, k=5):
        sim = self.features(x) @ self.bank.T
        idx = np.argsort(-sim, axis=1)[:, :k]
        return self.lab[idx].mean(axis=1)  # 近い例のうち、偽物の割合
