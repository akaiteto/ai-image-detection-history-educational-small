# -*- coding: utf-8 -*-
"""図6-B ZED（Cozzolino ら 2024, arXiv 2409.15875）の考え方。
論文：本物の写真だけで「まわりの画素から、次の画素をどれくらい予想できるか」のモデルを作る。
      調べる画像が、そのモデルにとってどれだけ「意外か」を測り、本物とのずれで判定する。偽物のデータは使わない。
このソース：左と上の画素から次の画素を予想する直線のモデル（最小二乗）を、本物だけで作る。
"""
import numpy as np


def _ctx(x):
    c = np.stack([x[:, 1:, :-1], x[:, :-1, 1:], x[:, :-1, :-1]], axis=-1).reshape(len(x), -1, 3)
    t = x[:, 1:, 1:].reshape(len(x), -1)
    return c, t


class Detector:
    def fit(self, real):
        c, t = _ctx(real)
        A = np.concatenate([c.reshape(-1, 3), np.ones((c.shape[0] * c.shape[1], 1))], axis=1)
        self.w = np.linalg.lstsq(A, t.reshape(-1), rcond=None)[0]
        e = self._err(real)
        self.mu, self.sd = e.mean(), e.std() + 1e-8  # 本物の「意外さ」のふつうの範囲
        return self

    def _err(self, x):
        c, t = _ctx(x)
        pred = c @ self.w[:3] + self.w[3]
        return np.log(((t - pred) ** 2).mean(axis=1) + 1e-10)

    def score(self, x):
        return np.abs(self._err(x) - self.mu) / self.sd  # 本物のふつうからどれだけ外れるか
