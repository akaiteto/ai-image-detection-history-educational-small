# -*- coding: utf-8 -*-
"""時代ごとの検出の考え方を、同じデータで並べて比べる。
数字は AUC（0.5＝当てずっぽう、1.0＝完全に見抜ける）。説明用のおもちゃのデータなので、論文の数字とは比べられない。
見どころ：
- 2019型（GANで学習した分類器）は GAN には強いが、学習で見ていない拡散・LDM で崩れる（図4-B）
- 作り直し（DIRE型）・オートエンコーダ（AEROBLADE型）は、その作り方の生成画像にだけ強く効く（図5-A・6-A）
- 偽物を覚えない（UnivFD型）・本物だけで測る（ZED型）は、見ていない相手にも比較的くずれにくい（図5-B・6-B）
"""
import numpy as np
from common import auc
from data import make_all
from methods import era1_classifier, era1_frequency, era3_reconstruction, era3_feature_nn, era4_ae_reconstruction, era4_zeroshot


def main():
    train = make_all(seed=0, n=200)  # 学習用
    test = make_all(seed=1, n=200)   # 試験用（学習で見ていない画像）

    clf = era1_classifier.Detector().fit(train["real"], train["gan"])          # GANだけで学習
    nn = era3_feature_nn.Detector().fit(train["real"][:20], train["gan"][:20])  # 少しの例だけ覚える
    zed = era4_zeroshot.Detector().fit(train["real"])                          # 本物だけ

    methods = {
        "2019 分類器（GANで学習）": clf.score,
        "2019 周波数のクセ": era1_frequency.score,
        "2023 作り直し（DIRE型）": era3_reconstruction.score,
        "2023 固定特徴＋最近傍（UnivFD型）": nn.score,
        "2024 AEで作り直し（AEROBLADE型）": era4_ae_reconstruction.score,
        "2024 本物だけで意外さ（ZED型）": zed.score,
    }
    fakes = ["gan", "diffusion", "ldm"]
    print(f"{'方法':<34}" + "".join(f"{f:>11}" for f in fakes))
    for name, fn in methods.items():
        r = fn(test["real"])
        print(f"{name:<30}" + "".join(f"{auc(r, fn(test[f])):>11.2f}" for f in fakes))


if __name__ == "__main__":
    main()
