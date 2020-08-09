---
layout: default
title: 数値計算
permalink: /numerical/
---
いろいろな話題について書く予定です。数学科に優しい記事を目指します。

---
## 数値計算の記事やpdf集
数値計算についての記事やpdfについてまとめる予定です。

[連立方程式の反復法（pdfリンク）](/blog_pdf/iterative_method/iterative_method.pdf)<br>
連立一次方程式の数値計算法です。Jacobi法、Gauss-Seidel法、SOR法について、狭義対角優位および既約優対角行列に対しての収束証明が書いてあります。

[共役勾配法（pdfリンク）](/blog_pdf/cg_method/cg_method.pdf)<br>
共役勾配法についての解説です。対称正定値行列に対する収束の証明が書いてあります。ただ証明がへたくそなのでそのうち書き直すかもしれません。

[SDEの数値解の強収束（pdfリンク）](/blog_pdf/SDEnumerical_strong_converge/SDEnumerical_strong_converge.pdf)<br>
SDEの数値計算において，数値解が真の解に強収束という意味で収束することの証明です。
内容の多くはMilsteinの本などに基づきます。執筆中。

[SPDEの数値計算（pdfリンク）](/blog_pdf/SPDE_numerical/SPDE_numerical.pdf)<br>
SPDEの数値計算についてのpdfです。SPDEの導入から書いています。
修士セミナーの板書ノートも兼ねています。執筆中。

---
## データ同化の記事集
データ同化(Data Assimilation)をJulia言語を用いて行ってみようという趣旨で書いてます。
一口にデータ同化と言ってもたくさんの手法があるので、それらのなかで主要なものを紹介していきたいと思います。

[データ同化の基礎]({% post_url 2020-04-17-introduction %})<br>
データ同化の基本概念について説明した記事です。

[Extended Kalman Filter]({% post_url 2020-04-18-ExtendedKF %})<br>
Extended Kalman Filter（拡張カルマンフィルター）についての解説記事です。

[Ensemble Kalman Filter 1]({% post_url 2020-05-03-EnKFSRF %})<br>
Ensemble Kalman Filter（アンサンブルカルマンフィルター）のうち、
最も基本的なPO法とSRFの解説です。Localizationについても書いています。

---
## プログラム置き場
趣味で書いた数値計算のプログラムと計算結果の置き場所です。
収束の議論も可能な限りはしたいです。

[SDEの数値計算]({% post_url 2020-08-09-SDENumerical %})<br>
確率微分方程式の数値計算のJulia言語による実装です。
結構適当に書いてます。収束の議論については
[SDEの数値解の強収束（pdfリンク）](/blog_pdf/SDEnumerical_strong_converge/SDEnumerical_strong_converge.pdf)
を参照してください。ただし、疑似乱数を取ることによる誤差は（いまのところ）扱っていません。