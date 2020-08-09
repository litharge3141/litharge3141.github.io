---
title: SDEの数値計算
---


<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$','$'],["\\\\(","\\\\)"]],
 displayMath: [ ['$$','$$'], ["\\\\[","\\\\]"] ]
 }
 });
</script>

Julia言語による確率微分方程式の数値計算について書きます。
特に断らない限り一意的な強解をもつ伊藤の確率微分方程式を考えます。

# Euler-Maruyamaスキーム

確率微分方程式

\begin{equation}
    dX(t) = f(t,X(t))dt + g(t,X(t))dB(t)
\end{equation}

を数値計算します。

## 陽的Euler-Maruyamaスキーム
実装が簡単なのでよく使われます。
ステップ数を$k$、時間の刻み幅を$h$とし、$t_{k} = kh$および$X_{k}$を数値解として

\begin{equation}
    X_{k+1} = f(t_{k},X_{k})h + g(t_{k},X_{k}) (B(t_{k+1}) - B(t_{k}))
\end{equation}

として近似します。$(B(t_{k+1}) - B(t_{k}))$は平均$0$分散$h$の正規分布に従うので、
疑似乱数をとって計算できます。

### 計算のコード
Black-Scholesの株価変動モデル
\begin{align}
    dX(t) = \alpha X(t) dB(t) + \beta X(t) dt
\end{align}
を初期条件$X(0)=1$のもとで解き、厳密解と数値解とBrown運動のサンプルパスをグラフにします。
パラメータは$\alpha=0.2,\beta=-0.5$としています（本当は両方正に取るべきですが）。

```
using LinearAlgebra
using DelimitedFiles
using Distributions
using Plots

#構造体の宣言
mutable struct DescretizeParameter
    Time::Float64
    IterationStep::Int
end

#moduleの宣言
module BlackScholes
    struct Parameter
        volatility::Float64
        drift::Float64
    end

    function DriftTerm(x::Float64,Parameter::BlackScholes.Parameter)
        return Parameter.drift*x
    end

    function StochasticTerm(x::Float64,Parameter::BlackScholes.Parameter)
        return Parameter.volatility*x
    end
end

using .BlackScholes


function EulerMaruyamaPlot()

    #方程式の刻み幅など
    DP = DescretizeParameter(1,500)
    dt = DP.Time / DP.IterationStep

    #Black-Scholesのパラメータ
    BSP = BlackScholes.Parameter(0.2,-0.5)

    plotT = 0:dt:DP.Time
    X = fill(0.0, DP.IterationStep+1)
    Xtrue = fill(0.0,DP.IterationStep+1)
    BrownMotion = fill(0.0,DP.IterationStep+1)
    randomness = Normal(0.0, sqrt(dt))
    rand_increment = rand(randomness, DP.IterationStep+1)
    X[1] = 1.0
    Xtrue[1] = 1.0
    BrownMotion[1] = 0

    for it in 1:DP.IterationStep
        X[it+1] = X[it] + (BlackScholes.DriftTerm(X[it],BSP)dt
            + BlackScholes.StochasticTerm(X[it],BSP)*rand_increment[it])

        Xtrue[it+1] = exp((BSP.drift - (BSP.volatility^2 /2.0))*(it+1)*dt
            + BSP.volatility*BrownMotion[it]*(it+1)*dt)

        BrownMotion[it+1] = BrownMotion[it] + rand_increment[it]
    end

    plot(plotT,X,label = "numerical solution")
    plot!(plotT,Xtrue,label = "strict solution")
    plot!(plotT,BrownMotion,label="BrownMotion")
end

EulerMaruyamaPlot()

```

パラメータはいろいろ変えて遊ぶといいと思います（DPやBSPを変えてください）。
数値解が厳密解にどれだけ近いかはBrown運動のご機嫌次第になります。
