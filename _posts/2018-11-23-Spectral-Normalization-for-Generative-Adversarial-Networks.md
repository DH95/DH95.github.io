---
title: Spectral Normalization for Generative Adversarial Networks
date : 2018-11-23
description : paper
---

> [Spectral Normalization for Generative Adversarial Networks] (https://arxiv.org/abs/1802.05957)

ICLR 2018에 나온 논문. GAN에서 Discriminator 를 안정하게 학습시키는 weight normalization 방법을 담고 있다. 수학적 이론이 흥미로워서 읽게 되었다.



## Introduction

Discriminator 의 성능을 높이는 것은 GAN에서 아주 중요한 문제다. GAN을 학습할 때 discriminator 가 density ration 를 잘못 추정하기 시작하면,  Generator 또한 target 분포를 잘 학습할 수 없게 된다. 특히 model distribution과 target distribution의 support가 완벽히 disjoint 할 때 이를 완벽하게 판별할 수 있는 discriminator 가 만들어 지는 순간, generator 의 학습은 gradient가 0이 되면서 끝나게 된다.

이러한 문제로 인해, discriminator를 학습할 때 일종의 제약을 걸어서 학습을 안정화시키는 발상이 나오게 된다.

이 논문에서는 새로운 normalization 방법인 'Spectral Normalization' 을 제안한다. 이 방법의 장점은

1. Lipshitz constant 가 유일한 하이퍼 파라미터, 튜닝이 별로 필요없다.
2. 구현이 쉽고 cost가 작다.

이라고 한다.

## Method

우리가 흔히 사용하는 단순한 discriminator는 다음과 같은 꼴로 나타낼 수 있다.


$$
f(x, \theta) = W^{(L+1)}a_L(W^{L}(a_{L-1}(W^{L-1}(...a_1(W^1x)...))))
$$


여기서 $$\theta := \{W^1,...,W^L, W^L+1\}$$ 이다. 

논문에선 이 $$f$$의 Lipschitz constant 를 조절함으로써 discriminator에 제약을 걸었다.

K-Lipschitz continuity란
$$
||f||_{Lip} \le K
$$
를 만족하는 함수를 말하는데,  Lipschitz norm 은

$$||f(x) - f(x^{'})||/ ||x-x^{'}|| \le M$$ 을 만족하는 가장 작은 M을 말한다.

즉, Lipschitz constant 를 제약하는 것은 곧 함수의 기울기에 대한 제약을 뜻한다고도 볼 수 있다.

그리고 이 Lipschitz norm 은 spectral norm과 관련이 깊은데,



작성중..