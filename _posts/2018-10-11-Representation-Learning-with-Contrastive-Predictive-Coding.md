---
title: Representation Learning with Contrastive Predictive Coding
date: 2018-10-11
description: paper
---

> [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)

NIPS 2018에 나온 deepmind의 따끈따끈한 논문이다.

이 논문의 목적은 unsupervised learning 으로 representation learning 을 효과적으로 해내는 것이다.

supervised learning 을 통한 latent representation 의 학습은 현재까지도 큰 성과를 내고 있지만, 데이터 효율성이나 일반화 기능에는 여전히 단점이 존재한다.

unsupervised learning 을 통한 high-level representation learning 은 많은 시도가 이루어지고 있는 분야이지만, 가장 널리 사용되는 전략은 현재까지의 정보로부터 미래의 정보를 예측하는 방식이다.

그 중  predictive coding은 신호 처리 및 데이터 압축의 전통적인 방식으로, 현재 신호에서 몇 개 이전의 값들로 부터 미래의 샘플 데이터를 예측 하는 방법이다. 신경 과학에서도 predictive coding 이란 용어가 쓰이는데, 우리 뇌가 주변 상황으로부터 예측한 결과와 실제 발생한 일에 따라 감각 입력을 평가하고 무엇이 잡음인지 신호인지를 결정하는 무의식적인 과정을 뜻한다.

그리고 이러한 predictive coding 방식은,

NLP를 공부해 본 사람이라면 잘 알고 있을 word2vec(https://arxiv.org/abs/1301.3781)에서 unsupervised learning 을 통해 단어 임베딩을 구하는데 사용된다.

![skip-gram](http://i.imgur.com/TupGxMl.png)



이와 같은 아이디어로, 이 논문에서는 high-level latent representation 을 학습하는 걸 일종의 prediction problem으로 두고, 이를 위한 모델 Contrastive Predictive Coding(CPC)를 제시한다. 그리고 다양한 domain(image, audio, natural language, RL) 에서  CPC를 통해 latent representation 을 학습하고, 이를 여러가지 task들에 적용하여 성능을 평가했다.

###  Main Idea

![contrastive predictive coding](https://camo.githubusercontent.com/ab285aadb87cc935d7b0ad1ac94c949ce6e06702/68747470733a2f2f692e696d6775722e636f6d2f444558633552342e706e67)

그림에서 볼 수 있듯, CPC의 모델 자체는 이해하기에 그리 어렵지 않다.

개인적으로 모델 자체보다 이렇게 모델을 설계하게 된 아이디어와 직관에 더 관심이 갔다.

논문에서 저자는 이 모델이 고차원의 신호들이 공유하는 중요한 정보만을 인코딩하고 필요없는 노이즈를 버리는 방식을  학습한다고 말한다.

보통 우리는 과거의 압축된 정보 c로 부터 미래의 정보 x를 예측할 때, $p(x\vert c)$ 를 직접 모델링하는 방식을 사용하지만, 이미지나 문장과 같은 경우 데이터가 가지고 있는 엔트로피는 수천 bits 임에 반해, high-level 의 latent variable 의 경우 그보다 훨씬 작은 정보량(클래스 라벨로 예를 들면, 1024가지의 category들이 있을 때, 이 variable 분포의 엔트로피는 10이 된다)을 가지기 때문에 이러한 방식은 최적의 방법이 아니라고 말한다.

즉, 데이터에 우리가 필요하지 않은 noise 가 엄청 많이 포함되어 있으므로, c로 부터 x를 직접 예측하는 건 제대로 된 방법이 아니라고 말하는 듯 하다.

아무튼, 이 논문에서는 대신 x와 c의 '상호 정보(mutual information)' 을 최대화 하는 쪽에 초점을 맞춘다.

$$
I(x;c) = \sum\limits_{x,c} p(x,c)\log\frac{p(x|c)}{p(x)} = H(x)-H(x|c)
$$

식에서 볼 수 있듯이, 상호 정보는 관측 전의 엔트로피와 관측 후의 엔트로피 차이를 의미한다.
MI를 maximizing 하는 것은 곧 c에 x를 추정하는데 중요한 정보가 많이 들어있기 때문에 미래 관측에 대한 불확실성인 $H(x|c)$ 가 작아짐을 의미한다고 해석했다.

### Contrastive Predictive Coding

위의 그림에서 볼 수 있듯, 구성은 총 세 가지 파트로, 

- 고차원 data의 정보를 압축하는 encoder $g_{enc}$ 
- 압축된 latent space에서 context vector 를 만들기 위한 autoregressive model $g_{ar}$
- 마지막으로 word2vec 에서 사용됐던 Noise-contrastive-estimation 방식으로 loss를 구하여 end-to-end 학습이 가능하게 한다.

위에서 말했듯, 이 논문에서는 $p(x_{t+k}|c_t)$  를 직접 모델링하지 않는다.
대신,  density ratio $f_k$ 를 모델링하는데,

$$
f_k(x_{t+k},c_t) \propto \frac{p(x_{t+k}|c_t)}{p(x_{t+k})}
$$

위에서 말한 mutual information 을 보존하기 위해 $\frac{p(x_{t+k}|c_t)}{p(x_{t+k})}$ 에 비례하게 둔다.(추후 설명)
이 논문에서는 log-bilinear model 을 사용해서

$$
f_k(x_{t+k},c_t) = \exp(z_{t+k}^TW_kc_t)
$$

density ratio를 추정한다. 여기서 $W_kc_t$ 대신 non linear network 나 RNN 류를 사용해도 된다고 한다.
이렇게 구한 density ratio $f_k$ 와 $z_{t+k}$를 이용해서, 우리는 $x_{t+k}$ 를 구할 수 있게 된다. 비록 $p(x), p(x|c)$ 를 직접 알 수는 없지만, 저 확률 분포에서의 sampling 은 가능하므로(이 부분은 현재 이해도가 떨어짐..), 앞으로 noise-contrastive-estimation 을 사용할 수 있게 된다. 

### Noise Contrastive Estimation Loss

word2vec 에서와 비슷하게, 여기서도 loss 함수로 nce loss 를 사용한다. 논문에선 N 개의 random sample $X ={x_1, ... x_N}$ 뽑는데, $p(x_{t+k}|c_t)$ 에서 positive sample 하나, 나머지 N-1개를 negative sample 을 $p(x_{t+k})$ 에서 뽑은 다음, 이를 구별하는 법을 학습한다.
여기서 제시한 loss function은,

$$
\mathcal{L}_N = - \displaystyle \mathop{\mathbb{E}}_{X}[\log\frac{f_k(x_{t+k},c_t)}{\sum\limits_{x_j\in X}f_k(x_j, c_t)}]
$$

이런 형태가 되는데, positive sample 을 구별하는 categorical cross-entropy 형태이다. log 안의 $\frac{f_k}{\sum\limits_{X}f_k}$ 가 곧prediction 을 나타낸다.
그런데 이 prediction의 optimal value는, $x_i$가 positive sample, 즉 $p(x_i|c_t)$ 에서 뽑히고 나머지는 $p(x_i)$ 에서 뽑힐 확률과 같다.

$$
\begin{align} p(d=i|X,c_t) & = \frac{p(x_i|c_t)\prod\limits_{l\neq i}p(x_l)}{\sum\limits_{j=1}^N p(x_j|c_t)\prod\limits_{l\neq j}p(x_l)} 
\\\\ & = \frac{\frac{p(x_i|c_t)}{p(x_i)}}{\sum\limits_{j=1}^N\frac{p(x_j|c_t)}{p(x_j)}}\end{align}
$$
즉, $f(x_{t+k}, c_t)$ 의 optimal value는 위에서 말했던 것처럼 $\frac{p(x_{t+k}|c_k)}{p(x_{t+k})}$ 에 비례하게 되고, 이는 negative sample을 어떻게 뽑아도 보존된다. 또한, Appendix 에서는 mutual information 의 lower bound에 대한 증명을 제공했다.
$$
I(x_{t+k}, c_t)\geq \log(N)-\mathcal{L}_{N}
$$

식에서 볼 수 있듯,  N의 크기를 키울 수록, 또 $\mathcal{L}_{N}$ 을 maximize 할 수록 lower bound 가 증가한다.

### Experiment

위에서 설명한 모델을 바탕으로, vision, audio, natural language, reinforcement learning 4가지 분야에 대한 실험을 진행했다. 각 실험에 대한 상세한 정보나 스펙은 논문을 참고하길 바란다.

####  Audio

![Speaker classification](/assets/post_image/1539223980806.png) ![1539224110644](/assets/post_image/1539224110644.png)

#### Vision

![1539224190859](/assets/post_image/1539224190859.png){: width="300" height="300"}

#### Natural Language

![1539224337211](/assets/post_image/1539224337211.png)

#### Reinforcement Learning

![1539224529659](/assets/post_image/1539224529659.png)

아직 코드는 공개되지 않았지만, vision 쪽의 코드는 다른 사람이 구현해놓은게 존재한다. 이해하는데 도움이 될 것이다. 

https://github.com/davidtellez/contrastive-predictive-coding



