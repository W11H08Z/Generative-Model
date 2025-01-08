## 项目概述

本项目旨在探索和比较四种不同的生成式模型在奖杯数据集上的表现，包括朴素非线性隐变量模型（NNLVM）、变分自编码器（VAE）、生成对抗网络（GAN）和扩散模型。关于项目的详细介绍参考[]()

## 环境配置

```shell
pip install -r requirements.txt
```

## 生成式模型介绍

### Naive Nonlinear Latent Variable Model (NNLVM)

NNLVM是一种基础的生成式模型，通过非线性变换将隐变量映射到数据空间。其核心思想是利用隐变量的高斯分布特性，结合非线性映射函数，生成与真实数据相似的样本。然而，由于维数灾难和采样点有限等问题，NNLVM在奖杯数据集上的表现并不理想，生成的图像较为模糊。

运行如下代码即可训练
```shell
bash train_NNHVM.sh
bash train_NNHVM_CNN.sh
```

### Variational Autoencoder (VAE)

VAE在NNLVM的基础上引入了编码器和解码器结构，通过最小化重构误差和KL散度来优化模型。VAE能够学习到数据的潜在表示，并在隐变量空间中进行采样生成新样本。尽管VAE在理论上具有优势，但在实际应用中仍可能出现后验坍塌现象，导致生成样本的质量受限。

运行如下代码即可训练
```
bash train_VAE.sh
bash train_VAE_CNN.sh
```

### Generative Adversarial Network (GAN)

GAN由生成器和判别器组成，通过对抗训练的方式优化模型。生成器负责生成逼真的样本，而判别器则负责区分真实样本和生成样本。GAN在生成高质量样本方面具有显著优势，样本质量方差较大，并且可能出现模式崩溃等问题。

运行如下代码即可训练
```
bash trian_GAN.sh
bash train_GAN_CNN.sh
```

### Diffusion Model

扩散模型通过逐步逆转噪声过程来生成数据，具有生成样本质量高且稳定的特点。其核心在于利用神经网络学习噪声逆转过程，并通过优化对数似然函数的下界来训练模型。在奖杯数据集上，扩散模型展现出了优异的生成性能。

运行如下代码即可训练
```
bash train_Diffusion_Model.sh
```