---
layout: post
title:  "Coursera: Notes on Time Series"
date:   2015-08-15 15:00:00
---

### **Stationary Processes** ###

#### Concept 1: Strictly Stationarity

* Y_t has the same mean, variance for all t
* Any function/transformation g() of a strictly stationary process is also strictly stationary
![strictly stationarity](/images/strictly_stationary.png)

#### Concept 2: Covariance (Weakly) Stationary Processes

![covariance stationarity](/images/covariance_stationary.png)

#### Concept 3: Autocorrelation Function (ACF)

Plot the autocorrelation against j, where j is time lag

#### **Gaussian White Noise Model**

![gaussian white noise process](/images/gaussian_white_noise.png)

#### **Independent White Noise Model**

![independent white noise process](/images/independent_white_noise.png)

#### **Moving Average (MA) Processes**
Specifically talked about MA(1), it's a `convariance stationary process` as \rho_t doesn't depend on t, but does depend on j.

* Mean and Variance are constant over time
* Autocovariance(1) is \theta/(1+\theta^2), and autocovariance(k) is 0 for k > 1
![ma 1](/images/ma_1.png)

#### **Autoregressive (AR) processes**

If we **assume** AR(1) is covariance stationary, then it exhibits a geometric decay in auto-correlation. Random variables close in time are more closely related than random variables are farther away in time.

**Ergodicity**: Y_t and Y_t-j for j large are essentially independent from each other. Autoregressive processes has this property.

* E[Y_t] = \mu
* Var(Y_t) = \sigma^2 / (1 - \phi^2)
* corr(Y_t, Y_t-1) = \phi
* corr(Y_t, Y_t-j) = \phi^j
* lim_j->\infty corr(Y_t, Y_t-j) -> 0

![Autoregressive process](/images/autoregressive_process.png)

### **Non-Stationary Processes**

Concept 4: This simply means that `something` is not constant over time. Non-stationarity is not very descriptive enough, and there are many flavors of where the non-stationary is coming from.

#### **Deterministically trending process**

Non-deterministic in the mean. i.e. E[Y_t]
![deterministically trending process](/images/deterministically_trending_process.png)

#### **Random Walk**

Non-deterministic in the variance. i.e. Var(Y_t)
![random walk](/images/random_walk.png)