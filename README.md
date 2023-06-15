# Efficient Bayesian Inference by Conjugate-computation Variational Message Passing

---
**Abstract**

Variational message passing is an efficient Bayesian inference method in factorized probabilistic models composed of conjugate factors from the exponential family (EF) distributions. In many applications, a more accurate model for the process under consideration can be obtained by inserting nonlinear deterministic factors in the model. Unfortunately, variational messages that pass through nonlinear nodes cannot be analytically computed in closed form. In this paper, we derive an efficient algorithm for passing variational messages through arbitrary deterministic factors. Our method is based on projecting outgoing messages onto an EF distribution. We implemented our algorithm in RxInfer, which is an open-source message passing-based inference package in Julia. The resulting implementation yields efficient message passing-based inference in arbitrary models composed of stochastic and deterministic factors. We compare our method to alternative state-of-the-art inference methods and find lower (i.e., better) free energy residuals for the proposed method.


---
This repository contains all experiments of the paper.

**Disclaimer**

For the moment, paper denepds on private package `ExponentialFamily` which is not yet open-sourced. We will open-source it soon.
You can obtain the package by contacting the authors.

