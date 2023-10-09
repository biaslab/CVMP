# Efficient Bayesian Inference by Conjugate-computation Variational Message Passing

---
**Abstract**

Variational message passing is an efficient Bayesian inference method in factorized probabilistic models composed of conjugate factors from the exponential family (EF) distributions. In many applications, a more accurate model for the process under consideration can be obtained by inserting nonlinear deterministic factors in the model. Unfortunately, variational messages that pass through nonlinear nodes cannot be analytically computed in closed form. In this paper, we derive an efficient algorithm for passing variational messages through arbitrary deterministic factors. Our method is based on projecting outgoing messages onto an EF distribution. We implemented our algorithm in RxInfer, which is an open-source message passing-based inference package in Julia. The resulting implementation yields efficient message passing-based inference in arbitrary models composed of stochastic and deterministic factors. We compare our method to alternative state-of-the-art inference methods and find lower (i.e., better) free energy residuals for the proposed method.


---
This repository contains the Sunspot experiment from the paper. The experiment is implemented in Julia using the RxInfer package.

In this repository there are 3 notebooks:

- `sunspot_cvmp.ipynb`: contains the experiment with the proposed method
- `sunspot_nuts.ipynb`: contains the experiment with the NUTS sampler (implemented in Turing.jl)
- `sunspot_svmp.ipynb`: contains the experiment with the SVMP method (implemented in RxInfer.jl)

The paper also use results of AIS-MP algorithm on the same dataset for the comparison, which is not presented in this repo and implemeted in the ForneyLab package [here](https://github.com/biaslab/AIS-MP/blob/correcting-free-energy/demos/GammaStateSpace%20AIS-MP.ipynb).
