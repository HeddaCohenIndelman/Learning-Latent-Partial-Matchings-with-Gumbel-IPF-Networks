# Learning-Latent-Partial-Matchings-with-Gumbel-IPF-Networks


## Abstract 
Learning to match discrete objects has been a central task in machine learning, often facilitated by a
continuous relaxation of the matching structure.
However, practical problems entail partial matchings due to missing correspondences, which pose difficulties to the one-to-one matching learning techniques that dominate the state-of-the-art. 
This paper introduces Gumbel-IPF networks for learning latent partial matchings. 
At the core of our method is the differentiable Iterative Proportional Fitting (IPF) procedure that biproportionally projects onto the transportation polytope of target marginals. 
Our theoretical framework also allows drawing samples from the temperature-dependent partial matching distribution. 
We investigate the properties of common-practice
relaxations through the lens of biproportional fitting and introduce a new metric, the empirical
prediction shift. Our method's advantages are demonstrated in experimental results on the semantic keypoints partial matching task on the Pascal VOC, IMC-PT-SparseGM, and CUB2001 datasets.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-latent-partial-matchings-with-gumbel/graph-matching-on-cub)](https://paperswithcode.com/sota/graph-matching-on-cub?p=learning-latent-partial-matchings-with-gumbel)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-latent-partial-matchings-with-gumbel/graph-matching-on-pascal-voc)](https://paperswithcode.com/sota/graph-matching-on-pascal-voc?p=learning-latent-partial-matchings-with-gumbel)
## How to use this repository 
We use unchanged peer methods implementations in the unified [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) project as much as possible. 

### Datasets 
Datasets should be downloaded and organized as instructed in the [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) project. 

