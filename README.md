# Learning-Latent-Partial-Matchings-with-Gumbel-IPF-Networks

This is the PyTorch implementation of the AISTATS 2024 paper titled "Learning Latent Partial Matchings with Gumbel-IPF Networks".

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

