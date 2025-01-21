# Code for Phase-aware Training Schedule Simplifies Learning in Flow-Based Generative Models
This is the official repo for the experiments from the paper [Phase-aware Training Schedule Simplifies Learning in Flow-Based Generative Models](https://arxiv.org/abs/2412.07972) by S. Aranguri and F. Insulla (under review).

## Generating MNIST digits
In ..., there is an implementation of a score-based diffusion model using the U-Net to generates MNIST digits which shows that the way we propose to train in our paper improves class accuracy more than 5x over regular methods when trained on a limited number of epochs.

## Numerical check of theoretical predictions
In the `theoretical_predictions` folder, there is `learnGMM.py` which implements a neural network that is trained to generated sample from a mixture of two gaussians. Further, in `overlaps.ipynb` we provide the code to numerically check the agreement between the theoretical predictions made in the paper for the learned weights against actual values of the weights for SGD trained neural network. 
