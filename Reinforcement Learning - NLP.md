<h1>Reinforcement learning in NLP and its applications</h1>

References
- [RLHF - chatbots and LLMs](https://learning.oreilly.com/library/view/deep-reinforcement-learning/9798868802737/)
- []()
- []()
- []()

# Overview of reinforcement learning 

# Markov decision process

# Model-based approaches

# Model-free approaches

# Function approximation

# Deep Q-learning


# Poolicy gradient algorithms

# Combining policy-gradient and Q-learning

# Proximal policy optimization (PPO) and RLHF

PPO algorithm uses human annotated preference as a reward mode to fine-tune LLMs to follow the human preferences. 

History: it is noted controling size of policy updat helps to stablize the model better than controlling the update of network parameters. There are two approaches for controlling the policy update: 1-- trust region policy optimization, and 2- proximal policy optimization. They both make sure a new update to policy does not change it significantly. 

In TRPO approach, we build a hard constrain around the policy update via defining Kullback-Liebler (KL) distance between current and new policy and limit it to small value. Basically we build a trust region around the policy update to control its volatility. However, computing KL distance is not easy and instead we employ approximation (Tylor) approach and replace KL with sample-based estimation. Then we compute Hessian of the sample average KL divergence as a part of the second step. And finaly update parameters that both improves the sample loss and satisfies the KL divergence constraint [TRPO paper](https://arxiv.org/pdf/1502.05477). Although the paper shows the proposed approach works, it requires calculating and inverting a very large matrix for Hessian computation part.

PPO is an alternate between sampling data through interaction with the environment and optimizing a “surrogate” objective function using stochastic gradient ascent. This makes it more efficient compare with TRPO approach [PPO paper](). 

## PPO - step by step

### Score function and MLE estimator
Given probability distribution of some samples, computigng joint probability conditioned on the unknown parameter $\theta$ can be computed as $P(x_1, ..., x_n|\theta) = \prod P(x_i|\theta)$. After transforming both side with log-transformer we have $\log P(x_1, x_2| ...|x_n|\theta) = \sum \log P(x_i|\theta)$. Finally taking average and multiply it by -1, we will have what is called _negative log loss_ as follow: $NLL = - \frac{1}{n\}\log P(x_1, x_2| ...|x_n|\theta) =  - \frac{1}{n} \sum \log P(x_i|\theta)$. The optimization goal then will be to minimize $NLL$ w.r.t. $\theta$. In stochastic gradient descent approach, we adjust $\theta$ by computing gradient of $NLL$ w.r.t. $\theta$: $\theta = \theta + \alpha \nabla_\theta NLL$ 

Score function  $score(\theta)  =  \nabla_\theta \log P(x_i| \theta)$ is the term inside the sum in $\nabla_\theta NLL= - \frac{1}{n} \sum \nabla_\theta \log P(x_i| \theta)$. Score function in statistics is used to form most likelihoodestimator (MLE). Here we find $\theta$ that makes the derivative zero. In another word minimize NLL.

In the realm of machine learning, the score function is particularly useful in gradient-based optimization algorithms. It serves as a guide to navigate the parameter space in order to find optimal parameters that minimize or maximize a chosen objective function. In reinforcement learning, the score function plays a key role in policy gradient methods, where it helps estimate the gradient of expected rewards with respect to policy parameters, enabling the development of strategies that maximize cumulative rewards.




### Covariance of the score function, Hessian Matrix and Fisher Information Matrix

Hessian H, it is a square matrix that computed by taking the second-order partial derivative of a scalar function such as the likelihood function. The negative expected Hessian of log likelihood is equal to the Fisher Information Matrix, F: $E_{P(x|\theta)}[H_{\log P(x|\theta)}] = -F$

, and natural gradients,


# Multi-agent reinforcement learning (MARL)

# 
