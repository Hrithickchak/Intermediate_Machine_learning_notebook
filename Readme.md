
In this project:
implemented 2 different GLMs based on the derivations that I have done in class indermediate machine learning 2. 
As a warm-up---and a soft introduction to Julia--- extended previous implementation of linear regression to (a) use an L2 regularizer and then (b) to use the ADAM optimizer. Our implementation of linear regression is simple: we use stochastic gradient descent (SGD), with a scalar fixed stepsize, where each update only uses one sample 
	
first created a synthetic dataset. The purpose of this is to create controlled situations to test your methods. We can create synthetic datasets that match the assumptions underlying a method, and so test our implementations. If they do not behave as we expect, then this might indicate a bug (conceptually or in our code).
	
A synthetic data where the targets $y$ are generated by sampling from a Gaussian where the mean is a linear function $f$ of $\mathbf{x}$ with some fixed variance $\sigma^2$. This can be done programatically by (1) generating a random input vector $\mathbf{x}$, (2) sampling noise $\epsilon \mathcal{N}(0, \sigma^2)$ and (3) setting the target to $y = f(\mathbf{x}) + \epsilon$. 
	
We did not discuss how to generate $\mathbf{x}$, nor did we make assumptions about $p(\mathbf{x})$ in linear regression. But, to create a synthetic dataset, we need to decide what distribution to use. A simple choice is to generate these inputs randomly from Gaussians (just because we are so used to Gaussians). To make the outcome more interesting on this synthetic data, we generate the inputs from an inverse Gaussian distribution. We also add dependencies between these inputs, to see the impact on linear regression. This function that generates $\mathbf{x}$ is in generate_features.
	
implemented simple linear regression with stochastic gradient descent updates using a fixed stepsize.
However, it is common to use more sophisticated optimization techniques such as ADAM which adapt the stepsize based on information from consecutive updates.
In this section, we will extend our linear regression implementation to also allow the use of the ADAM optimizer.

specify most of the data-generating function, but we will get you to help us with part of the implementation. We need your help implementing the softmax and helping us generate labels using our data-generating function. Together, we implement a function `g(x)` that produces a vector ``y \in [0,1]^4`` for each sample `x`, where we ensure that we have class balance. In other words, the dataset has a roughly even number of samples per class, with at least 3000 of each class present. Further, we ensure that `g(x)` induces a valid Multinomial distribution over the 4 classes. Therefore, `y = g(x)` should sum to 1 for each input vector

The LinearModel  is immutable (to create a mutable struct, we need to explicitly state that it is mutable). Hence, we cannot change the reference of lm.W to refer to another weight vector, W'. What we can do though, is mutate lm.W, and the immutable LinearModel struct will not complain (indeed, the reference lm.W has not changed, but what is at this reference has changed). If we use the = operator, this will re-assign the lm.W reference to refer to a new object. So lm.W = lm.W .- sgd.α * ΔW is not an allowable operation. If we use the vectorized version (.=), then we will re-assign the elements of the lm.W struct (i.e. mutate the lm.W struct, without changing the reference to lm.W), which is allowable. Hence, the followling line of code works:
lm.W .-= sgd.α * ΔW
