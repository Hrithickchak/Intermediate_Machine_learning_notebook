### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ e1909389-766f-4fad-b1b8-1efd4fb3685e
using CSV, PlotlyJS, LinearAlgebra, Random, Distributions, PlutoUI

# ╔═╡ 8c52dc53-c488-48da-86e0-18c8625579c8
md"
# Intermediate Machine Learning Algorithms and their implementation
"

# ╔═╡ 84c9b59f-377e-405f-836e-fcd287e964d2
md"
# Hrithick Chakraborty
"

# ╔═╡ b8504526-9e4f-4e12-a900-99081d2c0641
md"#### Preamble
 - Loading Packages
 - Generating Utilities
"

# ╔═╡ 4731f58f-1697-40ee-9cf9-fd31311c1bf5
_check_complete(complete) = complete ? "✅" : "❌"

# ╔═╡ 6d5b815d-fb8f-4dc7-9355-83f2e4e30c92
import LinearAlgebra: norm, norm1, norm2

# ╔═╡ efd24c67-8ed0-45bc-ba63-fc3ec53ff840
PlutoUI.TableOfContents()

# ╔═╡ 777edfdf-7163-4b40-8dfe-cc55e741423f
# A couple of test utilities.
begin
	function isclose(a, b, prec=1.e-8)
		norm(a - b) < prec
	end
end

# ╔═╡ fb422f47-6c47-4a7e-b044-26ca14574cad
begin

	md"""
	# Generalized Linear Models

	implement 2 different GLMs based on the derivations that we have done in class and in the notes.
	As a warm-up---and a soft introduction to Julia---you will extend our implementation of linear regression to (a) use an L2 regularizer and then (b) to use the ADAM optimizer. Our implementation of linear regression is simple: we use stochastic gradient descent (SGD), with a scalar fixed stepsize, where each update only uses one sample $(\mathbf{x}_i, y_i)$ (rather than a larger mini-batch):
	
	$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta (f_{\mathbf{w}_t}(\mathbf{x}_i) - y_i) \mathbf{x}_i$$
	
	After completing these two modifications to linear regression, then you will implement your own multinomial logistic regression.

	"""
end

# ╔═╡ cce3a5a5-9e37-4ca8-a9bc-8ad1de773d33
num_samples = 30000

# ╔═╡ 1b90c1ff-1585-4c75-9961-6a411ad37531
# Convenient to have number of features assigned to a variable
num_features = 15 + # base features 
	           1    # added column of ones

# ╔═╡ 96fa5d34-8a99-4be4-9917-78f79b53b490
"""
	generate_features(rng)

This function generates features according to a InverseGaussian distribution 
(μ = 0.1 and λ = 1).
"""
function generate_features(rng)
	# We use a funky distribution (the inverse gausian) to help 
	# detect bugs later on! If we use a Normal dist here, might 
	# hide some bugs.
	μs = rand(rng, 10)
	X = reduce(hcat, [rand(rng, InverseGaussian(μs[i], 1), num_samples) for i in 1:10])

	# Maybe Add some components which are linearly dependent
	X = hcat(X, 0.33*X[:, 1] .+  0.67*X[:, 5])
	X = hcat(X, -X[:, 4])
	X = hcat(X, X[:, 3] .- X[:, 8])
	X = hcat(X, X[:, 9] .+ 0.5*X[:, 10])
	X = hcat(X, X[:, 6] .- 0.5*X[:, 2])
	

	# let's also append a column of ones to act as a bias unit
	X = cat(X, ones(num_samples), dims=2)
	
	return X
end

# ╔═╡ 0c8e7b09-e202-412c-8e5d-f963a77820a8
md"
In the below, specify the linear function that we will use to generate the synthetic data. We leave it up to you to pick this linear function, but there are a few restrictions:
1. Don't use $f(x) = c$ for some constant $c$.
2. Make sure $f(x)$ depends on at least 2 columns of $X$
3. Your function should be single variate (1 output)
"

# ╔═╡ 8d6eb957-2833-4cae-8e78-f0ac2f562952
W_true = let

	w_temp = rand(Random.Xoshiro(10), Normal(0, 0.2), num_features, 1)

	# TODO: Here you can modify w_temp. The weight W_true will be set to w_temp. This is the place where you get to specify the weights for the function $f$, below written as f(X::AbstractMatrix).
	
	w_temp = w_temp .^ 1/2


	w_temp
end

# ╔═╡ 78447784-2d46-4b25-9654-59e9e962b255
# TODO: f should be a linear function, that uses the weights W_true
function f(X::AbstractMatrix)

	y_target = X * W_true	
end


# ╔═╡ 9d037c81-61bb-46da-89b1-384672128a47
__check_f_vec_addition = let
	rng = Random.Xoshiro(42)
	a = rand(rng, Uniform(-10, 10), 100, num_features)
	b = rand(rng, Normal(2, 3), 100, num_features)

	f(a) + f(b) ≈ f(a + b)
end

# ╔═╡ e39fbb78-a37f-4da4-9edf-31562387d20e
__check_f_scalar_mult = let
	rng = Random.Xoshiro(42)
	a = rand(rng, Uniform(-10, 10), 100, num_features)
	c = rand(rng, Uniform(-5, 5))

	c .* f(a) ≈ f(c .* a)
end

# ╔═╡ eb8b1686-f7f4-4db1-b944-e893e0f3e9e5
begin
	# ! This code block is for testing, feel free to hide it! You don't need to do anything here.
	
	# Simply check to ensure that f is a linear function, i.e. f must satisfy:
	# 1. f(a) + f(b) = f(a+b)
	# 2. c * f(a) = f(c * a)

	__check_syn_data = try
		__check_f_scalar_mult && __check_f_vec_addition
	catch
		false
	end
	md"""
	## $(_check_complete(__check_syn_data)) Synthetic Data [5pts]

	You will start by creating a synthetic dataset. The purpose of this is to create controlled situations to test your methods. We can create synthetic datasets that match the assumptions underlying a method, and so test our implementations. If they do not behave as we expect, then this might indicate a bug (conceptually or in our code).
	
	Here you will first create synthetic data where the targets $y$ are generated by sampling from a Gaussian where the mean is a linear function $f$ of $\mathbf{x}$ with some fixed variance $\sigma^2$. This can be done programatically by (1) generating a random input vector $\mathbf{x}$, (2) sampling noise $\epsilon \mathcal{N}(0, \sigma^2)$ and (3) setting the target to $y = f(\mathbf{x}) + \epsilon$. 
	
	We did not discuss how to generate $\mathbf{x}$, nor did we make assumptions about $p(\mathbf{x})$ in linear regression. But, to create a synthetic dataset, we need to decide what distribution to use. A simple choice is to generate these inputs randomly from Gaussians (just because we are so used to Gaussians). To make the outcome more interesting on this synthetic data, we generate the inputs from an inverse Gaussian distribution. We also add dependencies between these inputs, to see the impact on linear regression. This function that generates $\mathbf{x}$ is in generate_features.
	
	Later you will reason on what types of datasets we expect l2 regularization to improve performance.
	
	We have provided the function that generates $\mathbf{x}$. Your job is to specify a linear function $f$ for this synthetic data generation. We will also provide the code that generates the Gaussian noise, to create the target $y$.
	
	"""
end

# ╔═╡ a16d8fff-4b50-4d56-9acb-7b96e0367d15
let

	__local_check_f_scalar_mult = try
		__check_f_scalar_mult
	catch
		false
	end

	__local_check_f_vec_addition = try
		__check_f_vec_addition
	catch
		false
	end
	
	md"""
	### $(_check_complete(__local_check_f_vec_addition && __local_check_f_scalar_mult)) Checkpoint: Is `f` Linear?
	
	To test this question we use two properties of linear functions:
	
	1. Preserve vector addition $(_check_complete(__local_check_f_vec_addition)): $$f(\mathbf{a}) + f(\mathbf{b}) ≈ f(\mathbf{a} + \mathbf{b})$$
	2. Preserve scalar multiplication $(_check_complete(__local_check_f_scalar_mult)): $$cf(\mathbf{a}) ≈ f(c * \mathbf{a})$$
	
	"""
end

# ╔═╡ 60f2b901-6fcd-43d9-b0f1-4f9e802e87e9
"""
	generate_targets(X::AbstractMatrix, rng::Random.AbstractRNG)

This function takes features `X` and a random number generate `rng` and returns the targets for our synthetic dataset. The targets are generated using the linear function `f(X)` plus noise sampled from a Normal distribution with μ = 0 and σ² = 1. 
"""
function generate_targets(X::AbstractMatrix, rng::Random.AbstractRNG)
	# Generate Y ~ Normal(f(x), 1)
	# You should use the above function:
	# f(X): For generating the non-perturbed targets
	# rand(rng, ...): For generating the random noise

	rng = Random.MersenneTwister(1270)
	noise = rand(rng, Normal(0, 1),30000,1)
	Target = f(X) + noise
	
	
	
end

# ╔═╡ 32e4a3ac-1864-4b1c-9b76-670c529b9d67
md"""Now we generate the data we are going to use for the Linear models!"""

# ╔═╡ 8bbc7546-46d6-4c90-993d-823e420373b6
X, Y = let
	rng = Random.Xoshiro(39432)
	X = generate_features(rng)
	Y = generate_targets(X, rng)
	X, Y
end

# ╔═╡ edefd098-d2d5-46c1-91e4-2529267b54f1
md"
If you have implemented the random perturbation correctly, you should see an approximately normal distribution below.
"

# ╔═╡ d86f970a-7dd2-4e5e-b44a-4c966a62a536
let
	
	trace = histogram(Dict(:x=>Y[:, 1]), kind="histogram")
	plt = Plot(trace)
	PlotlyJS.relayout(plt, height=400, showlegend=false)
end

# ╔═╡ 52502f3b-8a7c-41b6-8405-35a889964876
begin
	RMSE(x̂, x) = norm2(x̂ .- x) # equivalent to norm(x̂ .- x, 2)
	L2Error = RMSE
	L1Error(x̂, x) = norm1(x̂ .- x)
end

# ╔═╡ 17e1b66c-b5bd-4f6a-8682-c5f7fc686df3
begin
	# Create the shared parent type
	abstract type Model end

	# Create a placeholder for the `predict` function
	function predict end

	# Create a shared optimizer parent type
	abstract type Optimizer end
end

# ╔═╡ f9fe90fb-7a52-4769-866f-f548bb77d2bb
md"We split our data into a train/test split."

# ╔═╡ 04669dcd-7973-4e82-9e6f-ac155106a2b0
begin
	trainX = X[1:25000, :]
	testX = X[25001:end, :]

	trainY = Y[1:25000, :]
	testY = Y[25001:end, :]
end;

# ╔═╡ 75246b41-a7c1-4e3b-8ee0-89d733a22a15
begin
	mutable struct LinearModel <: Model
		W::Matrix
	end

	# add a default initializer (all weights=0)
	LinearModel(in, out=1) = LinearModel(
		# initialize W
		zeros(in, out)
	)

	# For consistency with notes, define inner-product with column vectors
	predict(lm::LinearModel, x::AbstractVector) = x'*lm.W
	# and with matrices
	predict(lm::LinearModel, X::AbstractMatrix) = X*lm.W
end

# ╔═╡ b66c2fc2-b3be-415b-88e0-a8415d2380a4
struct SGD <: Optimizer
	α::Number
end

# ╔═╡ 6ca67862-ecd6-4996-82be-d2caf9ea0445
md"We will set the number of epochs globally to be used by all the algorithms, and define our first run function."

# ╔═╡ ec44beaf-41b5-4e00-9f5e-2a17a7ff6606
NUM_EPOCHS = 10

# ╔═╡ cf388af0-3d04-41a2-b009-245356c78e63
md"
In this section, you will implement linear regression with l2 regularization (RLR).
Modify the update rule for the linear regression solution given above to add l2 regularization. The update is

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta (f_{\mathbf{w}_t}(\mathbf{x}_i) - y_i) \mathbf{x}_i - \eta \lambda  \mathbf{w}_t$$

Note that technically in the notes $\lambda$ is scaled by the number of samples. However, here we have a fixed number of samples. We therefore keep it simpler here and just specify an appropriate constant for the regularizer.
"

# ╔═╡ 1e9a6ebc-8e79-47cd-ad82-de114232f97a
# We will get you started with the RegularizedModel interface.
mutable struct RegularizedModel <: Model
	W::Matrix
	lambda::Number
	RegularizedModel(in::Int, out::Int, lambda::Number) = new(
		zeros(in, out),
		lambda
	)
end


# ╔═╡ a812b5a7-3c5e-4688-b38f-76f8d096ca03
md"The predict functions:"

# ╔═╡ 78b77bc9-2aab-4a7a-b272-ef03953fc4e6
# How do we use it to make predictions?
# for vectors
predict(m::RegularizedModel, x::AbstractVector) = x'*m.W

# ╔═╡ f74938f8-00f0-4c9e-aeda-9f02ffb81a7d
# and for Matrices
predict(m::RegularizedModel, X::AbstractMatrix) = X*m.W

# ╔═╡ 8513b104-5577-4831-b3b5-2eb61021f11f
md"Now we need to implement the `update!` function for the `RegularizedModel`! Below this function (which you need to implement) you should see a test cell. You can use this to test your code, but ask that you don't change the code in this cell. Doing so will result in point deductions!!!" 

# ╔═╡ adccdd65-c7aa-4b5b-9c7e-f17bd3232485
md"Now we train a regularized model λ = 1.0 for the synthetic dataset with SGD α=0.01."

# ╔═╡ b89f82d7-0298-4a8f-8163-04ce4533f9b8
md"
In the last section, we implemented simple linear regression with stochastic gradient descent updates using a fixed stepsize.
However, it is common to use more sophisticated optimization techniques such as ADAM which adapt the stepsize based on information from consecutive updates.
In this section, we will extend our linear regression implementation to also allow the use of the ADAM optimizer.

The original ADAM paper can be found [here](https://arxiv.org/pdf/1412.6980.pdf). The notes also provide an explanation and pseudocode for the ADAM optimizer.
"

# ╔═╡ 9b13b44c-9cf5-456b-b77a-a0969d28d5a2
md"
## ADAM Struct

First we need to a struct that defines the hyperparameters for ADAM.
ADAM also has a state which needs to be maintained (i.e. $m_t$ and $v_t$), we will store these on the object as well.
We will also use this type to tell Julia which `update!` function to use, the one for ADAM or the one for SGD.

The ADAM update rule is as follows. Update the exponential moving averages of the gradient and squared gradient, in `m` and `v` respectively.

``
m_{t} = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_w f(w)
``

``
v_{t} = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_w f(w))^2
``

Then make sure to do a *initialization bias correction* to correct for the fact we initialize the moving averages to zero.

``
\hat{m}_{t} = m_t / (1-\beta_1^t)
``

``
\hat{v}_{t} = v_t / (1-\beta_2^t)
``

Finally we update the model weights.

``
w_{t} = w_{t-1} - \alpha \frac{m_{t}}{\sqrt{v_{t}} + \epsilon}
``

where the square for $v_t$ is element-wise.
Both $\beta_1$, $\beta_2$, and $\epsilon$ are new hyperparameters for ADAM, and $\alpha$ is still a consistent global stepsize.
"

# ╔═╡ 661573b8-f465-48b8-bd1d-f2c720fd593b
mutable struct ADAM <: Optimizer
	α::Float64
	β1::Float64
	β2::Float64
	ϵ::Float64

	m::Matrix{Float64}
	v::Matrix{Float64}
	
	t::Int # for bias correction

	# here's how we will instantiate an ADAM object
	ADAM(α, β1, β2, in, out) =
		new(α, β1, β2, 1e-8, zeros(in, out), zeros(in, out), 0)
end

# ╔═╡ 75051421-8e37-491e-8e48-dda77363364e
md"
One nice thing about ADAM is that it does not require changes to our `train!` function that we've already implemented.
We still need to loop through each element in the dataset in a random order, then call the `update!` function with one data-point at a time.

So all we need to implement now is the new `update!` function.
We've started you off with the function signature.

**Hint:** Don't forget that elementwise operations in Julia are prepended by a dot. For example:
```julia
x = ones(10)
elementwise_sqrt = sqrt.(x)
elementwise_add = x .+ 1
```
"

# ╔═╡ b9c5a48b-6949-4ce2-abb6-f0669a88bd9f
md"
In this section of the notebook, we will be implementing another GLM: multinomial logistic regression.
As with the LR implementation, we will start by constructing a synthetic dataset.

The update rule can be found in section 5.5 of the notes.
"

# ╔═╡ 3fa4d186-b207-45ec-868a-84b6a5187aae
md"
Before we continue to implement `g(x)`, we need a helper `softmax` function. We need your help to implement that now. 
"

# ╔═╡ 653db006-f1e7-4f43-9f44-5a5681d030d8
function softmax(y::AbstractVector)
	fill(1/length(y), length(y)) # default uniform that has nothing to do with data.

	exp.(y) ./ (sum(exp.(y), dims=1))
end

# ╔═╡ e1c7b036-3fb8-4708-a8b4-edf8a0a46c27
md"
Now here is our `g(x)` implementation.
"

# ╔═╡ 29cdfb84-d10f-4d3c-a325-c2d445a7afbf
md"
Now finally we need your help using generate_probabilities to actually get the labels for our data matrix `X`.
"

# ╔═╡ 540cae75-db31-4d4d-b853-1031bc5bf2a8
md"Like before we start by defining the struct and the prediction functions. Except this time you need to implement the prediction functions for the `MultinomialLogisticModel`. You can use the `softmax` functions defined for matrices and vectors."

# ╔═╡ 92f32acf-9ad8-438a-b289-d46e1f97b17e
mutable struct MultinomialLogisticModel <: Model
	W::Matrix{Float64}
	MultinomialLogisticModel(in::Int, out::Int) = new(zeros(in, out))
end

# ╔═╡ f451b7b7-86a9-42bf-acb0-4f4f22710485
md"You implemented the softmax function for vectors, here we define it for matrices for you."

# ╔═╡ ce0f1859-cfa8-4c41-a684-55d97b0e2c6e
function softmax(x::AbstractMatrix) 
	exp.(x) ./ (sum(exp.(x), dims=2))
end

# ╔═╡ 8c027447-13e9-4761-bedf-be88f9fd8ae6
# Test Softmax Function
__check_softmax = 
	softmax([1,2,3,4]) ≈ [0.03205860328008499, 0.08714431874203257, 0.23688281808991013, 0.6439142598879722]

# ╔═╡ fd3b8a64-64f6-480a-bc53-110a1c78b1bb
function generate_probabilities(X, rng)

	W = zeros(num_features, 4)
	
	W = rand(rng, Normal(0, 0.5), num_features, 4)
	
	W[1:4, 1] .= 0
	W[4:8, 2] .= 0
	W[8:12, 3] .= 0
	W[12:15, 4] .= 0
	
	Y = X*W
	for r_y in eachrow(Y)
		r_y .= softmax(r_y)
	end
	Y
end

# ╔═╡ f2f88d5d-a939-4ec8-ac8d-fa7fabe12bbc
function generate_labels(X, rng)
	glmY = zeros(num_samples, 4)
	probs = generate_probabilities(X, rng)
	
	for idx in 1:num_samples
		# TODO: glmY[idx, :] = ...
		# Hint: Check the documentation for Distributions.Multinomial

		x = zeros(4)
		x[(idx % 4) + 1] = 1
		glmY[idx, :] = x

	end

	glmY, probs
end

# ╔═╡ 6e29a007-a773-40e8-94b7-8ac60f269b44
glmY, glmProbs = generate_labels(X,  Random.MersenneTwister(3403429));

# ╔═╡ aabe41bc-42f0-4d11-9373-df00d77a9fe8
# we expect at least 3000 of each class to be present
__check_syn_mn_3000 =  all(sum(glmY, dims=1) .> 3000)

# ╔═╡ 95f71de2-e736-41a4-a45f-98a5375a533e
# we expect y = g(x) to sum to 1 for all values of x
__check_syn_mn_probs = all(sum(glmProbs, dims=2) .≈ 1) 

# ╔═╡ 1ef81dfd-fd7b-4c38-9d32-e323fc60527a
# we expect none of y to be a one-hot vector
__check_syn_mn_onehot = all(glmProbs .!= 1)

# ╔═╡ 4b5171e1-0d17-4fd4-8156-92ff24c86d77
# we expect none of the probabilities to be exactly uniform
__check_syn_mn_uniform = let
	!any([all(r .≈ 1/length(r)) for r in eachrow(glmProbs)])
end

# ╔═╡ 5a28e1c0-4d9d-4539-9783-dbe392c39260
begin
	# ! This code block is for testing, feel free to hide it! You don't need to do anything here.
	__check_syn = try
		__check_syn_mn_3000 && __check_syn_mn_probs && __check_syn_mn_onehot && __check_syn_mn_uniform
	catch
		false
	end

	md"""
	## $(_check_complete(__check_syn)) Synthetic Data [10pts]

	Now we will generate a synthetic dataset for multinomial logistic regression (MLM). The only thing that changes is $p(y | \mathbf{x})$, the distribution over $y$. Again, we do not make particular assumptions about the distribution over $\mathbf{x}$. Therefore, we will simply keep the same data matrix `X`.
	And now we have different probabilistic assumptions on our response matrix, `Y`, which we need our synthetic dataset to meet.
	We will be doing 4-class classification, so our response matrix, `Y`, needs to be of 	shape `(samples, 4)`.
	
	This time we will specify most of the data-generating function, but we will get you to help us with part of the implementation. We need your help implementing the softmax and helping us generate labels using our data-generating function. Together, we implement a function `g(x)` that produces a vector ``y \in [0,1]^4`` for each sample `x`, where we ensure that we have class balance. In other words, the dataset has a roughly even number of samples per class, with at least 3000 of each class present. Further, we ensure that `g(x)` induces a valid Multinomial distribution over the 4 classes. Therefore, `y = g(x)` should sum to 1 for each input vector `x`.
	"""
end

# ╔═╡ c6232a18-5f86-4285-a5dc-26c21402c35a
let

	_local_300 = try
		_check_complete(__check_syn_mn_3000)
	catch
		_check_complete(false)
	end

	_local_probs = try
		_check_complete(__check_syn_mn_probs)
	catch
		_check_complete(false)
	end

	_local_onehot = try
		_check_complete(__check_syn_mn_onehot)
	catch
		_check_complete(false)
	end
	
	_local_uniform = try
		_check_complete(__check_syn_mn_uniform)
	catch
		_check_complete(false)
	end
	
	md"""
	Now we need to sample a response matrix `Y` according to the probabilities produced by `g(x)`. We will test a few properties of your resulting response matrix and probability matrix. Again do not change any of these test. The tests are below:
	-  $(_local_300) We expect at least 3000 of each class to be present.
	-  $(_local_probs) We expect y = g(x) to sum to 1 for all values of x.
	-  $(_local_onehot) We expect none of y to be a one-hot vector.
	-  $(_local_uniform) We expect none of the probabilities to be exactly uniform.
	"""
end

# ╔═╡ 1f7f5ba6-4b76-443c-aaa6-38cfb973f0f3
md"Implement the `predict` functions below."

# ╔═╡ 466ea5d6-611c-41db-a0a3-54e7fe77a7f5
function predict(m::MultinomialLogisticModel, x::AbstractVector)

	softmax(transpose(m.W) * x)

end

# ╔═╡ 5859d1fa-9ada-423d-9028-bf3c4bf0df99
function predict(m::MultinomialLogisticModel, X::AbstractMatrix)
	
	softmax(X * m.W)
	
end

# ╔═╡ 35016369-61a6-4e37-a58b-3a2d64038230
begin
	# ! This code block is for testing, feel free to hide it! You don't need to do anything here.
	
	# These tests should always be satisfied, since we provide the solutions
	__check_SLR = try
		test_lm_inputs = 5
		test_lm = LinearModel(test_lm_inputs, 1)
		
		# Test that 0⃗ input results in output value of 0
		z = zeros(test_lm_inputs)
		__test_z = predict(test_lm, z)[1] == 0
		
		# Set the weights to [1, 2, 3, ..., inputs] ./ 5
		# and check that predicting with the linear model results in
		# the sum(1, 2, 3, ..., inputs) / 5
		test_lm.W .= collect(1:test_lm_inputs)
		test_lm.W ./= 5
		one = ones(test_lm_inputs) 
		__test_one = predict(test_lm, one)[1] == sum(1:test_lm_inputs) / 5
		__test_z && __test_one
	catch
		false
	end

	md"""
	### Simple Linear Regression
	
	As mentioned above, this implementation of linear regression (LR) using stochastic gradient descent (SGD) with one sample. It is implemented as follows. For one epoch, the dataset is randomly shuffled, and then iterated over in order, updating for each sample $i$. A fixed scalar stepsize is used, rather than adaptive vector stepsizes.
	
	We opted for a simpler implementation of linear regression, where we do not use mini-batches. This implementation is simple to extend.  
	"""
end

# ╔═╡ edbd2e86-0d5c-41b9-babb-8fcf171fe36c
function update!(lm::LinearModel, sgd::SGD, x::AbstractVector, y::AbstractVector)
	ŷ = predict(lm, x)

	err = ŷ - y
	ΔW = err .* x

	# In Julia, there are two kinds of structs: mutable and immutable. The LinearModel above is immutable (to create a mutable struct, we need to explicitly state that it is mutable). Hence, we cannot change the reference of lm.W to refer to another weight vector, W'. What we can do though, is mutate lm.W, and the immutable LinearModel struct will not complain (indeed, the reference lm.W has not changed, but what is at this reference has changed). If we use the = operator, this will re-assign the lm.W reference to refer to a new object. So lm.W = lm.W .- sgd.α * ΔW is not an allowable operation. If we use the vectorized version (.=), then we will re-assign the elements of the lm.W struct (i.e. mutate the lm.W struct, without changing the reference to lm.W), which is allowable. Hence, the followling line of code works:
	lm.W .-= sgd.α * ΔW
end

# ╔═╡ 7b85d4d2-f405-448b-be9e-b2df4ffbf4bb
# TODO: implement the update func
function update!(lm::RegularizedModel, sgd::SGD, x::AbstractVector, y::AbstractVector)
	ŷ = predict(lm, x)

	lm.W .-= sgd.α .* ((ŷ - y) .* x .+ lm.lambda .* lm.W)

end

# ╔═╡ 99afcfa8-2135-4583-aa96-4501597623d0
# TODO: fill in the update! function
function update!(lm::LinearModel, opt::ADAM, x::AbstractVector, y::AbstractVector)

	ŷ = predict(lm, x)
	
	opt.m = opt.β1 .* opt.m .+ (1 .- opt.β1).*transpose((transpose(ŷ) - y)).*x
	opt.v = opt.β2 * opt.v .+ (1 .- opt.β2).*(transpose((transpose(ŷ) - y)).*x).^2
	
	opt.t += 1
	bias_correct_x = opt.m / (1 - (opt.β1^(opt.t)))
	bias_correct_y = opt.v / (1 - (opt.β2^(opt.t)))

	lm.W -= opt.α .* (bias_correct_x ./(sqrt.(bias_correct_y) .+ opt.ϵ ))

end

# ╔═╡ 6ae6a560-6717-4411-88f1-76d401d8f3b7
__check_mlm_predict_vec, __check_mlm_predict_mat = let
	rng = Random.MersenneTwister(123923)
	test_mlm = MultinomialLogisticModel(10, 3)
	test_mlm.W = rand(rng, size(test_mlm.W)...)
	ŷ = predict(test_mlm, rand(rng, 10))
	check_vec_type = ŷ isa Vector{Float64}
	check_vec_val = ŷ ≈ [0.47951297804699344, 0.2024430588620035, 0.318043963091003]
	
	Ŷ = predict(test_mlm, rand(rng, 8, 10))
	true_Y = [0.28751389731739474 0.45985256788719164 0.2526335347954137; 0.39198552729098973 0.3552167396707249 0.25279773303828534; 0.5297934038030266 0.16935113147454717 0.30085546472242625; 0.3750345657574792 0.41307758215921025 0.21188785208331065; 0.47407101483607245 0.23397266184482654 0.2919563233191011; 0.4501302094851239 0.18872091474452876 0.36114887577034727; 0.45479340522115125 0.17206549881754682 0.373141095961302; 0.4150171839000792 0.34339696953226173 0.24158584656765905]
	
	check_mat_type = Ŷ isa Matrix{Float64}
	check_mat_val = Ŷ ≈ true_Y

	check_vec_type && check_vec_val, check_mat_type && check_mat_val
end

# ╔═╡ 62809e5f-ea92-4634-97f2-0c69ba9c6e4f
md"""
Now we need to implement the `update!` function.
"""

# ╔═╡ 6c876ebe-6cd6-4868-9989-78c908f9186e
# TODO: fill in the update! function
# Hint: think about the shapes of the prediction and label. Are they vectors? Matrices? If so, what shape? How is this different from the LR implementation?

function update!(mlm::MultinomialLogisticModel, opt::ADAM, x::Vector, y::Vector)
	
	ŷ = predict(mlm, x)
	
	opt.m = opt.β1 .* opt.m .+ (1 .- opt.β1).*transpose(ŷ .- y).*x
	opt.v = opt.β2 .* opt.v .+ (1 .- opt.β2).*((transpose(ŷ .- y)).*x).^2
	
	opt.t += 1
	bias_correct_m = opt.m ./ (1 .- (opt.β1.^(opt.t)))
	bias_correct_v = opt.v ./ (1 .- (opt.β2.^(opt.t)))

	mlm.W .-= opt.α .* (bias_correct_m ./(sqrt.(bias_correct_v) .+ opt.ϵ ))


end


# ╔═╡ 7f8a21fd-9155-4c8a-b14a-bccf36b93bbe
function train!(m::Model, sgd::Optimizer, X::AbstractMatrix, Y::AbstractMatrix)
	# Random.seed!(1)
	rng = Random.MersenneTwister(1)
	for idx in randperm(rng, size(Y, 1))
		# grab a row of X
		# note that vectors in Julia are *always* column vectors by default
		# so this will give us a column vector x = X[idx, :]
		update!(m, sgd, X[idx, :], Y[idx, :])
	end
end

# ╔═╡ 7cf2245c-c876-4425-b634-41601ed19ddf
# use a function to avoid cluttering global scope with tmp variables
function runLinearModel(X::AbstractMatrix, Y::AbstractMatrix)
	# construct a model
	sgd = SGD(0.01)

	model = LinearModel(size(X, 2), 1)

	# Each call to train! performs a single epoch
	for _ in 1:NUM_EPOCHS
		train!(model, sgd, X, Y)
	end

	model
end

# ╔═╡ 4747aeec-89a2-415c-ac8e-1f6177ca3532
__linear_model = runLinearModel(trainX, trainY);

# ╔═╡ 141ce0e2-b2c3-4769-a67a-3c58311cd85d
RMSE(predict(__linear_model, testX), testY)

# ╔═╡ 4eca71f9-9342-4ed5-9ca8-ad3e39384d67
function runRLRModel(X::AbstractMatrix, Y::AbstractMatrix)
	# train
	sgd = SGD(0.01)

	model = RegularizedModel(size(X, 2), 1, 1.0)

	for _ in 1:NUM_EPOCHS
		train!(model, sgd, X, Y)
	end

	model
end

# ╔═╡ 8f164c34-0a80-43b3-92bf-7d5d0413a901
__rlr_model = runRLRModel(trainX, trainY);

# ╔═╡ 7036ad67-4987-46fd-a9d5-f1ad2760f3f3
RMSE(predict(__rlr_model, testX), testY)

# ╔═╡ 71146214-82b8-4da0-ae7e-985125cb80e8
_check_understanding_1 = 
	RMSE(predict(__linear_model, testX), testY) > RMSE(predict(__rlr_model, testX), testY)

# ╔═╡ 5f356bf4-69b0-4016-96fd-12096e5a02bb
let
	lr_error = try
		RMSE(predict(__linear_model, testX), testY)
	catch
		"something went wrong 🤔"
	end
	rr_error = try
		RMSE(predict(__rlr_model, testX), testY)
	catch
		"something went wrong 🤔"
	end

	local_check = try
		_check_understanding_1
	catch
		false
	end
	
	md"""
	## $(_check_complete(local_check)) Checkpoint: RLR better than LR?
	
	Now we get to revisit our synthetic data generator and ask: what data is more suitable for the assumptions underlying L2 regularized linear regression? 
	
	Go back to your simulated data generator and modify $f(X)$ so that RLR regression finds a better approximation than vanilla linear regression. Specifically, define a function $f(x)$ such that
	```
	RMSE(predict(lm, testX), testY) > RMSE(predict(rm, testX), testY)
	```

	
	The current values:
	- Linear Regression: $(lr_error)
	- Regularized Linear Regression: $(rr_error)
	"""
end

# ╔═╡ 92408523-8760-4aa5-8d49-d8ba28bc73e3
function runLinearModelWithAdam(X::AbstractMatrix, Y::AbstractMatrix)
	# passing arguments as (stepsize, \beta_1, \beta_2, features, outputs)
	adam = ADAM(0.01, 0.9, 0.999, size(X, 2), 1)

	model = LinearModel(size(X, 2), 1)

	for _ in 1:NUM_EPOCHS
		train!(model, adam, X, Y)
	end

	model
end

# ╔═╡ 034efe82-1bbf-406e-b45f-b93fab2ce121
__linear_adam_model = runLinearModelWithAdam(trainX, trainY);

# ╔═╡ efa6376b-481e-4eba-af38-f3868302911b
__check_adam_learn =
	isclose(RMSE(predict(__linear_model, testX), testY),
			RMSE(predict(__linear_adam_model, testX), testY), 1.0)

# ╔═╡ 33bac267-402a-4231-9456-6939c9cdd3de
let
	lm_error = try 
		RMSE(predict(__linear_model, testX), testY)
	catch
		"something went wrong 🤔"
	end

	lm_adam_error = try
		RMSE(predict(__linear_adam_model, testX), testY)
	catch
		"something went wrong 🤔"
	end

	_local_check = try
		_check_complete(__check_adam_learn)
	catch
		_check_complete(false)
	end
	
	
	md"""
	## $(_local_check) Checkpoint: Learning with Adam
	
	It is useful to know how to implement ADAM.  it likely will not perform better than SGD, because of how our synthetic data is generated. The inputs are reasonably scaled, so we are unlikely to have very different curvature in different dimensions. But, let's run it anyway and check! 
	
	Check below how well this optimization strategy is doing as compared to SGD:
	
	- SGD: $(lm_error)
	- Adam: $(lm_adam_error)
	
	See below for the learning code!
	
	"""
end

# ╔═╡ 3bb208bf-b75c-49a2-8c76-f41be26fb281
begin
	# Perform a few gradient steps and see if the weights are updated correctly


	__check_RR = try
		test_rlr_inputs = 3
		test_rlr_tolerance = 1e-5
		test_rlr = RegularizedModel(test_rlr_inputs, 1, 0.1)
		test_rlr_sgd = SGD(0.01)
		
		rlr_x = collect(1:test_rlr_inputs)
		rlr_y = [1]
		update!(test_rlr, test_rlr_sgd, rlr_x, rlr_y)
	
		rlr_x = (collect(1:test_rlr_inputs) .+ 1) ./ 2
		rlr_y = [2]
		update!(test_rlr, test_rlr_sgd, rlr_x, rlr_y)
	
		rlr_x = (collect(1:test_rlr_inputs) .+ 2) ./ 3
		rlr_y = [3]
		update!(test_rlr, test_rlr_sgd, rlr_x, rlr_y)
		
		train_test = isapprox(
			test_rlr.W,
			[0.056891877, 0.085672676, 0.114453474];
			atol = test_rlr_tolerance,
		)
		
		ŷ = predict(test_rlr, collect(-test_rlr_inputs:-1))
		predict_test = isapprox(
			ŷ,
			[-0.45647446];
			atol = test_rlr_tolerance,
		)
		train_test && predict_test
	catch
		false
	end
	
	md"""
	### $(_check_complete(__check_RR)) Regularized LR [15pts]
	"""
end

# ╔═╡ 420f3d89-a494-4546-9879-9b6db16f44fe
begin 
	# ! This code block is for testing, feel free to hide it! You don't need to do anything here.
	__check_LR = try
		__check_SLR && __check_RR && __check_syn_data 
	catch
		false
	end
	md"""
	# $(_check_complete(__check_LR)) Linear Regression (warm up)
	"""
end

# ╔═╡ ec300f7e-0d83-48ea-8cf8-55adb31780e2
begin
	__check_linear_models = try
		__check_SLR && __check_RR
	catch
		false
	end
	
	md"""
	## $(_check_complete(__check_linear_models)) Linear Models
	
	Each model must have a consistent API.
	We start with a struct which is a subtype of `Model`.
	Each subtype of `Model` must provide a `predict` function, which takes as input an instance of the model and a data matrix `X`, then returns a vector of predictions `y`.
	To update the weights of a model instance, you will need to implement the `update!` function, which takes the model instance, an optimizer, a data matrix of inputs, and a data vector of responses.
	Note the exclamation mark: the `update!` function will _mutate_ the weights belonging to the model instance.
	"""
end

# ╔═╡ a0c9c035-0fda-481d-815a-654e2d3a1a76
# Test cell for the update function for l2 regularized linear regression. Don't change this cell.
__check_RR_update = let
	test_rlr_inputs = 3
	test_rlr_tolerance = 1e-5
	test_rlr = RegularizedModel(test_rlr_inputs, 1, 0.1)
	test_rlr_sgd = SGD(0.01)
	
	rlr_x = collect(1:test_rlr_inputs)
	rlr_y = [1]
	update!(test_rlr, test_rlr_sgd, rlr_x, rlr_y)

	rlr_x = (collect(1:test_rlr_inputs) .+ 1) ./ 2
	rlr_y = [2]
	update!(test_rlr, test_rlr_sgd, rlr_x, rlr_y)

	rlr_x = (collect(1:test_rlr_inputs) .+ 2) ./ 3
	rlr_y = [3]
	update!(test_rlr, test_rlr_sgd, rlr_x, rlr_y)
	
	the_test = isapprox(
		test_rlr.W,
		[0.056891877, 0.085672676, 0.114453474];
		atol = test_rlr_tolerance,
	)
end

# ╔═╡ 87225565-d57c-4829-9e22-aabcdd59e571
begin
	# ! This code block is for testing, feel free to hide it! You don't need to do anything here.
	
	# Perform a few gradient steps and see if the weights are updated correctly

	__check_adam = let
		test_adam_inputs = 3
		test_adam_tolerance = 1e-5
		test_adam_model = LinearModel(test_adam_inputs, 1)
		test_adam = ADAM(0.01, 0.9, 0.999, test_adam_inputs, 1)
		
		test_adam_x = collect(1:test_adam_inputs)
		test_adam_y = [1]
		update!(test_adam_model, test_adam, test_adam_x, test_adam_y)
	
		test_adam_x = (collect(1:test_adam_inputs) .+ 1) ./ 2
		test_adam_y = [-2]
		update!(test_adam_model, test_adam, test_adam_x, test_adam_y)
	
		test_adam_x = (collect(1:test_adam_inputs) .+ 2) ./ 3
		test_adam_y = [3]
		update!(test_adam_model, test_adam, test_adam_x, test_adam_y)
		
		ŷ = predict(test_adam_model, collect(-test_adam_inputs:-1))
		predict_test = isapprox(
			ŷ,
			[-0.061532371355245453];
			atol = test_adam_tolerance)
		update_test = isapprox(
			test_adam_model.W, 
			[0.009569011794190506, 0.010764570223018575, 0.011296195526636793];
			atol = test_adam_tolerance,
		)
		predict_test && update_test
	end
	
	md"""
	# $(_check_complete(__check_adam)) ADAM Optimizer [15pts]
	"""
end

# ╔═╡ 4b45defb-85c8-4eec-8fb1-8b63597d23e1
# Test cell for the update function for regularize linear regression. Don't change this cell.
__check_adam_update = let
	test_adam_inputs = 3
	test_adam_tolerance = 1e-5
	test_adam_model = LinearModel(test_adam_inputs, 1)
	test_adam = ADAM(0.01, 0.9, 0.999, test_adam_inputs, 1)
	
	test_adam_x = collect(1:test_adam_inputs)
	test_adam_y = [1]
	update!(test_adam_model, test_adam, test_adam_x, test_adam_y)

	test_adam_x = (collect(1:test_adam_inputs) .+ 1) ./ 2
	test_adam_y = [-2]
	update!(test_adam_model, test_adam, test_adam_x, test_adam_y)

	test_adam_x = (collect(1:test_adam_inputs) .+ 2) ./ 3
	test_adam_y = [3]
	update!(test_adam_model, test_adam, test_adam_x, test_adam_y)
	
	isapprox(
		test_adam_model.W, 
		[0.009569011794190506, 0.010764570223018575, 0.011296195526636793];
		atol = test_adam_tolerance,
	)
end

# ╔═╡ e527f668-65bc-4414-97d4-7f74fdc62e0a
md"
## $(_check_complete(__check_adam_update)) ADAM Update

Implement the ADAM update in update! below. 

We have also provuded a test cell that you can use to test your code. Once again, do not change the code in this test cell as doing so will result in point reductions.
"

# ╔═╡ ac1e9949-9cc8-43a0-93dc-822fa75d304c
# Test MLM Update
__check_mlm_update = let
	test_mlm_inputs = 3
	test_mlm_outputs = 4
	test_mlm_tolerance = 1e-5  # Absolute tolerance to consider arrays equal
	test_mlm = MultinomialLogisticModel(test_mlm_inputs, test_mlm_outputs)
	test_mlm_adam = ADAM(0.1, 0.9, 0.999, test_mlm_inputs, test_mlm_outputs)

	# Update 1
	test_mlm_x = (collect(1:test_mlm_inputs) .- (test_mlm_inputs ÷ 2)) ./ 2
	test_mlm_y = zeros(test_mlm_outputs)
	test_mlm_y[test_mlm_outputs ÷ 2] = 1
	update!(test_mlm, test_mlm_adam, test_mlm_x, test_mlm_y)

	# Update 2
	test_mlm_x = collect(1:test_mlm_inputs) ./ 5
	test_mlm_y = zeros(test_mlm_outputs)
	test_mlm_y[end] = 1
	update!(test_mlm, test_mlm_adam, test_mlm_x, test_mlm_y)

	# Update 3
	test_mlm_x = -collect(1:test_mlm_inputs)
	test_mlm_y = zeros(test_mlm_outputs)
	test_mlm_y[begin] = 1
	update!(test_mlm, test_mlm_adam, test_mlm_x, test_mlm_y)

	expected = [-0.14249883639454075 -0.062484407566620465 -0.01810743755431654 0.16037612895458997; -0.27128430525463537 0.19392424158692567 -0.1510845274530498 0.012653251099602032; -0.2694032781383666 0.20126340307296198 -0.15097164658240314 -0.003880906748877491]
	
	isapprox(
		test_mlm.W,
		expected,
		atol = test_mlm_tolerance,
	)
end

# ╔═╡ abf5ae16-e0e3-4eda-9fa3-20d20ec39086
begin 
	# ! This code block is for testing, feel free to hide it! You don't need to do anything here.
	
	__check_mlm = try
		__check_syn && 
		__check_mlm_update && 
		__check_mlm_predict_vec && 
		__check_mlm_predict_mat
	catch
		false
	end
	md"""
	# $(_check_complete(__check_mlm)) Multinomial Logistic Regression
	"""
end

# ╔═╡ 5d27da0d-a5fa-4725-a1a7-911a49d22b63
let 
	# ! This code block is for testing, feel free to hide it! You don't need to do anything here.

	_local_check = try
		_check_complete(__check_mlm_update && 
						__check_mlm_predict_vec && 
						__check_mlm_predict_mat)
	catch
		_check_complete(false)
	end
	
	md"""
	##  $(_local_check) Estimator [30pts]
	"""
end

# ╔═╡ 23c6e59b-02f1-4170-85fa-9ed06cfc61aa
begin
	# define the train and test labels as before.
	glm_trainY = glmY[1:25000, :]
	glm_testY = glmY[25001:end, :]
end;

# ╔═╡ 6f053746-648c-49a8-bee2-dd0ee02ccfee
function runMLM(X::AbstractMatrix, Y::AbstractMatrix)
	adam = ADAM(0.001, 0.9, 0.999, size(X, 2), 4)

	model = MultinomialLogisticModel(size(X, 2), 4)

	for _ in 1:NUM_EPOCHS
		train!(model, adam, X, Y)
	end

	model
end

# ╔═╡ d1902176-e4ee-4c81-a2db-f68837f8f717
__mlm_model = runMLM(trainX, glm_trainY);

# ╔═╡ e884510d-36b6-49dc-ab89-335012133ecc
RMSE(predict(__mlm_model, testX), glm_testY)

# ╔═╡ 0bb94dff-b436-499a-91af-d51665ff107b
md"
Now let's run LR.
"

# ╔═╡ 92916419-3e63-422d-890e-2dce5cc37c6c
function runLRForClassification(X::AbstractMatrix, Y::AbstractMatrix)
	adam = ADAM(0.001, 0.9, 0.999, size(X, 2), 4)

	model = LinearModel(size(X, 2), 4)

	for _ in 1:NUM_EPOCHS
		train!(model, adam, X, Y)
	end

	model
end

# ╔═╡ 7f0ed8c5-b2f4-4571-9898-5487a159efd0
__linear_model_class = runLRForClassification(trainX, glm_trainY);

# ╔═╡ 6f3ec55a-754f-4b8b-8b79-f5640f0529bc
RMSE(predict(__linear_model_class, testX), glm_testY)

# ╔═╡ a2faef80-7d6f-41ae-a36a-364f0160fd75
function classification_error(ŷ, y)
	_, max_ŷ = findmax(ŷ, dims=2)
	_, max_y = findmax(y, dims=2)

	1 - (sum(max_ŷ .== max_y) / size(y, 1))
end

# ╔═╡ 7ce27171-c7f1-4aa3-8060-eb25bb6673b4
let
	lm_cerror = try
		classification_error(predict(__linear_model_class, testX), glm_testY)
	catch
		"something went wrong 🤔"
	end

	mlm_cerror = try
		classification_error(predict(__mlm_model, testX), glm_testY)
	catch
		"something went wrong 🤔"
	end
	
	md"""Below we check the classification error using the function `classification_error`. The error for the models are as follows:
	- `LinearModel`: $(lm_cerror)
	- `MultinomialLogisticModel`: $(mlm_cerror)
	"""
end

# ╔═╡ bd3d2b56-3c65-4f47-ad11-a4a17785a91f
__check_classification_error = let
	lm_cerror = classification_error(
		predict(__linear_model_class, testX), glm_testY)
	mlm_cerror = classification_error(predict(__mlm_model, testX), glm_testY) 
	lm_cerror > mlm_cerror
end

# ╔═╡ d502c3cc-a7c2-4fdc-9d02-96eef70e30b6
let
	_local_check = try
		_check_complete(__check_classification_error)
	catch
		_check_complete(false)
	end
	md"""
	##  $(_local_check) Checkpoint: Classification
	
	First we learn a `MultinomialLogisticModel` and then an LR model and test. We should see the `MultinomialLogisticModel` should perform better!

	We use LR in the most naive way: we predict the target vector. This means LR is trying to predict 0s and 1. The output from LR is not guaranteed to be an indicator vector; in fact, its unlikely to be. We return the class with the highest output from LR. For example, if LR outputs [1.3, -0.5, 0.1, -4.2], then we say that LR has classified this input as class 1, since 1.3 is the highest. 
	"""
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CSV = "~0.10.4"
Distributions = "~0.25.70"
PlotlyJS = "~0.18.8"
PlutoUI = "~0.7.40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "3fc88ac187d17cfb0c555b7c7e18afb084e8d3fe"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[deps.Blink]]
deps = ["Base64", "BinDeps", "Distributed", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Reexport", "Sockets", "WebIO", "WebSockets"]
git-tree-sha1 = "08d0b679fd7caa49e2bca9214b131289e19808c0"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.5"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "8579b5cdae93e55c0cff50fbb0c2d1220efd5beb"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.70"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "316daa94fad0b7a008ebd573e002efd6609d85ac"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.19"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "3399bbad4c9e9a2fd372a54d7b67b3c7121b6402"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.3"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "ae6676d5f576ccd21b6789c2cbe2ba24fcc8075d"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.5"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "1e566ae913a57d0062ff1af54d2697b9344b99cd"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.14"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "Pkg", "Sockets", "WebSockets"]
git-tree-sha1 = "82dfb2cead9895e10ee1b0ca37a01088456c4364"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "0.7.6"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "53d6325e14d3bdb85fd387a085075f36082f35a3"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.8"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "a602d7b0babfca89005da04d89223b867b55319f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.40"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "a8bbcd0b08061bba794c56fb78426e96e114ae7f"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.18"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─8c52dc53-c488-48da-86e0-18c8625579c8
# ╟─84c9b59f-377e-405f-836e-fcd287e964d2
# ╟─b8504526-9e4f-4e12-a900-99081d2c0641
# ╠═4731f58f-1697-40ee-9cf9-fd31311c1bf5
# ╠═e1909389-766f-4fad-b1b8-1efd4fb3685e
# ╠═6d5b815d-fb8f-4dc7-9355-83f2e4e30c92
# ╠═efd24c67-8ed0-45bc-ba63-fc3ec53ff840
# ╠═777edfdf-7163-4b40-8dfe-cc55e741423f
# ╠═fb422f47-6c47-4a7e-b044-26ca14574cad
# ╟─420f3d89-a494-4546-9879-9b6db16f44fe
# ╟─eb8b1686-f7f4-4db1-b944-e893e0f3e9e5
# ╟─cce3a5a5-9e37-4ca8-a9bc-8ad1de773d33
# ╠═1b90c1ff-1585-4c75-9961-6a411ad37531
# ╠═96fa5d34-8a99-4be4-9917-78f79b53b490
# ╟─0c8e7b09-e202-412c-8e5d-f963a77820a8
# ╠═8d6eb957-2833-4cae-8e78-f0ac2f562952
# ╠═78447784-2d46-4b25-9654-59e9e962b255
# ╟─a16d8fff-4b50-4d56-9acb-7b96e0367d15
# ╠═9d037c81-61bb-46da-89b1-384672128a47
# ╠═e39fbb78-a37f-4da4-9edf-31562387d20e
# ╠═60f2b901-6fcd-43d9-b0f1-4f9e802e87e9
# ╟─32e4a3ac-1864-4b1c-9b76-670c529b9d67
# ╠═8bbc7546-46d6-4c90-993d-823e420373b6
# ╟─edefd098-d2d5-46c1-91e4-2529267b54f1
# ╠═d86f970a-7dd2-4e5e-b44a-4c966a62a536
# ╟─ec300f7e-0d83-48ea-8cf8-55adb31780e2
# ╠═52502f3b-8a7c-41b6-8405-35a889964876
# ╠═17e1b66c-b5bd-4f6a-8682-c5f7fc686df3
# ╠═f9fe90fb-7a52-4769-866f-f548bb77d2bb
# ╠═04669dcd-7973-4e82-9e6f-ac155106a2b0
# ╟─35016369-61a6-4e37-a58b-3a2d64038230
# ╠═75246b41-a7c1-4e3b-8ee0-89d733a22a15
# ╠═b66c2fc2-b3be-415b-88e0-a8415d2380a4
# ╠═edbd2e86-0d5c-41b9-babb-8fcf171fe36c
# ╠═7f8a21fd-9155-4c8a-b14a-bccf36b93bbe
# ╠═6ca67862-ecd6-4996-82be-d2caf9ea0445
# ╠═ec44beaf-41b5-4e00-9f5e-2a17a7ff6606
# ╠═7cf2245c-c876-4425-b634-41601ed19ddf
# ╠═4747aeec-89a2-415c-ac8e-1f6177ca3532
# ╠═141ce0e2-b2c3-4769-a67a-3c58311cd85d
# ╟─3bb208bf-b75c-49a2-8c76-f41be26fb281
# ╟─cf388af0-3d04-41a2-b009-245356c78e63
# ╠═1e9a6ebc-8e79-47cd-ad82-de114232f97a
# ╟─a812b5a7-3c5e-4688-b38f-76f8d096ca03
# ╠═78b77bc9-2aab-4a7a-b272-ef03953fc4e6
# ╠═f74938f8-00f0-4c9e-aeda-9f02ffb81a7d
# ╟─8513b104-5577-4831-b3b5-2eb61021f11f
# ╠═7b85d4d2-f405-448b-be9e-b2df4ffbf4bb
# ╠═a0c9c035-0fda-481d-815a-654e2d3a1a76
# ╠═adccdd65-c7aa-4b5b-9c7e-f17bd3232485
# ╠═4eca71f9-9342-4ed5-9ca8-ad3e39384d67
# ╠═8f164c34-0a80-43b3-92bf-7d5d0413a901
# ╠═7036ad67-4987-46fd-a9d5-f1ad2760f3f3
# ╠═5f356bf4-69b0-4016-96fd-12096e5a02bb
# ╠═71146214-82b8-4da0-ae7e-985125cb80e8
# ╟─87225565-d57c-4829-9e22-aabcdd59e571
# ╟─b89f82d7-0298-4a8f-8163-04ce4533f9b8
# ╟─9b13b44c-9cf5-456b-b77a-a0969d28d5a2
# ╠═661573b8-f465-48b8-bd1d-f2c720fd593b
# ╟─75051421-8e37-491e-8e48-dda77363364e
# ╟─e527f668-65bc-4414-97d4-7f74fdc62e0a
# ╠═99afcfa8-2135-4583-aa96-4501597623d0
# ╠═4b45defb-85c8-4eec-8fb1-8b63597d23e1
# ╠═33bac267-402a-4231-9456-6939c9cdd3de
# ╠═efa6376b-481e-4eba-af38-f3868302911b
# ╠═92408523-8760-4aa5-8d49-d8ba28bc73e3
# ╠═034efe82-1bbf-406e-b45f-b93fab2ce121
# ╟─abf5ae16-e0e3-4eda-9fa3-20d20ec39086
# ╟─b9c5a48b-6949-4ce2-abb6-f0669a88bd9f
# ╟─5a28e1c0-4d9d-4539-9783-dbe392c39260
# ╟─3fa4d186-b207-45ec-868a-84b6a5187aae
# ╠═653db006-f1e7-4f43-9f44-5a5681d030d8
# ╠═8c027447-13e9-4761-bedf-be88f9fd8ae6
# ╠═e1c7b036-3fb8-4708-a8b4-edf8a0a46c27
# ╠═fd3b8a64-64f6-480a-bc53-110a1c78b1bb
# ╟─29cdfb84-d10f-4d3c-a325-c2d445a7afbf
# ╠═f2f88d5d-a939-4ec8-ac8d-fa7fabe12bbc
# ╟─c6232a18-5f86-4285-a5dc-26c21402c35a
# ╠═6e29a007-a773-40e8-94b7-8ac60f269b44
# ╠═aabe41bc-42f0-4d11-9373-df00d77a9fe8
# ╠═95f71de2-e736-41a4-a45f-98a5375a533e
# ╠═1ef81dfd-fd7b-4c38-9d32-e323fc60527a
# ╠═4b5171e1-0d17-4fd4-8156-92ff24c86d77
# ╟─5d27da0d-a5fa-4725-a1a7-911a49d22b63
# ╟─540cae75-db31-4d4d-b853-1031bc5bf2a8
# ╠═92f32acf-9ad8-438a-b289-d46e1f97b17e
# ╟─f451b7b7-86a9-42bf-acb0-4f4f22710485
# ╠═ce0f1859-cfa8-4c41-a684-55d97b0e2c6e
# ╟─1f7f5ba6-4b76-443c-aaa6-38cfb973f0f3
# ╠═466ea5d6-611c-41db-a0a3-54e7fe77a7f5
# ╠═5859d1fa-9ada-423d-9028-bf3c4bf0df99
# ╠═6ae6a560-6717-4411-88f1-76d401d8f3b7
# ╟─62809e5f-ea92-4634-97f2-0c69ba9c6e4f
# ╠═6c876ebe-6cd6-4868-9989-78c908f9186e
# ╠═ac1e9949-9cc8-43a0-93dc-822fa75d304c
# ╟─d502c3cc-a7c2-4fdc-9d02-96eef70e30b6
# ╠═23c6e59b-02f1-4170-85fa-9ed06cfc61aa
# ╠═6f053746-648c-49a8-bee2-dd0ee02ccfee
# ╠═d1902176-e4ee-4c81-a2db-f68837f8f717
# ╠═e884510d-36b6-49dc-ab89-335012133ecc
# ╟─0bb94dff-b436-499a-91af-d51665ff107b
# ╠═92916419-3e63-422d-890e-2dce5cc37c6c
# ╠═7f0ed8c5-b2f4-4571-9898-5487a159efd0
# ╠═6f3ec55a-754f-4b8b-8b79-f5640f0529bc
# ╟─7ce27171-c7f1-4aa3-8060-eb25bb6673b4
# ╠═a2faef80-7d6f-41ae-a36a-364f0160fd75
# ╠═bd3d2b56-3c65-4f47-ad11-a4a17785a91f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
