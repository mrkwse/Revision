# MLAP Things

## Lecture 8 - Regularization

### $p=0$: Subset selection

* For $p=0$: simply counts number of non-zero features
* Prefer models with fewer features
    * i.e., _simpler_ models with _fewer_ parameters
* Function not convex

N.B., $0^0 = 0$, while $x^0 = 1$ for $x > 0$.

#### Greedy Subset selection

* Input $(X,Y)$ and $K =$ no. features to select.
* Initially all parameters set to $0$, i.e., $\theta = 0$.
* Repeat until $K$ features chosen:
    * Select feature $F$ that gives greatest reduction in error.
    * Apply 1 gradient descent step along $F$ (leaving other features unmodified)
* When $K$ features have been selected:
    * Apply gradient steps only


> * Approximation for $p=0$ case.
> * Requires deciding $K$ in advance.
> * Not necessarily disadvantage, can find $K$ via other methods (Cross validation)
> * Greedy method, can get stuck in locally optimal K

### $p=1$ : LASSO Regularized Regression

_'L1 Regularization'_

* Smooth penalty over values of $\theta_j$.
* Unit diamond
* Generally an equilateral polytope

$$
L(\theta) = \lambda \left(\sum_i \mathbf{loss}(y_i, f(x_i))\right) + (1 - \lambda)\sum_j |\theta_j |^p
$$

$$
|\theta| = \left(\theta^2\right)^\frac{1}{2}
$$

$$
\frac{d}{d\theta}|\theta| = \frac{d}{d\theta}\left(\theta^2\right)^\frac{1}{2} =\frac{1}{2}\left(\theta^2\right)^{-\frac{1}{2}}.2\theta = \frac{\theta}{|\theta|} = \mathbf{sign}(\theta)
$$

$$
\mathbf{sign}(\theta) =
    \begin{cases}
        -1 \quad &\text{if} &\; \theta < 0 \\
        1 \quad &\text{if} &\; \theta > 0 \\
        0 \quad &\text{if} &\; \theta = 0
    \end{cases}
$$

#### Applying Lasso to Linear Regression

$$
L(\theta) = \lambda \left(\sum_i (y_i - \theta, X_i)^2\right) + (1 - \lambda)\sum_j |\theta_j |
$$

$$
\frac{\delta}{\delta \theta_i}\sum_j |\theta_j| = \mathbf{sign}(\theta_i)
$$

$$
\Delta L(\theta) = -2 \lambda\left[\sum_i X_i . (y_i - \theta.X_i)\right] + (1 - \lambda)\mathbf{sign}(\theta)
$$

Matrix:

$$
\Delta L(\theta) = -2 \lambda X^T (Y - X\theta) + (1 - \lambda)\mathbf{sign}(\theta)
$$

Solving
* Does not admit a closed form solution
* Can be solved efficiently via _special purpose_ algorithms
* Can be solved using _general purpose_ constrained optimisation methods
* Simple gradient based search not guaranteed to work

### $p=2$ : Ridge Regularized Regression

_'L2 regularization'_

* Smooth penalty over values of $\theta_j$

$$
\begin{align}
    L(\theta) &= \lambda \left( \sum_i (y_i - \theta.X_i)^2 \right) + (1 - \lambda) \sum_j |\theta_j|^2 \\
    \Delta L(\theta) &= -2\lambda \sum_i X_i . (y_i - \theta.X_i) + 2(1-\lambda) \theta
\end{align}
$$

Matrix:

$$
\Delta L(\theta) = -2 \lambda X^T (Y - X\theta) + 2(1-\lambda)\theta
$$

Closed form solution (derivative set to $0$):

$$
\theta = [\lambda X^T X + (1 - \lambda)I\,]^{-1} \lambda X^T Y
$$

Ridge regression can be solved via:
* Closed form solution (may be too slow)
* Gradient based methods
* Extended least squares solvers
* Constrained optimisation methods.

### Ridge vs Lasso

* Ridge tends to shrink all features equally
* Lasso can often set features to zero
* Lasso thought of as a soft version of subset selection

## Lecture 9 - Bayesian Interpretation of Regularisation

### Bayesian Interpretation of Linear Regression

$$
\begin{align}
    p(y_i | X_i, \theta, \sigma^2) &= \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-X\theta)^2}{2\sigma^2}} \\
    p(Y|X,\theta,\sigma^2) &= \prod_i p(y_i | X_i, \theta, \sigma^2) \\
    &= \prod_i \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-X\theta)^2}{2\sigma^2}} \\
\end{align}
$$

### Adding Priors

$$
\begin{align}
    p(\theta|Y,X,\sigma^2)p(Y|X, \theta^2) &= p(Y,\theta|X,\sigma^2) \\
    &= p(Y,\theta|X,\sigma^2)p(\theta|X, \sigma^2) \\
    &= p(Y,\theta|X,\sigma^2)p(\theta) \\
\end{align}
$$

With:

$$
p(Y|X,\sigma^2) = \int p(Y,\theta|X,\sigma^2)d\theta
$$

Hence posterior is:

$$
p(\theta|Y,X,\sigma^2) = \frac{p(Y|X, \theta, \sigma^2)p(\theta)}{\int p(Y,\theta|X,\sigma^2)d\theta}
$$


### MLE for Linear Regression

$$
\begin{align}
    \theta_{\mathbf{MLE}} &= \underset{\theta}{\arg\max} p(\theta|Y,X,\sigma^2) \\
    &\propto \underset{\theta}{\arg\max} p(Y|X,\theta, \sigma^2) \\
    &\propto \underset{\theta}{\arg\max} \prod_i \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-X_i\theta)^2}{2\sigma^2}} \\
    &\propto \underset{\theta}{\arg\max} \sum_i - \log\left(\sqrt{2\pi\sigma^2}\right) -\frac{(y_i-X_i\theta)^2}{2\sigma^2} \\
    &\propto \underset{\theta}{\arg\min} \sum_i (y_i - X_i\theta)^2
\end{align}
$$

### Regularization as MAP estimation

$$
\begin{align}
    \theta_{\mathbf{MAP}} &= \underset{\theta}{\arg\max} p(\theta|Y,X,\sigma^2) \\
    &\propto \underset{\theta}{\arg\max} p(Y|X,\theta, \sigma^2)p(\theta) \\
    &\propto \underset{\theta}{\arg\max} \left[\prod_i p(Y|X,\theta, \sigma^2)\right]e^{-\frac{(1-\lambda)}{\lambda}\sum_i|\theta_i|^P} \\
    &\propto \underset{\theta}{\arg\max} \log \left(\prod_i p(Y|X,\theta, \sigma^2)\right)-\frac{(1-\lambda)}{\lambda}\sum_i|\theta_i|^P \\
    &\propto \underset{\theta}{\arg\max} L(\theta, \sigma^2) - \frac{(1-\lambda)}{\lambda}\sum_i |\theta_i|^P \\
    &\propto \underset{\theta}{\arg\max} \sum_i - \log\left(\sqrt{2\pi\sigma^2}\right) -\frac{(y_i-X_i\theta)^2}{2\sigma^2} - \frac{(1 - \lambda)}{\lambda}\sum_i |\theta_i|^P\\
    &\propto \underset{\theta}{\arg\min} \sum_i (y_i - X_i\theta)^2 + \frac{(1 - \lambda)}{\lambda}\sum_i |\theta_i|^P \\
    &\propto \underset{\theta}{\arg\min} \sum_i \lambda(y_i - X_i\theta)^2 + (1 - \lambda)\sum_i |\theta_i|^P
\end{align}
$$

## Lecture 10 - Evaluation and Cross Validation

### Mean error

$$
\frac{1}{N} \sum_i \mathbf{error}(y_i, f(x_i))
$$

### Mean Squared Error (MSE)

$$
\frac{1}{N} \sum_{\langle X,y \rangle \in \text{TestData}} \mathbf{error}(y - f(X))^2
$$

Square root of MSE, Root Mean Squared Error (RMSE) is common.

### Hard Classification Accuracy

Hard classifier formulates error measure for a given test instance $\langle X,y \rangle$ as:

$$
I(c^{\mathbf{hard}}(X,\theta) = y)
$$

Average hard classification accuracy in test set:

$$
\frac{1}{N} \sum_{\langle y,X \rangle \in \text{TestData}} I(c^{\mathbf{hard}}(X,\theta) = y)
$$

### Soft Classification Accuracy

Average soft classification accuracy in Test Set:

$$
\frac{1}{N} \sum_{\langle X,c \rangle \in \text{TestData}} p(y=c|X,\theta)
$$

### Classification Error

Average __soft__ classification error in test set:

$$
\frac{1}{N} \sum_{\langle X,c \rangle \in \text{TestData}} ( 1- p(y=c|X,\theta))
$$

Average __hard__ classification error in test set:

$$
\frac{1}{N} \sum_{\langle y,X \rangle \in \text{TestData}} (1 - I(c^{\mathbf{hard}}(X,\theta) = y))
$$

> * Accuracy prefers high values
> * Error prefers low values

### Fitting Regularization Parameter $\lambda$

* Divide training data into actual training data, cross validation data, and test data.
* Error on cross validation data used to find best $\lambda$
* With training and cros validation data:
    * Train linear regression function or logistic regression classifier on training data for different values of $\lambda$
    * Calculate error (MSE, hard/soft classification error)
    * Choose $\lambda$ with minimum error on cross validation data

#### k-fold cross validation error

$$
\mathbf{errorCV}(f = \langle f^{-1}, \dots, f^{-k} \rangle, \lambda) = \frac{1}{N} \sum_i \mathbf{error}\left(y_i, f^{-f\,\mathbf{old}(i)}(X_i, \lambda)\right)
$$

Optimal $\lambda$ given by:
$$
\lambda^* = \mathbf{errorCV}(f = \langle f^{-1}, \dots, f^{-k} \rangle, \lambda)
$$

### CV to tune $\lambda$

1. Divide training data in $K$ folds
2. For each fold as CV data, and other folds as _training data_:
    1. Learn classifier/regression function on training data
    2. Predict output/class labels on CV folds
3. Use k-fold CV equation to predict averaged CV error. Use this to tune $\lambda$.

### Full CV/Test Setup

* K usually 5 or 10
* Test data unseen during CV
* Test data needs to be _representative_ and should not include data from training set
* Test data can be sampled from training data
* Test data can be used to estimate expected error on unseen data (i.e., generalisation error)

## Lecture 11 - Bayesian Networks

* Represent probability distributions.
* Parameters and/or structure of these p.d.s can be learnt from data using either Bayesian or non-Bayesian methods.

### Uncertain Evidence

Soft/uncertain evidence is if the variable is not fixed to a particular state (value), with the strength of belief about each state being given by probabilities.

#### Hard evidence

Certain a variable is in a particular state. In such a case, all the probability mass is in one of the vector components (i.e., $p(\text{outcome} = a) = 1$ and every other $p(\text{outcome} = \{b \vee c \vee d \vee \dots \}) = 0$)

#### Inference

Inference with soft evidence can be achieved via Bayes' rule. w/ soft evidence as $\tilde{y}$:

$$
p(x|\tilde{y}) = \sum_y p(x|y) p(y|\tilde{y})
$$

> Where $p(y=i|\tilde{y})$ represents the probability $y$ is in state $i$ under the soft evidence. $\tilde{y}$ is a _dummy_ variable representing what is definitely known (i.e., uncertain evidence)

### Jeffrey's rule

To form a join distribution given soft evidence $\tilde{y}$ and variables $x$, $y$, and $p_1(x,y)$:

1. Form the conditional:

$$
p_1(x|y) = \frac{p_1(x,y)}{\sum_x p_1(x,y)}
$$

2. Define the joint

$$
p_2(x,y|\tilde{y}) = p_1(x|y)p(y|\tilde{y})
$$

Soft evidence can be viewed as defining a new joint distribution.

<!-- $$
p_2(x,y|\tilde{y}) = \frac{p_1(x,y)}{\sum_x p_1(x,y)}p(y|\tilde{y})
$$ -->

### Examples of Bayesian Networks in ML

Prediction:

$$
p(\text{class}|\text{input})
$$

Time series:

* Markov chains, hidden Markov models

Unsupervised learning:

$$
p(\text{data}) = \sum_\text{latent}p(\text{data}|\text{latent})p(\text{latent})
$$

### Independence $\perp\!\!\!\perp$ in Bayesian Networks

Conditional independence of $A$ & $B$ given $C$, $A\perp\!\!\!\perp B\; |\; C$ :

$$
p(A,B|C) = p(A|C)p(B|C)
$$

Conditional dependence $A \not\!\perp\!\!\!\perp B \; | \; C$ :

$$
p(A,B|C) \propto p(A,B,C) = p(C|A,B)p(A)p(B)
$$

Marginal independence $A\perp\!\!\!\perp B$:

$$
p(A,B) = \sum_c p(A,B,C) = \sum_C p(A)p(B)p(C|A,B) = p(A)p(B)
$$


#### Colliders

If $C$ has more than one incoming link, then $A \perp\!\!\!\perp B$ and $A \not\!\perp\!\!\!\perp B \; | \; C$. Hence $C$ is a __collider__.

If $C$ has at most one incoming link, then $A \perp\!\!\!\perp B \; | \; C$  and $A \not\!\perp\!\!\!\perp B$. Hence $C$ is a __non-collider__.

### General Rule for independence In Bayesian Networks

Given three sets of nodes $\mathcal{X}$, $\mathcal{Y}$, $\mathcal{C}$, if all paths from any element of $\mathcal{X}$ to any element of $\mathcal{Y}$ are blocked by $\mathcal{C}$, then $\mathcal{X}$ and $\mathcal{Y}$ are conditionally independent given $\mathcal{C}$.

A path $\mathcal{P}$ is blocked by $\mathcal{C}$ if at least one of the following conditions is satisfied:

1. there is a collider in the path $\mathcal{P}$ such that neither the collider nor any of its descendants is in the conditioning set $\mathcal{C}$.
2. there is a non-collider in the path $\mathcal{P}$ that is in the conditioning set $\mathcal{C}$.

#### d-connected/separated

'd-connected' if there is a path from $\mathcal{X}$ to $\mathcal{Y}$ in the 'connection' graph, otherwise the variable sets are 'd-separated'.

d-separation implies $\mathcal{X} \perp\!\!\!\perp \mathcal{Y} \; | \; \mathcal{Z}$, but d-connection does not necessarily imply conditional dependence.

### Markov equivalence

Skeleton:

* Formed from a graph by removing arrows

Immorality:

* An immorality in a DAG is a configuration of three nodes $A,B,C$ such that $C$ is a child of both $A$ and $B$, with $A$ and $B$ not directly connected.

Markov equivalence:

* Two graphs represent the same set of independence assumptions iff they have same skeleton & set of immoralities.

### Limitations of Expressibility

$$
p(t_1, t_2, y_1, y_2, h) = p(t_1)p(t_2) p(y_i|t_1,h)p(y_2|t_2,h) \\
t_1 \perp\!\!\!\perp t_2,y_2 \quad\quad\quad t_2 \perp\!\!\!\perp t_1,y_1
$$

Still holds:
$$
p(t_1, t_2, y_1, y_2) = p(t_1)p(t_2) \sum_h p(y_i|t_1,h)p(y_2|t_2,h) \\
t_1 \perp\!\!\!\perp t_2,y_2 \quad\quad\quad t_2 \perp\!\!\!\perp t_1,y_1
$$

## Lecture 12 - Probability Estimation in BNs and naive Bayes

$$
p(v^1, \dots, v^N, \theta) = p(\theta) \prod_{n=1}^N p(v^N|\theta)
$$

(In above $v^i$ are all descendants of $\theta$)

## Lecture 13 - Undirected Graphical Models

### Graphical Models

* __Belief Network__ - Each factor is a conditional distribution.
* __Markov Network__ - Each factor corresponds to a potential (non-negative function).
    * Relates to strength of relationship between variables, but not directly related to dependence.
    * Useful for collective phenomena (e.g. image processing)
    * Corresponds to undirected graph.
* __Chain Graph__ - Marriage of Belief and Markov Networks.
    * Contains both directed and undirected links
* __Factor Graph__ - Barebones representation of the factorisation of a distribution.
    * Often used for efficient computation and deriving message passing algorithms.
    * Factor graphs one way of representing hypergraphs.
    * Hypergraph just a set of veritices.

### Markov Network

* __Clique__: Fully connected subset of nodes.
* __Maximal clique__: Clique that is not a subset of a larger clique.

Markov networks are undirected graphs in which there exist a potential (non-negative function) $\psi$ defined on each maximal clique.

Joint distribution is proportional to product of all clique potentials. E.g.:

$$
p(A,B,C,D,E) = \frac{1}{Z}\psi(A,C)\psi(C,D)\psi(B,C,E) \\ \; \\
Z = \sum_{A,B,C,D,E} \psi(A,C)\psi(C,D)\psi(B,C,E)
$$

### General Rule for Independence in Markov Networks

For a conditioning set $\mathcal{Z}$ and two sets $\mathcal{X}$ and $\mathcal{Y}$:

* Remove all links neighbouring the variables in the conditioning set $\mathcal{Z}$.
* If there is no path from any member of $\mathcal{X}$ to any member of $\mathcal{Y}$, then $\mathcal{X}$ and $\mathcal{Y}$ are conditionally independent given $\mathcal{Z}$.

### Alternative Rule for Independence in Belief Networks

$\mathcal{X} \perp\!\!\!\perp \mathcal{Y} \; | \; \mathcal{Z}$?
* __Ancestral graph__: Remove any node that is neither in $\mathcal{X} \cup \mathcal{Y} \cup \mathcal{Z}$ nor an ancestor of an node in this set, together with any edges in or out of such nodes.
* __Moralisation__: Add a line in between any two nodes with a common child. Remove directions
* __Separation__: Remove all links from $\mathcal{Z}$.
* __Independence__: If there are no paths from any node in $\mathcal{X}$ to one in $\mathcal{Y}$, then $\mathcal{X} \perp\!\!\!\perp \mathcal{Y} \; | \; \mathcal{Z}$

### The Boltzmann Machine

A Markov network (MN) on binary variables $\text{dom}(x_i) = \{0,1\}$ of the form:

$$
p(\mathbf{x}|\mathbf{w},b) = \frac{1}{Z(\mathbf{x},b)}e^{\sum_{i<j}w_{ij}x_i x_j + \sum_i b_i x_i}
$$

Where interactions $w_{ij}$ are the _weights_ and $b_i$ the biases.

* Model has been studied as basic model of distributed memory & computation. $x_i = 1$ represents a neuron _firing_ and $x_i = 0$ not firing. Matrix $\mathbf{w}$ describes which neurons are connected to each other. The conditional:

$$
p(x_i = 1 | x_{\backslash i}) = \sigma \left(b_i + \sum_{i \neq j} w_{ij} x_j\right), \quad \sigma(x) = \frac{e^x}{1+e^x}
$$

<!-- $$
p(x_i = 1 | x_{i}) = \sigma \left(b_i + \sum_{i \neq j} w_{ij} x_j\right), \quad \sigma(x) = \frac{e^x}{1+e^x}
$$ -->

* Graphical model of BM is an undirected graph with a link between nodes $i$ and $j$ for $w_{ij} \neq 0$. For all but specially constrained $\mathbf{w}$ inference will be typically intractable.
* Given a set of data $\mathbf{x}^1, \dots, \mathbf{x}^n$, parameters $\mathbf{w}, b$ can be set by maximum likelihood.

> Ommitted: Boltzmann machines are ‘Pairwise Markov networks’, The Ising model

### Expressiveness of Belief and Markov networks

> On paper

## Lecture 14 - Hidden Markov Models

### Time series

A time series is an ordered sequence:

$$
x_{a:b} = \{x_a, x_{a+1}, \dots , x_b \}
$$

Allows consideration of the _past_ and _future_ in the sequence. _x_ can be either discrete or continuous.

### Markov Models

For timeseries data $v_1, \dots, v_T$ we need a model $p(v_{1:T})$. $v_t$ are random variables with same domain (system _state)_. For casual consistency:

$$
p(v_{1:T}) = \prod_{t=1}^{T} p(v_t|v_{1:t-1})
$$

With the convention $p(v_t | v_{1:t-1}) = p(v_1)$ for $t=1$.

__Independence assumptions__:

* Often natural to assume influence of immediate past more relevant than remote past and, in Markov models, only a limited number of previous observaions are required to predict the future.

### Markov chains

Only the recent past is relevant:

$$
p(v_t | v_1, \dots, v_{t-1}) = p(v_t | v_{t - L}, \dots, v_{t-1})
$$

where $L \geq 1$ is the order of the Markov chain

$$
p(v_{1:T}) = p(v_1)p(v_2|v_1)p(v_3|v_2)\dots p(v_T|v_{T-1})
$$

For a stationary Markov chain the transitions $p(v_t = s'|v_{t-1} = s) = f(s',s)$ are time-independent ('homogeneous'). Otherwise chain is non-stationary ('inhomogeneous')


### Hidden Markov Models

The HMM defines a Markov chain or hidden ('latent') variables $h_{1:T}$. The observed ('visible') variables are dependent on the hidden variables through an emission $p(v_t|h_t)$. This defines a joint distribution:

$$
p(h_{1:T}, v_{1:T}) = p(v_1|h_1)p(h_1) \prod_{t=2}^T p(v_t|h_t) p(h_t| h_{t-1})
$$

For stationary HMM, the transition $p(h_t| h_{t-1})$ and emission $p(v_t|h_t)$ are constant through time.

### HMM Parameters

#### Transition Distribution

For a stationary HMM the transition distribution $p(h_{t+1}|h_t)$ is defined by the $H \times H$ transition matrix

$$
A_{i',i} = p(h_{t+1} = i'|h_t = i)
$$

and an initial distribution

$$
a_i = p(h_1 = i)
$$

#### Emission Distribution

For a stationary HMM and an emission distribution $p(v_t|h_t)$ with discrete states $v_t \in \{1, \dots, V\}$, we define a $V \times H$ emission matrix

$$
B_{i,j} = p(v_t = i|h_t = j)
$$

For continuous outputs, $h_t$ selects one of $H$ possible output distributions $p(v_t|h_t), \; h_t \in \{1, \dots, H\}$

### Classical Inference Problems

| | | |
| ---------- | --------- | --------|
| __Filtering__ | Inferring the present | $p(h_t|v_{1:t})$ |
| __Prediction__ | Inferring the future | $p(h_t|v_{1:s}) \quad t > s$ |
| __Smoothing__ | Inferring the past |$p(h_t|v_{1:u}) \quad t < u$ |
| __Likelihood__ | |  $p(v_{1:t})$|
| __Most likely hidden path__ | Viterbi alignment | $\underset{h_{1:T}}{\arg\max} p(h_{1:T}|v_{1:T})$

For prediction, one is also often interested in $p(v_t|v_{1:s})$ for $t > s$.

### Filtering $p(h_t|v_{1:t})$

If $a(b_t) \equiv p(h_t, v_{1:t})$:

$$
\alpha(h_t) = \underbrace{p(v_t|h_t)}_{\text{corrector}} \underbrace{\sum_{h_{t-1}} p(h_t|h_{t-1}) \alpha (h_{t-1})}_{\text{predictor}}, \quad t > 1
$$

with

$$
\alpha(h_1) = p(h_1, v_1) = p(v_1 | h_1)p(h_1)
$$

Normalisation gives filtered posterior:

$$
p(h_t|v_{1:t}) \propto \alpha(h_t)
$$

#### Likelihood $p(v_{1:T})$

$$
p(v_{1:T}) = \sum_{h_t} p(h_T, v_{1:T}) = \sum_{h_T} \alpha(h_T)
$$

### Parallel Smoothing

Can compute smoothed quantitiy by considering how $h_t$ partitions the series into the past and future:

$$
\begin{align}
    p(h_t, v_{1:T}) &= p(h_t, v_{1:t}, v_{t+1:T}) \\
    &= \underbrace{p(h_t, v_{1:t})}_{\text{past}} \underbrace{p(v_{t+1:T}|h_t, v_{1:t})}_{\text{future}} = \alpha(h_t) \beta(h_t)
\end{align}
$$

__Forward__: The term $\alpha(h_t)$ is obtained from the 'forward' $\alpha$ recursion.

__Backward__: Term $\beta(h_t)$ may be obtained using a 'backward' $\beta$ recursion.

Forward and backward recursions are independent and may therefore run in parallel, with results combined to obtain smoothed posterior.

> Omitted: $\beta$ recursion

### Computing the Pairwise Marginal $p(h_t, h_{t+1}|v_{1:T})$

$$
p(h_t, h_{t+1}|v_{1:T}) \propto \alpha(h_t)p(v_{t+1}|h_{t+1})p(h_{t+1}|h_t)\beta(h_{t+1})
$$

### Most likely joint state

Most likely path $h_{1:T}$ of $p(h_{1:T}|v_{1:T})$ is the same as the most likely state of:

$$
p(h_{1:T}|v_{1:T}) = \prod_t p(v_t | h_t) p(h_t| h_{t-1})
$$

Consider

$$
\max_{h_T} \prod_{t=1}^T p(v_t|h_t)p(h_t | h_{t-1}) \\
= \left\{\prod_{t=1}^{T-1} p(v_t|h_t)p(h_t|h_{t-1})\right\}\underbrace{\max_{h_T}p(v_T|h_T)p(h_T|h_{T-1})}_{\mu(h_{T-1})}
$$

$\mu(h_{T-1})$ conveys information from the end of the chain to the penultimate timestep.

Can define recursion:

$$
\mu(h_{T-1}) = \max_{h_t}p(v_T|h_T)p(h_T|h_{T-1})\mu(h_t), \quad 2 \leq t \leq T
$$

With $\mu(h_T) = 1$. This means the effect of maximising over $h_2, \dots, h_T$ is compressed into a message $\mu(h_1)$ so that the most likely state $h_1^{* }$ is given by:

$$
h^{* }_1 = \underset{h_1}{\arg\max}p(v_1|h_1)p(h_1)\mu(h_1)
$$

Once computed, backtracking gives:

$$
h^{* }_1 = \underset{h_t}{\arg\max}p(v_t|h_t)p(h_t|h^{* }_{t-1})\mu(h_t)
$$

### Prediction

Predicting future hidden variable

$$
p(h_{t+1}|v_{1:t}) = \sum_{h_t} p(h_{t+1}|h_t)\underbrace{p(h_t|v_{1:t})}_{\text{filtering}}
$$

Predicting future observation, one step ahead:

$$
p(v_{t+1}|v_{1:t}) = \sum_{h_t, h_{t+1}}p(v_{t+1}| h_{t+1})p(h_{t+1}|h_t)[(h_t|v_{1:t})
$$
