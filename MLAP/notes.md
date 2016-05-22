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
L(\theta) = \lambda \left(\sum_i \mathit{loss}(y_i, f(x_i))\right) + (1 - \lambda)\sum_j |\theta_j |^p
$$

$$
|\theta| = \left(\theta^2\right)^\frac{1}{2}
$$

$$
\frac{d}{d\theta}|\theta| = \frac{d}{d\theta}\left(\theta^2\right)^\frac{1}{2} =\frac{1}{2}\left(\theta^2\right)^{-\frac{1}{2}}.2\theta = \frac{\theta}{|\theta|} = \mathit{sign}(\theta)
$$

$$
\mathit{sign}(\theta) =
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
\frac{\delta}{\delta \theta_i}\sum_j |\theta_j| = \mathit{sign}(\theta_i)
$$

$$
\Delta L(\theta) = -2 \lambda\left[\sum_i X_i . (y_i - \theta.X_i)\right] + (1 - \lambda)\mathit{sign}(\theta)
$$

Matrix:

$$
\Delta L(\theta) = -2 \lambda X^T (Y - X\theta) + (1 - \lambda)\mathit{sign}(\theta)
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
    \theta_{\mathit{MLE}} &= \underset{\theta}{\arg\max} p(\theta|Y,X,\sigma^2) \\
    &\propto \underset{\theta}{\arg\max} p(Y|X,\theta, \sigma^2) \\
    &\propto \underset{\theta}{\arg\max} \prod_i \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-X_i\theta)^2}{2\sigma^2}} \\
    &\propto \underset{\theta}{\arg\max} \sum_i - \log\left(\sqrt{2\pi\sigma^2}\right) -\frac{(y_i-X_i\theta)^2}{2\sigma^2} \\
    &\propto \underset{\theta}{\arg\min} \sum_i (y_i - X_i\theta)^2
\end{align}
$$

### Regularization as MAP estimation

$$
\begin{align}
    \theta_{\mathit{MAP}} &= \underset{\theta}{\arg\max} p(\theta|Y,X,\sigma^2) \\
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
\frac{1}{N} \sum_i \mathit{error}(y_i, f(x_i))
$$

### Mean Squared Error (MSE)

$$
\frac{1}{N} \sum_{\langle X,y \rangle \in \text{TestData}} \mathit{error}(y - f(X))^2
$$

Square root of MSE, Root Mean Squared Error (RMSE) is common.

### Hard Classification Accuracy

Hard classifier formulates error measure for a given test instance $\langle X,y \rangle$ as:

$$
I(c^{\mathit{hard}}(X,\theta) = y)
$$

Average hard classification accuracy in test set:

$$
\frac{1}{N} \sum_{\langle y,X \rangle \in \text{TestData}} I(c^{\mathit{hard}}(X,\theta) = y)
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
\frac{1}{N} \sum_{\langle y,X \rangle \in \text{TestData}} (1 - I(c^{\mathit{hard}}(X,\theta) = y))
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
\mathit{errorCV}(f = \langle f^{-1}, \dots, f^{-k} \rangle, \lambda) = \frac{1}{N} \sum_i \mathit{error}\left(y_i, f^{-f\,\mathit{old}(i)}(X_i, \lambda)\right)
$$

Optimal $\lambda$ given by:
$$
\lambda^* = \mathit{errorCV}(f = \langle f^{-1}, \dots, f^{-k} \rangle, \lambda)
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
p(\text{class }|\text{ input})
$$

Time series:

* Markov chains, hidden Markov models

Unsupervised learning:

$$
p(\text{data}) = \sum_\text{latent}p(\text{data }|\text{ latent})p(\text{latent})
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

## Lecture 15 - Learning with Hidden Variables

### Hidden Variables and Missing Data

Missing Data

* Data entries are often missing, resulting in incomplete information to specify a likelihood

Observational Variables

* Observational variables may be split into visible (known state) and missing (states would nominally be known but are missing for a particular datapoint)

Latent Variables

* Not all variables in model are observed, so called hidden/latent variables. There are models that are essential for the model description but never observed. There may, for example, be latent processes essential to describe a model, but which cannot be directly measured.

### Hiden/missing Variables can Complicate Things

In learning parameters of models, previously was assumed the complete information was known to define all variables of the joint model of the data $p(v|\theta)$.

### Maximum Likelihood

* For hidden variables $h$ and visible variable $v$, there is a well defined likelihood:

$$
p(v|\theta) = \sum_h p(v,h|\theta)
$$

* Have to find parameter $\theta$ that maximises $p(v|\theta$).
* More numerically complex than when all variables are visible
* Still possible to perform numerical optimisation using any routine to find $\theta$
* EM algorithm is an alternative optimisation algorithm that can be very useful in producing simple & elegant updates for $\theta$ that converge to local optima.

### Kullback-Leibler (KL) Divergence

$\langle f \rangle_q$ denotes the expected value of $f$ with respect to distribution $q$. E.g., if $q$ is discrete, defined over $1, \dots, n$:

$$
\langle f \rangle_q = \sum_{i=1}^n f(i)q(i)
$$

KL divergence is the expected log odds between two distributions:

$$
\begin{align}
    \mathit{KL}(q(x)|p(x)) &= \left\langle
        \log \frac{q(x)}{p(x)}
    \right\rangle_{q(x)} \geq 0 \\
    \mathit{KL}(q(x)|p(x)) &= 0 \text{ iff } p = q
\end{align}
$$

KL divergence is roughly a measure of distance between distributions

### Variational EM

<!-- Key feature of EM algorithm is to form an alternative objective function for which parameter coupling effect is removed, meaning individual parameter updates can be achieved, akin to case of fully observed data. The marginal likelihood is replaced with a lower bound - this lower bound has the decoupled form.

#### Single Observation

The KL divergence between a 'variational' distribution $q(h|v)$ and the parametric model  -->

For i.i.d. data $\mathcal{V} = \{v^1, \dots, v^N\}$:

$$
\log p(\mathcal{V}|\theta) \geq - \sum_{n=1}^N
\left\langle
    \log q(h^n|v^n)
\right\rangle_{q(h^n|v^n)}
+
\sum_{n=1}^N
\left\langle
     \log q(h^n,v^n | \theta)
\right\rangle_{q(h^n|v^n)}
$$

This suggests an iterative process to optimise $\theta$:

* __E-step__ - for fixed $\theta$, find the distributions $q(h^n|v^n)$ that maximise the bound.
* __M-step__ - for fixed $\{q(h^n|v^n),n=1, \dots, N \}$, find the parameters $\theta$ that maximise the bound.

#### Classical EM

In the variational E-step above, the fully optimal setting is

$$
q(h^n|v^n) = p(h^n|v^n, \theta)
$$

## Lecture 16 - Markov Chains

### Equilibrium Distribution

Interesting to know how marginal $p(x_t)$ evolves through time:

$$
p(x_t = i) = \sum_j \underbrace{p(x_t = i| x_{t-1} = j)}_{M_{ij}}p(x_{t-1} = j)
$$

The marginal $p(x_t = i)$ has the interpretation of the frequency that state $i$ is visited at time $t$, given a start of $p(x_1)$ and random drawing of samples from the transition $p(x_T|x_{T-1})$. As repeated samples are taken of new states from the chain, the distribution at time $t$, for an initial distribution $\mathbf{p}_1(i)$ is:

$$
\mathbf{p}_t = \mathbf{M}^{t-1} \mathbf{p}_1
$$

If, for $t \to \infty, \; \mathbf{p}_\infty$ is independent of the initial distribution $\mathbf{p}_1$, then $\mathbf{p}_\infty$ is called the equilibrium distribution of the chain.

$$
p_\infty (i) = \sum_j (x_t = i | x_{t-1} = j) p_\infty (j)
$$

In matrix notation, this can be written as the vector equation

$$
\mathbf{p}_\infty = \mathbf{Mp}_\infty
$$

Hence the stationary distribution is proportional to the eigenvector with unit eigenvalue of the transition matrix.

## Lecture 17 - Sampling and Markov Chain Monte Carlo

### Sampling

Sampling concerns drawing realisations $\mathcal{X} = \{x^1, \dots, x^L\}$ from a distribution $p(x)$. For a discrete variable $x$, in the limit of a large number of samples, the fraction of samples in state $\mathsf{x}$ tends to $p(x = \mathsf{x})$. I.e.,:

$$
\lim_{L \to \infty} \frac{1}{L} \sum_{l=1}^L \left[ x^l = \mathsf{x} \right] = p(x = \mathsf{x})
$$

In the continuous case, can consider a small region $\Delta$ such that the probability that the samples occupy $\Delta$ tends to the integral of $p(x)$ over $\Delta$.

#### Sampling to approximate averages

For a finite set of samples, expectations can be approximated via:

$$
\left\langle f(X) \right\rangle_{p(x)} \approx \frac{1}{L} \sum_{l=1}^L f(x^l) \equiv \hat{f}_\mathcal{X}
$$

Subscript in $\hat{f}_\mathcal{X}$ emphasizes that the approximation is dependent on the set of samples drawn.

### Sampling as Approximation Techniques

* A procedure that 'faithfully' samples from $p(x)$ can be used to approximate averages.
* Suggests a general class of approximation methods for computing averages wrt. otherwise computationally intractable distributions
* Must find sampling procedures that can draw samples from distributions that are analytically & computationally intractable

### Drawing Independent Samples

* Difficult is in generating independent samples
* Sampling high-dimensional distributions is difficult and few guarantees exist that ensure that in a practical timeframe, _independent_ samples are produced.
* A dependent sampling scheme amy be unbiased, but the variance of the estimate may be so high that a large number of samples may be required for accurate approximation of expectations.

### Continuous case

* Calculate cumulant density function:

$$
C(y) = \int^y_{-\infty} p(x) dx
$$

* Then sample $u$ uniformly from $[0,1]$, and obtain corresponding sample $x$ by solving $C(x) = u \implies x = C^{-1}(u)$

> For certain distributions (e.g., Gaussian), numerically efficient alternative procedures exist, usually based on co-ordinate transformations.

### Multi-variate Sampling

* Can generalise 1D discrete case to a higher-dimensional distribution $p(x_1, \dots, x_n)$ by translating this into an equivalent 1D distribution.
* Enumerate all possible joint states $(x_1, \dots, x_n)$, giving each a unique integer $i$ from $1$ to total no. states ($n$), and construct a univariate distribution with probability $p(i) = p(\mathsf{x})$ for $i$ corresponding to the multi-variate state $\mathsf{x}$.
* Generally, this is impractical since number of states grows exponentially with no. variables.

### Ancestral Sampling for Belief Networks

* Rename variable indices so parent variables always precede children.

$$
p(x_1, \dots, x_6) = p(x_1) p(x_2) p(x_3 | x_1, x_2) p(x_4 | x_3) p(x_5 | x_3) p(x_6 | x_4, x_5)
$$

* Sample first from nodes without parents ($x_1$ and $x_2$).
    * Given these values then sample from their immediate offspring ($x_4$ and $x_5$).
    * Finally sample the offspring's offspring
* Despite presence of loops in a graph, such a procedure is straightforward. Procedure holds for both discrete and conrinuous variables.
* Ancestral or 'forward' sampling is a case of perfect sampling since each sample is indeed independently drawn from required distribution.

### Ancestral Sampling with Evidence

* If sampling from $p(x_1,x_2,x_2,x_4,x_5 | x_6)$. Via Bayes' rule:

$$
\frac{p(x_1) p(x_2) p(x_3|x_1,x_2) p(x_4|x_3) p(x_5 | x_3) p(x_6|x_4,x_5)}
    {\sum_{x_1,x_2,x_3,x_4,x_5} p(x_1) p(x_2) p(x_3|x_1,x_2) p(x_4|x_3) p(x_5|x_3) p(x_6|x_4,x_5)}
$$

 > Now $x_4$ and $x_5$ are coupled. Could attempt to find an equivalent new forward sampling structure, but v. complex.

* Alternative is to proceed with forward sampling from non-evidential distribution and discard any samples that don't match evidential states. Generally not recommended as small probability that sample will be consistent with evidence.

### Gibbs sampling

* Consider a particular variable $x_i$, to sample. Under Bayes' rule:

$$
p(x) = p(x_i|x_1, \dots, x_{i-1}, x_{i + 1}, \dots, x_n) p(x_1, \dots, x_{i-1}, x_{i + 1}, \dots, x_n)
$$

* Given a join initial state $x^1$, from which the _parental_ state $x^1_1, \dots, x^1_{i-1}, x^1_{i+1}, \dots, x^1_n$ can be read, draw a sample $x^2_i$ from:


$$
p(x_i|x^1_1, \dots, x^1_{i-1}, x^1_{i+1}, \dots, x^1_n) \equiv p(x_i|x_{\backslash i})
$$

> Assumed this distribution is easy to sample as it is univariate. This new joint sample (in which only $x_i$ has updated) is:

$$
x^2 = (x^1_1, \dots, x^1_{i-1}, x^2_i, x^1_{i+1}, \dots, x^1_n)
$$

* Then another variable $x_j$ is selected to sample, and by continuing this procedure, a set of samples $x^1, \dots, x^L$ is generated in which each $x^{l+1}$ differs from $x^l$ in only a single component

> Omitted figure

* $p(x_i|x_{\backslash i})$ is defined by the Markov blanket of $x_i$:

$$
p(x_i | x_{\backslash i}) \propto p(x_i|pa(x_i)) \prod_{j \in ch(i)} p(x_j|pa(x_j))
$$

* For continuous variable $x_i$, the summation is replaced by integration
* Evidence is readily dealt with by clamping for all samples the evidential variables into their evidential states. No need to sample for these variables since their states are known
* Gibbs sampling is particularly straightforward to implement, but a drawback is that samples are strongly dependent

### Gibbs Sampling as a Markov Chain

Can write Gibbs sampling as a procedure that draws from

$$
x^{l+1} \sim q(x^{l+1}|x^l)
$$

for some distribution $q(x^{l+1}|x^l)$. Choosing the variable to update, $x_i$, at random from a distribution $q(i)$, Gibbs sampling corresponds to the Markov transition

$$
q(x^{l+1}|x^l) = \sum_i q(x^{l+1}|x^l, i)q(i), \quad q(x^{l+1}|x^l, i) = p(x_i^{l+1}|x_{\backslash i}^l) \prod_{j \neq i} \delta (x_j^{l+1}, x_j^l)
$$

Then:

$$
\begin{align}
    \int_x q(x'|x)p(x) &= \sum_i q(i) \int_x q(x'|x_{\backslash i})p(x) \\
    &= \sum_i q(i) \int_x \prod_{j \neq i} \delta (x_j', x_j) p(x_i' | x_{\backslash i}) p(x_i, x_{\backslash i}) \\
    &= \sum_i q(i) \int_{x_i} p(x_i'| x_{\backslash i}') p(x_i, x_{\backslash i}') \\
    &= \sum_i q(i) p(x_i'|x_{\backslash i}')p(x_{\backslash i}') = \sum_i q(i)p(x') = p(x')
\end{align}
$$

Hence, drawing from $q(x'|x)$ will, for $l \gg 1$, draw samples from $p(x)$

### Markov Chain Monte Carlo (MCMC)

* Assume there exists a multi-variate distribution in the form

$$
p(x) = \frac{1}{Z}p^* (x)
$$

> Where $p^* (x)$ is the unnormalised distribution and $Z = \int_x p^* (x)$ is the normalization constant

* Assume ability to evaluate $p^* (x = \mathsf{x})$, for any state $\mathsf{x}$, but not $p(x = \mathsf{x})$ since $Z$ is intractable.
* MCMC sampling is to sample, not directly form $p(x)$, but from a distribution such that, in the limit of a large no. samples, effectively the samples will be from $p(x)$
* Achieved by forward sampling from a Markov transition whose stationary distribution is equal to $p(x)$

### Markov Chains

* Consider conditional distribution $q(x^{l+1}|x^l)$. After long time $L \gg 1$, the samples are from $q_\infty (x)$, which is defined as:

$$
q_\infty (x') = \int_x q(x'|x) q_\infty (x)
$$

* Find for a given distribution $p(x)$ a transition $q(x'|x)$ that has $p(x)$ as its stationary distribution. If possible, then can draw samples from the Markov chain by forward sampling and take these as samples from $p(x)$ as the chain converges towards its stationary distribution.
* Note that for every distribution $p(x)$, there will be more than one transition $q(x'|x)$ with $p(x)$ as its stationary distribution. (Many different MCMC techniques)
