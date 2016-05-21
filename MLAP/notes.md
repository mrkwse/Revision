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