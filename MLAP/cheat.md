# MLAP

## Lecture 1

### DeMorgan's Law

$$
\begin{align}
\overline{(\bigcup_{i=1}^nE_i)} &= \bigcap_{i=1}^n \overline{E_i} \\
\overline{(\bigcap_{i=1}^n E_i)} &= \bigcup_{i=1}^n \overline{E_i}
\end{align}
$$

### Axioms of Probability

1.
$$0\leq p(A) \leq 1$$
2.
$$p(S) = 1$$
3. If $A$ and $B$ are mutually exclusive events in $S$ then:

$$
p(A\cup B) = p(A) + p(B)
$$

### Conditional Probability

$$
p(A|B) = \frac{p(A\cap B)}{p(B)}
$$

Estimated by:

$$
p(A|B) \approx \frac{c(A,B)}{c(B)}
$$

$A$ independent iff:

$$
p(A|B) = p(A)
$$

Therefore, $A$ and $B$ independent iff:

$$
p(A,B) = p(A)p(B)
$$


### D-D-D-DROP THE BAYES'

Theorem:

$$
p(A|B) = \frac{p(B|A)p(A)}{p(B)}
$$

## Lecture 2

### Binomial Random Variable

$$
p(X = k) = p(x) = \binom{n}{k} p^k (1-p)^{n-k}
$$

Binomial Theorem:

$$
(a + b)^n = \sum_{k=0}^n \binom{n}{k} a^k b^{n-k}
$$

#### Binomial Distribution

$$
p(X=k|N,\theta) = \frac{N!}{k!(N-k)!}\theta^k (1-\theta)^{N-k}
$$

### Expectation of Discrete Random Variable

$$
E[X] = \sum_{i=1}^{\infty}x_i p(X=x_i)
$$

_The weighted average of all possible values $X$ can take_


### Variance

$$
\mathbf{Var}(X) = E[(X-E[X])^2] = \sum_x (x-E[X])^2 p(X=x)
$$

### Covariance

$$
\begin{align}
\mathbf{Cov}(X,Y) &= E[(X-E[X])(Y-E[Y])] \\
\mathbf{Cov}(X,Y) &= \sum_x \sum_y (x-E[X])(y-E[Y])\;p(X=x, Y=y)
\end{align}
$$

### Quadratic Formula

$$
x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$

## Lecture 3 - Distributions

### Binomial Distribution

<!-- $$
p(X=k|N,\theta) = \frac{N!}{k!(N-k)!}\theta^k (1-\theta)^{N-k}
$$ -->

$$
p(X=k|N,\theta) = \binom{N}{k}\theta^k (1-\theta)^{N-k}
$$

### Cumulative Density (CDF)

$$
F(a \leq X \leq b) = \int_a^b p(x) dx
$$

$p(x)$ can be any distribution (e.g., Beta, binomial, etc.)

pdf and cdf are related by:

$$
\frac{d}{dx} F(x) = p(x)
$$

Expectation:

$$
E[X] = \int^\infty_{-\infty} xp(x) dx
$$

Variance:

$$
\mathbf{Var}(X) = \int^\infty_{-\infty} (x-E[X])^2 p(x) dx
$$

#### Bayes' for Continuous Random Vars

$$
p(y|x) = \frac{p(x,y)}{p(x)} = \frac{p(x|y)p(y)}{\int p(x|y)p(y) dy}
$$

### Normal (Gaussian) Distribution !!!

$$
p(X=x) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

### Gamma Function !!!

$$
\Gamma (N) = (N-1)!
$$

### Beta Distribution

$$
p(\theta|\alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)} \sim \mathbf{Beta}(\alpha,\beta)
$$

Via derivation:

$$
B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$

#### Beta normalization constant !!!

$$
B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$

#### Beta-Binomial Distribution !!!

<!-- $$
p(c|\alpha,\beta) = \frac{N!}{c_1!c_2!}\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \frac{\Gamma(c_1 + \alpha)\Gamma(c_2+\beta)}{\Gamma(c_1+c_2+\alpha+\beta)}
$$ -->

$$
\begin{align}
p(c|\alpha,\beta) &= \frac{N!}{c_1!c_2!}\frac{B(c_1+\alpha,c_2+\beta)}{B(\alpha,\beta)} \\
&= \binom{N}{c_1!c_2!}
\end{align}
$$

## Lecture 4

### Inference

$$
p(\theta|c) = \frac{p(c|\theta)p(\theta)}{\int p(c|\theta)p(\theta)}
$$

* $p(\theta|c)$ is the posterior distribution
* $p(\theta)$ is the prior distribution
* $p(c|\theta)$ is the likelihood

#### MLE Inference

$$
\theta_{\mathbf{MLE}} = \arg \max_\theta p(c|\theta)
$$

#### MAP Inference

$$
 \theta_{\mathbf{MLE}} = \arg \max_\theta \frac{p(c|\theta)p(\theta)}{\int p(c|\theta)p(\theta)d\theta}
$$

### Multinomial Distribution

$$
p(c|\theta) = \frac{C!}{\prod_i c_i!} \prod_i \theta^{c_i}
$$

* Generalises binomial distribution to $k > 2$
* Equivalent to binomial for $k = 2$

### Dirichlet Distribution

Generalizes Beta distribution to the $k-1$ probability simplex.

$$
p(\theta|\alpha) = \frac{\Gamma(\alpha)}{\Gamma(\alpha)\Gamma(\beta)}\prod_i\theta^{\alpha-1} \sim \mathbf{Dir}(\alpha_1, \dots, \alpha_k)
$$

_With:_
$\alpha = (\alpha_1, \dots, \alpha_k), \;\theta=(\theta_1,\dots, \theta_k), \; \sum_i \theta_i = 1, \; A=\sum_i\alpha_i$

### Dirichlet-Multinomial Distribution

$$
p(c|\alpha) = \frac{C!}{\prod_i c_i !} \frac{\Gamma(A)}{\prod_i \Gamma(\alpha_i)} \frac{\prod_i \Gamma(c_i + \alpha_i)}{\Gamma(C+A)}
$$

## Lecture 5 - Linear Regression

### Linear function properties

* Additivity: $f(u+v) = f(u) + f(v)$
* Homogeneity: $f(a*v) = a*f(v)$

### Loss Functions for Regression

Absolute value loss:

$$
|y - f(x)|
$$

Squared loss:

$$
(y-f(x))^2
$$

$\varepsilon$-sensitive loss:

$$
\max(|y-f(x)|-\varepsilon, 0)
$$

#### Total loss

$$
\sum_i \mathbf{loss}(y_i, f(x_i))
$$

or

$$
L(\theta) = \sum_i \mathbf{loss}(y_i, \theta.X_i)
$$

### Least Squares Regression

$$
\theta^* = \underset{\theta}{\arg\min} \sum_i \mathbf{loss}(y_i, \theta.X_i)
$$

Gradient vector:

$$
\begin{align}
\Delta L(\theta) &= \; \left\langle \frac{\delta L(\theta)}{\delta \theta_1}, \frac{\delta L(\theta)}{\delta \theta_2}, \; \dots \right\rangle \\
&= -\sum_i 2(y_i - \theta.X_i) . X_i \\
&= -2 X^T (Y-X\theta)
\end{align}
$$


$$
\left(abc\right)
$$
## Lecture 6 - Data Normalisation

### Data Standardization

Mean subtraction:

$$
\forall x_i : x_i = x_i - \mu_i
$$

Variance normalisation:

$$
\forall x_i : x_i = \frac{x_i - \mu_i}{\sigma_i}
$$

### Euclidean Norm

$$
|x| = \sqrt{x_1^2 + x_2^2 + x_3^2 + \dots + x_k^2}
$$

Norm transform:

$$
\left\langle
x_1, x_2, \dots, x_k
\right\rangle
\mapsto
\left\langle
\frac{x_1}{|x|}, \frac{x_2}{|x|}, \dots, \frac{x_k}{|x|}
\right\rangle
$$

## Feature Engineering/Basis Expansion

_No equations. Many, many graphs._


## Lecture 7 - Logistic Regression

* Decision boundary - point separating two classes

### Step Function Classifier

$$
\mathbf{step}(x) =
    \begin{cases}
        1 & \quad \text{if }\; x>0 \\
        0 & \quad \text{if }\; x\leq 0
    \end{cases}
$$

### Sigmoid

$$
\mathbf{sigmoid}(x) = \frac{1}{1+e^{-x}}
$$

### Logistic Regression Classifier

<!-- $$
\begin{align}
    p(y=1|x) &= \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x} \\
    p(y=0|x) &= 1 - \frac{1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}} = \frac{1}{1+e^x}
\end{align}
$$ -->

$$
\begin{align}
    p(y=1|X,\theta) &= \frac{e^{X\theta}}{1+e^{X\theta}} \\\\
    p(y=0|X,\theta) &=  \frac{1}{1+e^{X\theta}}
\end{align}
$$

_Alternatively, separate parameters $\theta_1,\theta_2$ for two classes:_

$$
\begin{align}
    p(y=1|X,\theta) &= \frac{e^{X\theta_1}}{e^{X\theta_1}+e^{X\theta_2}} \\\\
    p(y=0|X,\theta) &=  \frac{e^{X\theta_2}}{e^{X\theta_1}+e^{X\theta_2}}
\end{align}
$$

## Lecture 8 - Regularisation

### Shrinkage Functions

$$
L'(\theta) = \lambda L (\theta) + (1 - \lambda ) \sum_j |\theta_j |^p
$$

3 Different regularization methods:

* $p=0$ : Subset selection
* $p=1$ : Lasso
* $p=2$ : Ridge

$$
L(\theta) = \lambda \left(\sum_i \mathbf{loss}(y_i, f(x_i))\right) + (1 - \lambda)\sum_j |\theta_j |^p
$$

Choosing $\lambda$:

* $\lambda = 0$ : ignore fit, chose $\theta = 0$
* $\lambda = 1$ : choose least squares solution
* $\lambda  = 0.2$ : choose model giving equal weight to smaller model & fit

## Lecture 9 - Bayesian Interpretation of Regularisation

### Bayesian Interpretation of Linear Regression

$$
\begin{align}
    p(y_i | X_i, \theta, \sigma^2) &= \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-X\theta)^2}{2\sigma^2}} \\
    p(Y|X,\theta,\sigma^2) &= \prod_i p(y_i | X_i, \theta, \sigma^2) \\ &= \prod_i \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i-X\theta)^2}{2\sigma^2}} \\
\end{align}
$$

### Posterior

$$
p(\theta|Y,X,\sigma^2) = \frac{p(Y|X, \theta, \sigma^2)p(\theta)}{\int p(Y,\theta|X,\sigma^2)d\theta}
$$

### MLE For Linear Regression

$$
\begin{align}
\theta_{\mathbf{MLE}} &= \underset{\theta}{\arg\max} p(\theta|Y,X,\sigma^2) \\
&\propto \underset{\theta}{\arg\min} \sum_i (y_i - X_i\theta)^2
\end{align}
$$


### Regularization as MAP estimation

$$
\begin{align}
\theta_{\mathbf{MAP}} &= \underset{\theta}{\arg\max} p(\theta|Y,X,\sigma^2) \\
&\propto \underset{\theta}{\arg\max} p(Y|X,\theta, \sigma^2)p(\theta) \\
    &\propto \underset{\theta}{\arg\min} \sum_i \lambda(y_i - X_i\theta)^2 + (1 - \lambda)\sum_i |\theta_i|^P
\end{align}
$$


##  Lecture 10 - Evaluation and Cross validation

### Mean error

$$
\frac{1}{N} \sum_i \mathbf{error}(y_i, f(x_i))
$$

### Mean Squared Error (MSE)

$$
\frac{1}{N} \sum_{\langle X,y \rangle \in \text{TestData}} \mathbf{error}(y - f(X))^2
$$

Square root of MSE, Root Mean Squared Error (RMSE) is common.

### Hard Classificaton Accuracy
