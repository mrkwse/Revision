# MLAP Spring

## I - Manifold Learning (PCA)

### Manifold learning

* Objects typically characterized by features
* $m$ features $\to m$-dimensional space
* Arena is $m$-dimensional vector space
* Raw pixel values: $m$ by $n$ gives $mn$ features
* Feature space is space of all $m$ by $n$ images



### Manifold

* Space of all face-like images smaller than space of all images
* Assumption is faces lie on a smaller _manifold_ embedded in the global space.
* Would be best to just work on Manifold
    * Calculation & representation efficiency

<!-- ### Low Dimensional Data in Higher Dimensional Space

* Pixel trigrams $x=(g_1, g_2, g_3)^T$
* Occupy a sausage shaped region -->

__Manifold__: A space that locally looks Euclidean

__Manifold learning__: Finding the manifold representing the objects we are interested in

> All objects should be on the manifold, non-objects should be outside.

### Manifold Learning

* __Principal Component Analysis__: Find most significant dimensions in terms of variance, discard dimensions of insignificant variance.
* __Multidimensional scaling__: Embed similarity data into vector space
* __Linear Discriminant Analysis__: Similar to PCA, but define directions to best separate data belonging to different classes.
* __Isomap__: Data resides on a developable manifold. Data unwraps into a lower dimension space.
    * E.g., data resides on a swiss roll, and we unroll the data into a planar sheet.
    * Preserve adjacency structure of data, as represented by a NN-graph of geodesic distances on manifold
* __Locally Linear Embedding__: Data resides on a manifold with intrinsic curvature (e.g. a sphere) which flattens locally.
    * Preserve local adjacency structure of geodesic distance

### Principal Components

__PCA__:

* _Spread_ of data is important (measured by _variance)_
* Rigid transformation of data: translation and rotation only
* Possibly discard redundant or unimportant dimensions to get subspace

#### Method

*  Choose new origin and axes for the data with following properties:
    1. Origin is at mean of data
    2. First axis $\mathbf{u}_1$ is in the direction of maximum data variance
    3. Second axis $\mathbf{u}_2$ is perpendicular to first, and subject to that constraint has the maximum data variance.
    4. …
    5. Profit?
* New and old axes are related by a rigid transform
* New axes give best description in terms of describing most variance in smallest no. axes

### Covariance

$$
cov(x,y) = \frac{1}{n} \sum_{i=1}^n (x_i - \mu_x)(y_i - \mu_y) \\
\begin{align}
\mu_x &= \frac{1}{n} \sum_{i=1}^n x_i \\
\mu_y &= \sum_{i=1}^n y_i
\end{align}
$$

Matrix:

* Set of centered long vectors:

$$
S = {\mathbf{x}_1, \dots, \mathbf{x}_n}
$$
*
    * With centering:

$$
\frac{1}{n} \sum_{u=1}^{n} \mathbf{x}_i = 0
$$

* Data matrix (vectors as columns)

$$
\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_n)
$$

* Covariance matrix

$$
\sum = \frac{1}{n}\mathbf{XX}^T = \frac{1}{n}
\begin{pmatrix}
    \mathbf{x}_1 \\
    \vdots \\
    \mathbf{x}_n
\end{pmatrix}
(\mathbf{x}_1 | \dots | \mathbf{x}_n)
$$

* Elementwise

$$
\sum (u,v) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i (u) \mathbf{x}_i (v)
$$

### Matrices, Packing, and Origami

* Column vectors $\mathbf{x}$ - data vector, $\mathbf{u}$ - eigenvector, both of length $l$.
* Are columns of $l \times n$ data matrix $\mathbf{X} = (\mathbf{x}_1 | \dots | \mathbf{x}_n)$ and a $m \times m$ eigenvector matrix $\mathbf{U} = (\mathbf{u}_1 | \dots | \mathbf{u}_m)$.
* _Elements of outer product_ $(\mathbf{XX}^T)_ {u,v} = \sum_{i=1}^n \mathbf{x}_i (u) \mathbf{x}_i (v)$
    * Covariance of vector components $u$ and $v$
* _Elements of inner product_ $(\mathbf{X}^T\mathbf{X})_ {i,j} = \sum_{u=1\dots m} \mathbf{x}_i (u) \mathbf{x}_j (u)$
    * Inner product of data vectors $i$ and $j$

### Computing Variance Along a Line

* __zero-mean__ the data-points
    * Moves center of data-points to the origin
* The position of a point along $\mathbf{u}$ ($\mathbf{u}$ is a unit vector) is given by:

$$
\begin{align}
\mathbf{x}(u) &= | \mathbf{x} | \cos \theta \\
&= \mathbf{x} . \mathbf{u}
\end{align}
$$

* Variance of points along this axis can be written as:

$$
\begin{align}
\mathit{Var}_u [\mathbf{x}] &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}_i.\mathbf{u})^2 \quad &&\text{: variance of centered q'ty} \\
&= \frac{1}{n} \mathbf{u}^T \mathbf{XX}^T \mathbf{u} \quad &&\text{: replace sum with matrix} \\
&= \mathbf{u}^T \Sigma \mathbf{u}
\end{align}
$$

### Maximisation

Eigendecomposition of covariance matrix

$$
\Sigma = \sum_{i=1}^m \lambda_i \mathbf{u}_i \mathbf{u}_i^T
$$

Variance

$$
\mathit{Var}[\mathbf{u}] = \sum_{i=1}^m \lambda_i \mathbf{u}^T \mathbf{u}_i \mathbf{u}_i^T \mathbf{u} = \sum_{i=1}^m \lambda_i (\mathbf{u}^T \mathbf{u}_i)^2
$$

Maximum when $\mathbf{u}$ is in the direction of eigenvector $\mathbf{u}_i$, i.e., $\mathbf{u}^T \mathbf{u}_i = 1$, when $\mathit{var}[\mathbf{u}] = \lambda_i$

* Problem is therefore to find $\mathbf{u}$ to maximise $\mathbf{u}^T \Sigma \mathbf{u}$ with $\Sigma = \frac{1}{n}\mathbf{XX}^T$
* Linear algebra gives us a solution to this problem via the eigenvalues and eigenvectors of the covariance matrix
* Principal eigenvector is in the direction of maximum variance, ($\mathbf{u}$ equal to the principal eigenvector maximizes $\mathbf{u}^T \Sigma \mathbf{u}$)
* Second eigenvector is orthogonal to the first, and in the direction of maximum variance * Gives new set of axes defined by eigenvectors

* Covariance matrix always has the following properties
    * Symmetric
    * Positive semidefinite (all eigenvectors either zero or positive)
* Therefore has following eigendecompositon

$$
\Sigma = \mathbf{UDU}^T \quad \mathbf{U} = \begin{pmatrix}
\mathbf{u}_0 & \mathbf{u}_1 & \dots
\end{pmatrix}
$$

* New coordinate system is created from first by _rotating_ data-points
    * Brings calculated directions in line with axes
    * Rotation matrix is matrix of eigenvectors $\mathbf{U}$
* New data matrix is
    * A rotation:

$$
\mathbf{X}' = \mathbf{U}^T \mathbf{X}
$$

### transformations

* Covariance matrix of transformed data is:

$$
\begin{align}
\Sigma' &= \frac{1}{n} \mathbf{XX}'\,^T \\
&= \frac{1}{n} \mathbf{U}^T\mathbf{X} \left(\mathbf{U}^T\mathbf{X}\right)^T = \frac{1}{n} \mathbf{U}^T \mathbf{XX}^T \mathbf{U} \\
&= \mathbf{U}^T \Sigma \mathbf{U}
\end{align}
$$

* Diagonal since $\Sigma = \mathbf{UDU}^T$,  $\Sigma' = \mathbf{U}^T \mathbf{UDU}^T \mathbf{U}=\mathbf{D}$.

$$
\Sigma' = \begin{pmatrix}
    \lambda_0 & 0 & \dots \\
    0 & \lambda_1 & 0 \\
    \vdots & 0 & \ddots
\end{pmatrix}
$$

* Covariance between two features zero (uncorrelated)
* Variance of each feature is the associated _eigenvalue_

### Reducing Number of dimensions

* Interesting data is often in a smaller space to the feature space we are using (lower dimensionality)
* PCA is a rigid transform - points are in a new position & orientation, but the relative positions are the same; nothing lost
* Can use PCA to _reduce dimensionality_
* After PCA, a component may have an eigenvalue of zero
    * Implies variance of data in that dimension is zero
    * Dimension can be dropped for _free (no error_ in representation)
* Component may have a 'small' eigenvalue
    * Implies variance is small
    * Can drop dimension with _small cost_ to the accuracy


### Embedding error

Following theorem holds for PCA:

* Let $\mathbf{x}_i'$ be the point positions after PCA
* Let $\hat{\mathbf{x}}_i$ be the point positions after PCA and we have dropped some dimensions

MSE:

$$
\mathit{MSE} = \frac{1}{n} \sum_i |\mathbf{x}_i' - \hat{\mathbf{x}}_i|^2 = \sum_{j \in \text{dropped}} \lambda_j
$$

* Always drop smallest eigenvalues first
* PCA gives the minimum square error between the reduced dimensionality representation and the original

## II - Manifold Learning (MDS)

### Motivation for methods

* Distance between sample points in feature space fully describes arrangement of points
    * Up to a translation, rotation, and reflection of feature space
    * Relative position of points encapsulated by the distances
* PCA took starting points and ended up with new points on a new manifold
* Multidimensional Scaling (MDS) allows direct change between distances to points on a manifold.

* Equivalent to PCA with Euclidean distances

### Distances

* Start w/ set of _distances_ between points
    * Don't have to be Euclidean distances
    * Some data not expressed as points in feature space, rather as relative distances
    * _'Dissimilarities'_ as don't need to be metric/real distances

### Distance Matrix

* Squared distance matrix - square & symmetric
    * $\mathbf{D}$ contains _squared_ distance between points
    * Distance from $A$ to $B$ same as distance from $B$ to $A$
    * Zero Diagonal

$$
\mathbf{D} = \begin{pmatrix}
0 & d_{01}^2 & d_{02}^2 & \dots \\
d_{01}^2 & 0 & d_{12}^2 & \dots \\
d_{02}^2 & d_{12}^2 & \ddots & \dots \\
\vdots & \vdots & \dots & 0
\end{pmatrix}
$$

### Problem Formulation

* Find set of points that have, under Euclidean metric, same squared distance matrix as $\mathbf{D}$
* Let $\mathbf{X}$ be the matrix of these points. Hence:

$$
d_{ij}^2 = (\mathbf{x}_i - \mathbf{x}_j)^2
$$

* Goal is to find the points in $\mathbf{X}$ so this holds

$$
\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_n) \\
d_{ij}^2 = (\mathbf{x}_i - \mathbf{x}_j)^2 = \mathbf{x}_i^2 + \mathbf{x}_j^2 - 2\mathbf{x}_i . \mathbf{x}_j
$$

* Introduce kernel matrix $\mathbf{K}$:

$$
\mathbf{K} = \mathbf{X}^T\mathbf{X} \quad K_{ij} = \mathbf{x}_i^T \mathbf{x}_j = \mathbf{x}_i . \mathbf{x}_j
$$
* V. important matrix
* Hence:

$$
D_{ij} = d_{ij}^2 = K_{ii} + K_{jj} - 2K_{ij}
$$

* Therefore, $\mathbf{K}$ directly in terms of $\mathbf{D}$:

$$
\mathbf{K} = - \frac{1}{2} \left(
    \mathbf{I} - \frac{
        \mathbf{J}
    }{
        n
    }
\right) \mathbf{D} \left(
    \mathbf{I} - \frac{
        \mathbf{J}
    }{
        n
    }
\right)
$$

> Here $n$ is the number of points, $\mathbf{I}$ is the $n$ by $n$ identity matrix, and $\mathbf{J}$ is an $n$ by $n$ matrix containing $1$ in every entry

* Hence can find $\mathbf{K}$ purely from known square distance matrix
* $\mathbf{K}$ is symmetric (if $\mathbf{D}$ is)
* Knowing $\mathbf{K}$, can find $\mathbf{X}$ from $\mathbf{K} = \mathbf{X}^T\mathbf{X}$

### Coordinate Matrix from Eigendecomposition

* Can find $\mathbf{X}$ using the eigenvecetor decomposition:

$$
\begin{align}
\mathbf{K} &= \mathbf{U\Lambda} \mathbf{U}^T = \sum_l \lambda_l \mathbf{u}_l \mathbf{u}_l^T \\
&= \underbrace{\mathbf{U\Lambda}^{^1/_ 2}}_{\mathbf{X}^T} \underbrace{\mathbf{\Lambda}^{^1/_ 2}\mathbf{U}^T}_{\mathbf{x}} \\
\mathbf{X} &= \mathbf{\Lambda}^{^1 /_ 2} \mathbf{U}^T
\end{align}
$$

$$
\begin{align}
\Lambda &= \begin{pmatrix}
\lambda_1 & 0 & \dots & 0 \\
0 & \lambda_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & \dots & \dots & \lambda_n
\end{pmatrix} \\ \; \\
\Lambda^{^1 / _ 2} &= \begin{pmatrix}
\sqrt{\lambda_1} & 0 & \dots & 0 \\
0 & \sqrt{\lambda_2} & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & \dots & \dots & \sqrt{\lambda_n}
\end{pmatrix}
\end{align}
$$
* Giving solution for $\mathbf{X}$.
* Finding $\mathbf{X}$ relies on ability to take square root of eigenvalues
* Following MDS procedure to find $\mathbf{K}$ from $\mathbf{D}$ will sometimes return negative eigenvalues.

### Negative eigenvalues

* Assuming $\mathbf{T}=\mathbf{X}^T\mathbf{X}$ implies points can be represented in Euclidean space
    * If they cannot, discard negative Eigenvalues, giving an _approximate_ result

#### Dealing with Negative Eigenvalues (Psuedo-inverse)

Eigenvalues in MDS:

* _Zero eigenvalues_: Can discard these & corresponding dimensions with no const
* _Negative eigenvalues_: Must discard these and corresponding dimensions, and the result becomes approximate
* _Small positive eigenvalues_: Can discard these and corresponding dimensions with small cost to get lower dimensional representation - keep largest eigenvalues

### Algorithm Summary

1. Begin with symmetric distance matrix $\mathbf{D}$
2. Compute kernel matrix $\mathbf{K}$

$$
\mathbf{K} = - \frac{1}{2} \left(
    \mathbf{I} - \frac{
        \mathbf{J}
    }{
        n
    }
\right) \mathbf{D} \left(
    \mathbf{I} - \frac{
        \mathbf{J}
    }{
        n
    }
\right)
$$

3. Compute eigendecomposition of $\mathbf{K}$

$$
\mathbf{K} = \mathbf{U\Lambda} \mathbf{U}^T
$$

4. Set all negative eigenvalues to zero
5. Set small positive values to zero if dimensionality reduction necessary
6. Compute $\mathbf{X} = \mathbf{\Lambda}^{^1 / _ 2} \mathbf{U}^T$

## Non-Euclidean Data

Matrix of dissimilarities $\mathbf{D}$ between objects can be:

* Euclidean (no intrinsic curvature)
* Non-Euclidean, metric (curved manifold)
* Non-metric (no point-like manifold representation)

### Implications for Kernel Embedding

* Find similarity matrix

$$
\mathbf{K} = - \frac{1}{2} \mathbf{C} \mathbf{D}_s \mathbf{C}
$$

* Data is Euclidean iff $\mathbf{K}$ is positive semidefinite (no negative Eigenvalues)
    * $\mathbf{K}$ is a kernel, explicit embedding from kernel embedding
* Can then use $\mathbf{K}$ in a kernel algorithm

### Metric Data

Dissimilarities may be Non-metric

Data is metric if it obeys following metric conditions:
1. $D_{ij} \geq 0$ (non negativity)
2. $D_{ij} = 0$ iff $i=j$ (identity of indiscernibles)
3. $D_{ij} = D_{ji}$ (symmetry)
4. $D_{ij} \leq D_{ik} + D_{kj}$ (triangle inequality)

Reasonable dissimilarities should meet (1) and (2).

### Causes of non-metricity

* 'Extended objects'
    * Data lives on a manifold and gives rise to triangle violations $D_{ij} \leq D_{ik} + D_{kj}$
    * Noise in measurement of $\mathbf{D}$
