# [Coursera Machine Learning course](https://www.coursera.org/learn/machine-learning)

## Week 1

### What is Machine Learning

> The field of study that gives computers the ability to learn without being explicitly programmed.
> – Arthur Samuel

More recently:

> A computer program is said to learn from experience E, with respect to some task T, and some performance measure P, if its performance on T as measured by P improves with experience E.
> – Tom Mitchel

Example: playing checkers.

- E = the experience of playing many games of checkers
- T = the task of playing checkers.
- P = the probability that the program will win the next game.

Types of machine learning:

- Supervised learning
- Unsupervised learning
- (Reinforcement learning)
- (Recommender systems)

### Supervised Learning

Example: house prices, "right answers" are given

"Fit a model in the data"

- **Regression** problem: Continuous value output
- **Classification** problem: Discrete valued output

Feature = attribute

### Unsupervised Learning

Structure of the data - clusters

Applications of clustering algorithms:

- Computing clusters organisation
- Market segmentation
- Social network analysis
- Astronomical data analysis

### Model Representation

Linear regression!

Housing prices: Price (1000s $), Size (feet squared)

Training set

Notation:

- **m** = Number of training examples
- **x**'s = "input" variable / features
- **y**'s = "output" variable / "target" variable
- **(x, y)** = One training example
- **(x(i), y(i))** = i-th training example

"Univariate" linear regression

> learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.

```
          ┌─────────────────┐
          │   Training set   │
          │                  │
          └─────────────────┘
                   │
                   │
                   ▼
          ┌─────────────────┐
          │    Learning      │
          │    algorithm     │
          └─────────────────┘
                   │
                   │
      x            ▼        predicted y
                 ┌───┐
(living area ───▶│ h │───▶  (predicted
  of house)      └───┘    price of house)
```

### Cost function

"Minimise the difference between the hypotheses and the actual value"

Squared error function: Average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

Works well for most problems. Most common (for linear regression problems). There are other cost functions.

Contour plots - charts concentric circles of equal values for a given set of variables

### Gradient descent

Very common algorithm. Not just linear regression.

- Start with some θ0 and θ1
- Keep changing θ0, 01 to reduce J(θ0, θ1) until we hopefully end up at a minimum

Local minimums / local optimums! Common starting value: 0, 0

![Gradient descent illustration](assets/function-gradient-descent.png)

Follow the derivative (tangent line) of the cost function, down slope.

Simultaneous updated of both θ0 and θ1 (interdependent)

Alpha gradient descent step:

- Too small, gradient descent too slow
- Too large, overshoot the minimum, fail to converge, *or even diverge*.

Eventually converges regardless of learning rate alpha, because derivative tends to 0, thus taking smaller steps.

Convex function (bowl shaped): no local optima, just one global optimum.

Quadratic gradient descent:

![](assets/quadratic-gradient-descent.png)

## Matrices and Vectors

> Matrix: Rectangular array of numbers

Dimension of matrix: number of rows x number of columns

> Vector: Matrix with n x 1 dimension

Identity matrix, denoted I or ][n*m

## Week 2

### Octave

- Install: ``.
- Start: `octave`

```sh
# Install
brew install octave
# Start
octave
# Help
help <func_name>
```

### Multivariate Linear Regression

Notation:

- `n` = number of features
- `x(i)` = input (features) of i-th training example
- `x(i)j` = value of feature j in i-th training example

Hypothesis, multivariate edition:

```python
hθ(x) = θ0 + 01x1 + θ2x2 + θ3x3 + θ4x4
# Any number of variables, with x0 = 1.
hθ(x) = θ0x0 + 01x1 + ... + θnxn
```

#### Gradient descent – Feature scaling

If features use different scales, gradient descent will be very slow (lots of steps). The countour graph is very skewed.

Put all features on the same scale - approximate value range: -1 <= xi <= 1 range (not smaller, not bigger). Rule of thumb: -3 – 3, -1/3 – 1/3 are ok

Mean normalisation: Give features a 0 mean by reducing all feature values by the prev mean (in training set).

```python
# u1 = avg
# s1 = sd, or range
x1 <- (x1 - u1) / s1
```

#### Gradient descent – Learning rate

Plot J(θ) cost function per number of iterations. J(θ) should decrease after every iteration.

See after how many iterations the gradient descent has converged.

Gradient descent not working - most common reason is too high learning rate α.

- For sufficiently small α, J(θ) should decrease on every iteration.
- But if α is too small, gradient descent can be slow to converge.
- Try α: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1 (steps of 3x)

#### Features and polynomial regression

Housing prices prediction: hθ(x) = θ0 + θ1 * frontage + θ2 * depth, or hθ(x) = θ0 + θ1 * area (frontage * depth)

Instinct for the shape of the data based on the shape of different regressions - quadratic, cubic, square root. Use feature scaling accordingly.

- Combine multiple features into one.

### Computing parameters analytically

#### Normal equation

> Method to solve for θ analytically

Gradient descent:

- Need to choose α
- Needs many iterations.
- Works well even when n (nb features) is large

Normal equation:

- No need to choose α.
- Don't need to iterate
- Need to compute O(n3).
- Slow if n is very large.

Rule of thumb: less than 1000 features = normal equation

θ = (X^T*X)^-1 * X^T * y

X = attributes matrix padded with 1 of size m * (n + 1)
y = m-dimensional vector

#### Non-invertible matrix in normal equation

- Redundant features
- Too many features

Delete some features, or use regularization

### Octave Tutorial
