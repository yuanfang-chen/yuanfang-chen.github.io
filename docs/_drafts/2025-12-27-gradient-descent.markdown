---
layout: post
title:  "Gradient Descent"
# date:   2025-12-08 11:18:26 -0800
categories: ML
typora-root-url: ..
mathjax: true
---

Gradient descent is the mathematical "compass" used to find the minimum of a function. Imagine you are standing on a foggy mountainside; you can't see the bottom, but you can feel the slope under your feet. To get to the valley, you simply take steps in the direction where the ground slopes downward most steeply.

Here is a simple example using a basic quadratic function.

------

### 1. The Goal

We want to find the value of $x$ that minimizes the function:

$$f(x) = x^2$$

We know from basic algebra that the minimum is at \\(x = 0\\), but gradient descent will find this iteratively.

### 2. The Components

To run the algorithm, we need three things:

- **The Gradient (Slope):** The derivative of \\(x^2\\) is \\(f'(x) = 2x\\). This tells us which way is "up."
- **The Learning Rate (\\(\alpha\\)):** This is the size of the steps we take. Let's use **0.1**.
- **Starting Point:** Let’s start at a random spot, like \\(x = 5\\).

### 3. The Step-by-Step Iteration

We update our position using this formula:



$$x_{new} = x_{old} - \alpha \cdot f'(x_{old})$$

| Iteration | Current x | Gradient (\\(2x\\)) | Step (\\(α⋅grad\\))         | New x    |
| --------- | --------- | ------------------- | --------------------------- | -------- |
| **1**     | 5.0       | 10.0                | \\(0.1 \times 10 = 1.0\\)   | **4.0**  |
| **2**     | 4.0       | 8.0                 | \\(0.1 \times 8 = 0.8\\)    | **3.2**  |
| **3**     | 3.2       | 6.4                 | \\(0.1 \times 6.4 = 0.64\\) | **2.56** |
| **...**   | ...       | ...                 | ...                         | ...      |
| **10**    | 0.53      | 1.06                | \\(0.106\\)                 | **0.42** |

### 4. Why it Works

- **Direction:** When $x$ is positive (like 5), the gradient is positive. Subtracting it moves us to the **left** (toward 0).
- **Slowing Down:** As we get closer to the bottom, the slope (gradient) gets smaller. This naturally causes our steps to get smaller, preventing us from overshooting the target.
- **Convergence:** Eventually, the gradient becomes so close to zero that our position stops changing. We have "converged" on the minimum.

------

#### Key Pitfalls

- **Learning Rate too high:** You might "bounce" back and forth across the valley and never reach the bottom.
- **Learning Rate too low:** The descent will be agonizingly slow, taking thousands of steps to reach the bottom.

## [stochastic gradient descent](https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16) (SGD)
SGD consider only a single example at a time and to take update steps based on one observation at a time. The drawbacks are
- statistically, single example is not representative
- computationally, batched computation (matrix-vector multiplication) of gradient descent is much more efficient than single gradient descent (vector-vector multiplication) 

## minibatch stochastic gradient descent
take a minibatch of observations. a number between 32 and 256, preferably a multiple of a large power of 2, is a good start

## why use mean square error (MSE)

$$-\log P(\mathbf{y}\mid\mathbf{X})=\sum_{i=1}^n\frac{1}{2}\mathrm{log}(2\pi\sigma^2)+\frac{1}{2\sigma^2}\left(y^{(i)}-\mathbf{w}^\top\mathbf{x}^{(i)}-b\right)^2$$

minimizing the mean squared error is equivalent to the maximum likelihood estimation of a linear model under the assumption of additive Gaussian noise.

## References

https://gemini.google.com/share/30b83f34b8e5

