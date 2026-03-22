---
layout: post
title:  "Autograd总结"
# date:   2025-12-08 11:18:26 -0800
categories: ML
typora-root-url: ..
mathjax: true
---

### sub-gradient of minimum norm

To understand the **sub-gradient of minimum norm**, we first have to step away from "classic" calculus (where a slope is a single number) and look at **Convex Analysis**.

### 1. What is a Sub-gradient?

For a differentiable function like $f(x) = x^2$, there is exactly one tangent line at any point. But for a function with a "kink" (like the absolute value $|x|$ at $x=0$), there isn't one unique tangent. Instead, there is a **whole range** of lines that can touch that point without ever going above the function.

Any slope $g$ that satisfies this "staying below the function" property is called a **sub-gradient**. The set of all these possible slopes at a point is called the **sub-differential**.

### 2. The "Minimum Norm" Rule

If you have a set of many valid sub-gradients (slopes), which one should a computer choose? PyTorch uses the **minimum norm** rule as a tie-breaker.

- **"Norm"** is just a fancy word for the "length" or "magnitude" of the gradient vector.
- **"Minimum Norm"** means picking the slope that is closest to zero.

------

### Example: The Absolute Value Function $|x|$

Let's look at $f(x) = |x|$ at the point $x = 0$:

- The slope from the left is $-1$.
- The slope from the right is $+1$.
- **The Sub-differential:** Any value in the interval $[-1, 1]$ is a valid sub-gradient.
- **The Minimum Norm:** Out of all numbers between $-1$ and $1$, which one has the smallest magnitude (is closest to $0$)?
  - **Answer:** $0$.

This is why, in PyTorch, `torch.abs(torch.tensor(0.0, requires_grad=True)).backward()` results in a gradient of **0**.

------

### Example: ReLU

For $f(x) = \text{max}(0, x)$ at $x = 0$:

- The slope from the left is $0$.
- The slope from the right is $1$.
- **The Sub-differential:** Any value in the interval $[0, 1]$.
- **The Minimum Norm:** The smallest value in $[0, 1]$ is **0**.

------

### Why do we do this?

1. **Stability:** Picking the smallest possible gradient helps prevent "gradient explosion" during training. If we picked a large sub-gradient arbitrarily, the weights might jump too far.
2. **Consistency:** It provides a deterministic, mathematical rule so the behavior of the library is predictable.
3. **Optimality:** In optimization theory, if $0$ is in the sub-differential, it means you have reached a local minimum. By picking the minimum norm (0), the optimizer naturally "stops" moving because it recognizes it has nothing more to do.

[Image comparing gradient of x squared vs sub-gradient of absolute value at zero]

------

**Would you like to see how this works for functions with multiple inputs, like `torch.max(a, b)`, where the sub-gradient involves a vector?**

## tape-based reverse mode auto-differentiation

In the world of Automatic Differentiation (AD), a **"tape"** (also called a **Wengert List**) is a data structure that records every operation performed on tensors during the "Forward Pass" in the exact order they occurred.

Think of it like a **flight data recorder** for math.

------

### How the Tape Works

When you perform operations in a framework like PyTorch or TensorFlow (in eager mode), the library isn't just calculating the answer; it is "writing" to a tape.

1. **The Forward Pass (Recording):** As you run `y = x * w + b`, the engine records:
   - The **Inputs** ($x, w, b$).
   - The **Operation** (Multiplication, then Addition).
   - The **Outputs** ($y$).
2. **The Backward Pass (Replaying):** When you call `.backward()`, the engine "rewinds" the tape. It reads the operations in **reverse order** and applies the chain rule to each one to propagate the gradients from the end back to the beginning.

------

### Why is it called a "Tape"?

The analogy comes from old-school magnetic tapes or ticker tape.

- **Sequential:** You can only record what you do as you do it.
- **Rewindable:** To find the derivative, you must start at the end of the tape and work your way back to the start.

### Tape-Based (Dynamic) vs. Static Graphs

This "tape" concept is what makes PyTorch a **Dynamic Computational Graph** system.

| **Feature**                  | **Tape-Based (PyTorch/Autograd)**             | **Static Graph (Old TensorFlow)**             |
| ---------------------------- | --------------------------------------------- | --------------------------------------------- |
| **When is the graph built?** | On-the-fly, as the code runs.                 | Before the code runs (defined first).         |
| **Control Flow**             | You can use standard Python `if` and `while`. | Requires special operations (like `tf.cond`). |
| **Flexibility**              | The "tape" can be different every iteration.  | The graph is fixed and "re-used."             |
| **Analogy**                  | A tape recorder capturing a live performance. | A pre-written blueprint of a building.        |

------

### Example: Tape in Action

If your code has a conditional:

Python

```
if x > 0:
    y = x * 2
else:
    y = x / 2
```

If `x` is 5, the tape records **"Multiply by 2"**. If `x` is -5, the tape records **"Divide by 2"**. The tape only cares about what *actually happened* in that specific forward pass.

------

### Is the tape always there?

No. This is why we use `with torch.no_grad():`. This command tells PyTorch: **"Stop the tape recorder."**

- **Tape ON:** Consumes more memory (stores inputs for later) but allows for `.backward()`.
- **Tape OFF:** Much faster and uses less memory; used for inference/prediction where you don't need gradients.

**Would you like to see how the tape handles "In-place" operations, which often cause the "tape" to throw an error?**





## References

- https://www.cs.ubc.ca/~fwood/CS340/lectures/AD1.pdf
- https://www.cs.ubc.ca/~fwood/CS340/lectures/AD2.pdf
- [Automatic differentiation in PyTorch](https://openreview.net/pdf?id=BJJsrmfCZ)
- [Automatic differentiation wiki](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [A Gentle Introduction to `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Example implementation of reverse-mode autodiff](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC)
- [Reverse-mode automatic differentiation from scratch, in Python](https://sidsite.com/posts/autodiff/)
- [micrograd](https://github.com/karpathy/micrograd)
- [Simple Autograd](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC#scrollTo=rnvTWs4W4Hea)

