"""
Tutorial: How AI Learns — Linear Regression from Scratch
========================================================

This script teaches the core training loop used by neural networks:
  1. Forward pass  — make a prediction
  2. Loss          — measure how wrong the prediction is
  3. Gradients     — compute how each weight affects the loss
  4. Update        — nudge weights to reduce the loss

We fit a simple line  y = w*x + b  to a small dataset using gradient descent.
No ML libraries are used — only basic Python and math.
"""

import random


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
# Each input x has a known target y_true. The model must learn w and b so that
# y_pred = w*x + b is close to y_true for every point.

X = [1, 2, 3, 4]
Y_TRUE = [3, 5, 7, 9]  # underlying relationship: y = 2*x + 1


# ---------------------------------------------------------------------------
# 2. Model parameters (weights)
# ---------------------------------------------------------------------------
# w (weight)  — scales the input; controls the slope of the line
# b (bias)    — shifts the output; controls where the line crosses the y-axis
#
# We start with random values because the model does not know the answer yet.
# Training will adjust w and b until predictions improve.

w = random.random()
b = random.random()


# ---------------------------------------------------------------------------
# 3. Hyperparameters
# ---------------------------------------------------------------------------
# learning_rate — how big each weight update step is (too large = unstable,
#                 too small = very slow learning)
# steps         — how many times we repeat the train loop

LEARNING_RATE = 0.03
STEPS = 100


def predict(weight: float, bias: float, x: float) -> float:
    """Forward pass: compute the model's prediction for a single input."""
    return weight * x + bias


def mean_squared_error(weight: float, bias: float, xs: list, ys: list) -> float:
    """
    Loss function (MSE): average squared difference between predictions and targets.

    L = (1/n) * sum( (w*xi + b - yi)^2 )

    Squaring the error means large mistakes are penalized more than small ones.
    """
    n = len(xs)
    total_loss = 0.0
    for x, y in zip(xs, ys):
        error = predict(weight, bias, x) - y
        total_loss += error ** 2
    return total_loss / n


def compute_gradients(weight: float, bias: float, xs: list, ys: list) -> tuple[float, float]:
    """
    Gradients: partial derivatives of the loss with respect to w and b.

    For one data point with error = (w*x + b - y):
      dL/dw = 2 * error * x
      dL/db = 2 * error * 1

    We average these over the full dataset (batch gradient descent).
    """
    n = len(xs)
    grad_w = 0.0
    grad_b = 0.0

    for x, y in zip(xs, ys):
        error = predict(weight, bias, x) - y
        grad_w += 2 * error * x
        grad_b += 2 * error

    return grad_w / n, grad_b / n


def train(
    weight: float,
    bias: float,
    xs: list,
    ys: list,
    learning_rate: float,
    steps: int,
) -> tuple[float, float]:
    """
    Training loop: repeat forward pass → loss → gradients → weight update.

    Weight update rule (gradient descent):
      w = w - learning_rate * dL/dw
      b = b - learning_rate * dL/db

    We move each weight in the direction that reduces the loss.
    """
    for step in range(steps):
        grad_w, grad_b = compute_gradients(weight, bias, xs, ys)

        weight -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if step % 20 == 0 or step == steps - 1:
            loss = mean_squared_error(weight, bias, xs, ys)
            print(
                f"  step {step:3d} | w={weight:.4f}  b={bias:.4f}  loss={loss:.6f}"
            )

    return weight, bias


def main() -> None:
    print("=" * 60)
    print("  How AI Learns: Training a Linear Model from Scratch")
    print("=" * 60)

    # --- Before training ---
    print("\n[1] Initial random weights")
    print(f"    w = {w:.4f}")
    print(f"    b = {b:.4f}")
    print(f"    loss = {mean_squared_error(w, b, X, Y_TRUE):.4f}")

    # --- Training ---
    print(f"\n[2] Training for {STEPS} steps (learning_rate={LEARNING_RATE})")
    final_w, final_b = train(w, b, X, Y_TRUE, LEARNING_RATE, STEPS)

    # --- After training ---
    print("\n[3] Updated weights")
    print(f"    w = {final_w:.4f}  (target ~ 2.0)")
    print(f"    b = {final_b:.4f}  (target ~ 1.0)")
    print(f"    loss = {mean_squared_error(final_w, final_b, X, Y_TRUE):.6f}")

    print("\n[4] Predictions vs. ground truth")
    for x, y in zip(X, Y_TRUE):
        y_pred = predict(final_w, final_b, x)
        print(f"    x={x}  y_true={y}  y_pred={y_pred:.4f}")

    print("\n" + "=" * 60)
    print("  Training complete. Lower loss = better fit.")
    print("=" * 60)


if __name__ == "__main__":
    main()
