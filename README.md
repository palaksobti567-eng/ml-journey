# ml-journey
# Week 1: House Price Predictor (Linear Regression from Scratch)

## Problem
Predict a house's price from its size using a straight line, price = w × size + b.

## Data
5 houses. Size in 1000 sqft: [1.0, 1.7, 2.0, 2.5, 3.0]. Price in $1000s: [250, 300, 480, 430, 630].

## Method
- Cost function: J(w, b) = 1/(2m) Σ (w·x + b − y)²
- Gradients derived on paper (see `gradient-derivation.jpg`):
  ∂J/∂w = (1/m) Σ (f − y)·x and ∂J/∂b = (1/m) Σ (f − y)
- Gradient descent written in NumPy: start at w = 0, b = 0 and repeatedly step against the gradient.
- Final run used learning rate 0.1 and 1000 iterations.

## Result
| Method | w | b |
|---|---|---|
| My gradient descent | 180.701 | 49.371 |
| `np.polyfit` (exact) | 180.703 | 49.365 |

- My analytic gradient matches a numerical estimate at (w=0.5, b=10): (−914.286, −406.98) vs (−914.286, −406.98).
- Example: a 2,200 sqft house is predicted at about $447k. With only 5 houses and one feature, treat this as a learning exercise, not a real valuation.



![Fitted line](fitted-line.png)




![Learning rates that converge](lr-converging.png)



## Learning rate comparison


![All learning rates](lr-divergence.png)



| Learning rate | Behaviour |
|---|---|
| 0.001 | Converges, very slowly |
| 0.01 | Converges, slower than needed |
| 0.1 | Converges fast |
| 0.5 | Diverges |
| 1 | Diverges faster |

## What went wrong
- [Your note: the typo `metplotlib`, and the missing `/(2*eps)` in the numerical gradient, which showed up as a gradient of −0.018 instead of −914.]
- Learning rates 0.5 and 1 diverged. [In your own words: the step is so big that it jumps past the minimum and lands higher on the other side, where the slope is steeper, so each jump is bigger than the last and the cost grows every step.]
- For this data, the largest stable learning rate is about 0.36. Anything above that diverges.