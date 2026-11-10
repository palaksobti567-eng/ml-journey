# Week 4: MNIST Digit Classifier (Neural Network from Scratch Understanding)

## Goal
Train a small neural network to recognise handwritten digits (0-9) and, more importantly, understand what happens inside it. The project is about understanding, not about reaching a record score.

## Data
- MNIST: 70,000 grayscale images of 28x28 pixels.
- The 60,000 training images were split into **50,000 train** and **10,000 validation**. The **10,000 test** images were kept untouched until the final evaluation.
- Pixels were scaled from 0-255 to 0-1 so training is stable.

## Model
Flatten (784 inputs) -> Dense 128 (ReLU) -> Dense 10 (softmax)

- 101,770 trainable parameters (784x128 + 128 + 128x10 + 10).
- Adam optimizer, sparse categorical cross-entropy loss, 5 epochs, batch size 32.
- Last epoch: training accuracy 98.50%, validation accuracy 97.39%.

## Results

| Model | Validation accuracy | Test accuracy |
|---|---|---|
| Always guess the most common digit ("1") | about 11.4% | not run |
| Logistic regression on raw pixels | 92.77% | not run |
| Neural network (1 hidden layer, 128 units) | 97.39% | **97.27%** |

The test set was evaluated **once**, after all decisions were made. Test and validation scores agree within noise (0.12 points, about 12 images out of 10,000), so there is no sign that I over-tuned to the validation set.

## Findings
- The hidden layer is worth its cost: errors drop from about 7.2% (logistic regression) to about 2.6% (network), roughly 64% fewer mistakes.
- Training accuracy (98.5%) is ahead of validation (97.4%), and validation flattened after epoch 3-4. This is an early sign of overfitting.
- Each score comes from a single run. Differences of a few tenths of a point should not be over-read; the 4-5 point gap between the network and logistic regression is large enough to trust.

## Error analysis
- **Confusion matrix** (validation set, rows = true digit, columns = predicted digit): accuracy is one number, while the matrix shows which digits get mixed up. The network got 261 of 10,000 validation images wrong.
- Biggest confusion in my matrix: _TODO: fill in from your matrix (true digit -> predicted digit, count)_.
- I displayed 20 misclassified validation images. Some were genuinely ambiguous or badly written (a 6 that looks like an "L", a 9 that looks like a lowercase "a"), so part of the error is probably unavoidable. Others were classic look-alike pairs such as 4/9, 5/3 and 3/8.
- Caution: the 20 images are only the first 20 errors, not a random sample. True 9s, 8s and 3s appeared most often, but I treated this as a hint and checked it against the full confusion matrix.

## Understanding the network: forward pass rebuilt in NumPy
To avoid treating Keras as a black box, I rebuilt the forward pass with the trained weights:

1. Flatten each image to 784 numbers.
2. Hidden layer: `z1 = x @ W1 + b1`, then ReLU (`np.maximum(0, z1)`).
3. Output layer: `z2 = a1 @ W2 + b2`, then softmax (with the row maximum subtracted for numerical stability).

Weight shapes: W1 784x128, b1 128, W2 128x10, b2 10, which add up to the same 101,770 parameters Keras reports. `np.allclose(my_probs, keras_probs, atol=1e-5)` returned `True`, so my version matches Keras. A layer is only a matrix multiply plus bias followed by an activation.

## What went wrong
- I ran the scaling cell twice, which divided pixels by 255 two times (max value 0.0039 instead of 1.0). I fixed it by rerunning load, split and scaling in order, once each. Lesson: notebook cells change shared state, so run order matters.
- I first copied the hidden layer as the output layer (128 units instead of 10). The model summary showed the wrong output shape `(None, 128)`, which is how I caught it.
- TensorFlow could not be installed on my local Python 3.13 (`No matching distribution found`), so I trained in Google Colab.

## How to run
Open `week4_mnist.ipynb` in Google Colab and run all cells top to bottom. Training starts from random weights, so your scores may differ from mine by a few tenths of a point.

## Possible next steps
- Add regularisation (dropout) or early stopping to reduce the overfitting gap.
- Try a convolutional network, which suits images better than flattened pixels.
