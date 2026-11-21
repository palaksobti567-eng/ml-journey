# Week 5: SMS Spam Classifier

Classify SMS messages as spam or ham (normal). Compares a majority-class baseline, TF-IDF + Logistic Regression, and a small Keras neural network. Thresholds are chosen on a validation set, and the test set is used once.

## Problem

Spam is the rare class (about 13% of messages), so accuracy is misleading: a model that always says "ham" scores about 86.6% and catches no spam. This project reports precision, recall and F1 for the spam class instead.

## Dataset

- **UCI SMS Spam Collection**: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- 5,572 messages, one per line: `label<TAB>text`, labels `ham` / `spam`
- About 13.4% spam
- The data is not committed (`data/` is in `.gitignore`). Download the zip and extract `SMSSpamCollection` into `data/`.

## Method

| Step | Detail |
|---|---|
| Split | Stratified 60 / 20 / 20 train / validation / test, `random_state=42`. Split first, before any fitting. |
| Baseline | `DummyClassifier(strategy="most_frequent")` |
| Model 1 | `Pipeline`: `TfidfVectorizer(max_features=5000)` then `LogisticRegression(max_iter=1000)` |
| Model 2 | Keras: Dense(32, ReLU), Dense(1, sigmoid), Adam, binary cross-entropy, 10 epochs, batch size 32 |
| Leakage control | TF-IDF is fit on the training text only. The pipeline does this for Model 1. For Keras, `fit_transform` on train and `transform` on validation and test. |
| Threshold rule | Set before looking at test: **highest recall with precision >= 0.95 on validation.** |
| Chosen thresholds | LogReg = 0.35, Keras = 0.25 |

### Why this threshold rule

A false positive blocks a real message (a bank code, a friend's text). A false negative lets spam through, which is annoying but usually harmless. So precision must stay high, and recall is then maximised within that limit.

## Results

### Validation set (used to choose thresholds)

| Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority baseline | n/a | 0.000 | 0.000 | 0.000 |
| LogReg | 0.35 | 0.977 | 0.859 | 0.914 |
| Keras | 0.25 | 0.951 | 0.913 | 0.932 |

The majority baseline has validation accuracy 0.866.

### Test set (evaluated once)

| Model | Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| LogReg | 0.35 | 0.971 | 0.907 | 0.938 | 136 | 4 | 14 |
| Keras | 0.25 | 0.935 | 0.953 | 0.944 | 143 | 10 | 7 |

Test set: 1,115 messages, 150 of them spam.

### Reading the results

- Both models are far above the baseline, which catches no spam.
- Keras catches 7 more spam than LogReg (143 vs 136) but blocks 6 more real messages (10 vs 4). F1 is nearly identical (0.944 vs 0.938).
- This is a trade-off, not a clear win. If blocking a real message is the costlier error, LogReg is the safer choice, and it is simpler and faster to train.

## What went wrong / limitations

- **Small test set.** With only 150 spam messages, one or two messages move recall by about 1 point. Test scores came out higher than validation, which is most likely noise, not a better model. The difference between the two models is within that noise.
- **Close threshold calls.** LogReg at 0.30 had validation precision 0.949 and Keras at 0.25 had 0.951, both right at the 0.95 rule. The rule was fixed before the test run and not changed afterwards.
- **Mild overfitting in Keras.** Training loss reached 0.012 while validation loss flattened at about 0.065 from epoch 7 on.

- No cross-validation, learning curves or error analysis of misclassified messages yet.

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

1. Download the dataset (link above) and place `SMSSpamCollection` in `data/`.
2. Open `spam.ipynb`, select the `.venv` kernel, then **Restart and Run All**.

TensorFlow must support your Python version. If the install fails, run the Keras cells in Google Colab.

## Files

| File | Purpose |
|---|---|
| `spam.ipynb` | Full analysis, runs top to bottom |
| `requirements.txt` | Pinned dependencies (generate with `python -m pip freeze` inside this project's venv) |

| `data/` | Dataset (not committed) |

## What changed from the plan

- Added a validation-chosen threshold for each model instead of using the default 0.5.
- Skipped the optional SVM row and dropout experiments to keep to the minimum deliverable.
- Related project: `week5-backprop` (backprop from scratch with a numerical gradient check).

## Next steps

- Error analysis: read the 7 to 14 missed spam and the blocked real messages.
- Stratified cross-validation (mean ± std) to check whether the models really differ.
- Add the optional SVM row for comparison.
