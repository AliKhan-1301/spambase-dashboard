# What gives spam away?

An analysis of the UCI Spambase dataset with a live dashboard: **[alikhan-1301.github.io/spambase-dashboard](https://alikhan-1301.github.io/spambase-dashboard/)**

## Result

After removing duplicates, 4,210 emails remain, 39.9% of them spam. A logistic-regression filter classifies 93.2% of held-out emails correctly (AUC 0.976), against 76.6% for the best single-word rule and 60.1% for always answering "not spam".

Spam leans on punctuation, money and calls to action. The strongest real-mail signals, though, are details of one inbox, which limits how far the model travels.

## Cleaning

| Check | Result |
|---|---|
| Rows loaded | 4,601 (57 features + 0/1 label) |
| Missing or non-numeric values | 0 |
| Negative values, percentages above 100, invalid labels | 0 |
| Exact duplicate rows | 391 removed (8.5%) |
| Identical features with opposite labels | 3 pairs, kept and flagged |
| Rows after cleaning | 4,210 (1,679 spam, 2,531 real) |

Duplicates were removed before splitting, so no email can appear in both the training and the test set.

## Findings

Share of emails that contain each feature at least once:

| Spam signal | Spam | Real mail |
|---|---|---|
| `!` | 83% | 29% |
| `$` | 60% | 11% |
| free | 54% | 10% |
| remove | 43% | 2% |
| money | 36% | 2% |

| Real-mail signal | Spam | Real mail |
|---|---|---|
| hp | 3% | 40% |
| hpl | 1% | 30% |
| george | 0% | 26% |
| 1999 | 5% | 28% |
| 650 | 2% | 17% |

Spam also shouts. The average run of consecutive capital letters is 9.9 characters in spam against 2.4 in real mail, and the longest run averages 102 against 19.

## Model

Logistic regression on log(1+x), standardised features, with a stratified 70/30 split (seed 42): 2,947 emails to train, 1,263 to test.

| Metric (test set, threshold 0.5) | Value |
|---|---|
| Accuracy | 93.2% |
| Precision | 92.8% |
| Recall | 89.9% |
| F1 | 91.3% |
| AUC | 0.976 |
| Real emails wrongly blocked | 35 of 759 (4.6%) |

Train accuracy is 94.3%, close to the test score, so the model is not overfitting much. The largest weights towards spam are `$`, remove, `!` and the total number of capital letters. The largest towards real mail are george, hp, meeting and edu.

The dashboard's threshold slider recomputes the confusion matrix from the 1,263 test-set scores. Raising the threshold to 0.80 cuts wrongly blocked real mail from 35 to 18, at the cost of letting 100 spam emails through instead of 51.

## Caveat

The dataset documentation says the non-spam emails came from the creators' own work and personal mail, which is why "george" and the area code "650" predict real mail. The model has partly learned whose inbox it is reading, so it would need retraining on any other mailbox. The emails are also from 1999.

## Files

- `data/spambase_raw.csv`: the dataset as downloaded
- `data/spambase_clean.csv`: de-duplicated
- `analysis.mjs`: validation, cleaning, analysis and the model. Run `node analysis.mjs` (Node 18+, no dependencies) to rebuild `dashboard-data.js`
- `index.html`: the dashboard (Chart.js, loaded with an integrity hash)

## Data source

Hopkins, M., Reeber, E., Forman, G. and Suermondt, J. (1999). *Spambase*. UCI Machine Learning Repository. https://doi.org/10.24432/C53G6X. Licensed CC BY 4.0.

Analysis by [Ali Khan](https://alikhan-1301.github.io).
