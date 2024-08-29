# Results of the models with different preprocessing techniques

## Script base_logreg

### Run 1: Baseline Logistic Regression

- Filled missing values of numerical features with median and categorical features with mode.
- Replaced the individual values of the categorical features which have high unique values.
  - Threshold: 0.02 (Proportion threshold below which a category is considered rare.)
- Used one-hot encoding for categorical features.
- Use StandardScaler for numerical features.

```bash
Fold=0, MCC=0.6593286609493691
Fold=0, Accuracy=0.8306627162173218
Fold=1, MCC=0.6577848187626045
Fold=1, Accuracy=0.8298815025610012
Fold=2, MCC=0.6578235389193939
Fold=2, Accuracy=0.8298622529431864
Fold=3, MCC=0.6587059045715644
Fold=3, Accuracy=0.8303017858832927
Fold=4, MCC=0.6593675944613145
Fold=4, Accuracy=0.8306947989136799
```

### Run 2: Removed Columns

- Removed the columns with greater than 40% missing values.

```bash
Fold=0, MCC=0.5394574728818569
Fold=0, Accuracy=0.7714364546053909
Fold=1, MCC=0.537104124164315
Fold=1, Accuracy=0.7703279974462174
Fold=2, MCC=0.5383833399131047
Fold=2, Accuracy=0.7709969216652844
Fold=3, MCC=0.5387593976620945
Fold=3, Accuracy=0.7711493144729855
Fold=4, MCC=0.538481987007113
Fold=4, Accuracy=0.7710594829231828
```

**Observation**: The performance has dropped when the columns are removed.

### Run 3

- Changed the threshold for rare categories to 0.01.
- Removed the columns with greater than 50% missing values.

```bash
Fold=0, MCC=0.5402706466612397
Fold=0, Accuracy=0.7717123657940708
Fold=1, MCC=0.5394874785754549
Fold=1, Accuracy=0.7713867264260358
Fold=2, MCC=0.5398871088107922
Fold=2, Accuracy=0.7715423275033727
Fold=3, MCC=0.5396845707765894
Fold=3, Accuracy=0.7714637248972953
Fold=4, MCC=0.5412397474459396
Fold=4, Accuracy=0.7722882501936993
```

**Observation**: The performance has not changed much.

### Run 4

- Changed the threshold for rare categories to 0.01.
- Not removing any columns.

```bash
Fold=0, MCC=0.6645050404013358
Fold=0, Accuracy=0.832927754580206
Fold=1, MCC=0.6639254429789625
Fold=1, Accuracy=0.8326710930093408
Fold=2, MCC=0.6643634662177198
Fold=2, Accuracy=0.8328250899518599
Fold=3, MCC=0.6643308411245172
Fold=3, Accuracy=0.8328154651429525
Fold=4, MCC=0.6650848595948751
Fold=4, Accuracy=0.8332678311616022
```

**Observation**: The performance has improved little bit. I think this is the best result we can get with this setup. To improve it further, we need to try different models or make some major changes in the preprocessing.

## Script base_rf

- In this script, I have used RandomForestClassifier with the default parameters.

### Run 1: Baseline RandomForest

```bash
Fold=0, MCC=0.9834076966806502
Fold=0, Accuracy=0.9917739966537747
Fold=1, MCC=0.9829475532780092
Fold=1, Accuracy=0.991546209509632
Fold=2, MCC=0.9825124152774455
Fold=2, Accuracy=0.9913296513092146
Fold=3, MCC=0.9834628957070806
Fold=3, Accuracy=0.9918012669456792
Fold=4, MCC=0.9835807225984636
Fold=4, Accuracy=0.9918606199339417
```
