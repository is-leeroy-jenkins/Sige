# Inferential Statistics

## 🧭 Purpose

The Inferential Statistics section compares numeric columns and categorical relationships using standard statistical tests.

## 🧪 Welch t-Test

Schedule-X runs pairwise Welch t-tests across selected numeric columns. Welch's t-test does not assume equal variance between compared samples.

| Output | Meaning |
|---|---|
| `var1` | First numeric column. |
| `var2` | Second numeric column. |
| `tstat` | Test statistic. |
| `pvalue` | Probability of observing the result under the null hypothesis. |

## 🧪 Mann-Whitney U

The Mann-Whitney U test provides a non-parametric comparison between selected numeric columns. It can be useful when distributional assumptions are weak.

| Output | Meaning |
|---|---|
| `var1` | First numeric column. |
| `var2` | Second numeric column. |
| `U` | Mann-Whitney U statistic. |
| `p-Value` | Test p-value. |

## 🧪 One-Way ANOVA

ANOVA compares numeric values across groups defined by a selected categorical column.

| Requirement | Description |
|---|---|
| Numeric columns | One or more numeric features selected. |
| Grouping column | One categorical column selected. |
| Groups | At least two groups are required. |

## 🧪 Chi-Square Test

The chi-square test evaluates association between two categorical columns using a contingency table.

| Output | Meaning |
|---|---|
| Chi-square statistic | Magnitude of difference between observed and expected counts. |
| p-value | Statistical evidence against independence. |
| Contingency table | Observed category-pair counts. |

## ✅ Recommended Use

1. Run descriptive statistics first.
2. Select columns with enough non-missing observations.
3. Use meaningful categorical groupings.
4. Interpret results with domain context.
5. Treat p-values as screening evidence, not final decisions.

## ⚠️ Limitations

Statistical tests may be sensitive to missing values, repeated measures, non-independent columns, structural zeros, and policy-driven distributions. Review data lineage and analytical intent before drawing conclusions.
