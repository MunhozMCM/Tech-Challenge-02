# ML Experiments — Decision Log

All numbers below were measured on the 20% held-out test set (2,466 sessions), stratified split, `SEED = 42`.

## Decision 1: Lowering the classification threshold to 0.3

### Context

Both models output a purchase **probability** per session, not a hard label. A threshold converts that probability into a binary prediction:

- probability ≥ threshold → predict **Purchase (1)**
- probability < threshold → predict **No Purchase (0)**

sklearn's default threshold is **0.5** ("more likely than not").

### Problem with the default threshold

The dataset is imbalanced: only ~15.5% of sessions end in a purchase. Both models learn this prior and become conservative at the default threshold:

| Metric (Purchase class) | LR @ 0.5 | LR @ 0.3 | RF @ 0.5 | RF @ 0.3 |
|---|---|---|---|---|
| Recall | 0.361 | **0.542** | 0.031 | **0.715** |
| Precision | 0.738 | 0.666 | 1.000 | 0.611 |
| F1 | 0.485 | **0.597** | 0.061 | **0.659** |

The Random Forest case is extreme: at 0.5 it flags almost nobody (recall 0.031). With `max_depth=5` (the production configuration in `src/modelo.py`), each tree outputs smoothed leaf probabilities and their average rarely exceeds 0.5 for the minority class — the default threshold makes the production model nearly useless, even though its ranking quality is the best of all models (AUC-ROC 0.903). The weak recall is purely a threshold problem, not a model quality problem.

### Business justification

For a purchase-propensity use case, the two types of errors have asymmetric costs:

| Error type | What happens | Typical cost |
|---|---|---|
| False Negative (miss a likely buyer) | No intervention (recommendation, nudge, remarketing); conversion opportunity lost | High |
| False Positive (flag a non-buyer) | Incentive/campaign shown to someone who wouldn't buy | Low |

Missing a potential buyer costs the full basket value; a wasted nudge costs cents. Therefore **higher recall is preferable**, even at the cost of lower precision.

### Chosen threshold: 0.3

Threshold 0.3 was selected to meaningfully improve recall while keeping precision operationally acceptable, and is applied consistently to all models for fair comparison. It should be revisited once the business can quantify the average value of a converted session and the cost of an intervention; a precision-recall curve analysis can then find the optimal threshold.

---

## Decision 2: Dropping BounceRates and ProductRelated_Duration due to multicollinearity

### Evidence

Two diagnostics were computed on the numerical features before model training:

**Pearson correlation (r) — feature pairs above 0.6:**

| Pair | r | Action |
|---|---|---|
| BounceRates ↔ ExitRates | **0.913** | drop BounceRates |
| ProductRelated ↔ ProductRelated_Duration | **0.861** | drop ProductRelated_Duration |
| Informational ↔ Informational_Duration | 0.619 | both kept |
| Administrative ↔ Administrative_Duration | 0.602 | both kept |

**Variance Inflation Factor (VIF):** the highest values are ProductRelated (6.34), ProductRelated_Duration (6.01), ExitRates (5.98) and BounceRates (5.56). Unlike the classic VIF > 10 case, no feature crosses the problematic threshold here — the drops are driven by the pairwise correlations above 0.85, which are strong enough to destabilize logistic-regression coefficients and split SHAP credit arbitrarily between the twins.

### Which member of each pair to keep

- **ExitRates kept** over BounceRates: stronger correlation with the target (r = −0.207 vs −0.151) and it measures the more general behavior (last-page exits vs single-page bounces, which are a subset).
- **ProductRelated kept** over ProductRelated_Duration: marginally stronger target signal (r = 0.159 vs 0.152), and a page count is more robust than a duration — the EDA showed the duration columns carry the most extreme outliers in the dataset (upper whisker ~3.4k vs max ~64k seconds).

The pairs at r ≈ 0.6 (counts vs durations for Administrative and Informational) were kept: the correlation is moderate and each member carries distinct signal.

---

## Decision 3: One-hot encoding the integer-coded categorical columns

`OperatingSystems`, `Browser`, `Region` and `TrafficType` arrive as integers (1, 2, 3, …), but they are **nominal identifiers** — "Browser 2" is not twice "Browser 1". Treating them as continuous features imposes a fake ordering that a linear model would interpret as a monotonic effect.

They are therefore one-hot encoded together with `Month`, `VisitorType` and `Weekend` (`pd.get_dummies`, `drop_first=True`), producing 66 features from the 15 retained columns.

> **Note for the DVC pipeline:** `src/pipeline.py` currently applies `LabelEncoder` to the categorical columns, which keeps them as single ordinal integers. That is acceptable for tree models (splits can isolate any code) but sub-optimal for linear ones, and `LabelEncoder` is designed for targets, not features. Migrating the pipeline to `OneHotEncoder` inside a `ColumnTransformer` (as done in the telco project) is a documented improvement candidate.

`SpecialDay` stays numeric: it is a genuine ordinal measure (closeness to a special date, 0 → 1), so its ordering is meaningful.

---

## Observation 1: Model comparison — Dummy vs Logistic Regression vs Random Forest

All models evaluated at threshold = 0.3 on the same test set:

| Metric | Dummy Classifier | Logistic Regression | Random Forest |
|---|---|---|---|
| Accuracy | 0.845 | 0.887 | 0.885 |
| Precision (Purchase) | 0.000 | **0.666** | 0.611 |
| Recall (Purchase) | 0.000 | 0.542 | **0.715** |
| F1 (Purchase) | 0.000 | 0.597 | **0.659** |
| AUC-ROC | 0.500 | 0.880 | **0.903** |
| PR-AUC | 0.155 | 0.614 | **0.705** |

### Key findings

1. **The Dummy Classifier's 84.5% accuracy is a trap** — it never flags a single buyer (recall 0). It exists purely as the lower-bound sanity check and shows why accuracy is the wrong headline metric for this dataset.
2. **Random Forest beats Logistic Regression by a clear margin** (unlike the telco project, where LR tied the neural network). The gap is largest exactly where it matters: PR-AUC 0.705 vs 0.614 and recall 0.715 vs 0.542. The tree ensemble captures non-linearities the linear model cannot — chiefly the threshold-like behavior of `PageValues` (zero for ~78% of sessions, strongly predictive when positive).
3. **This validates the production model choice**: `src/modelo.py` trains exactly this Random Forest configuration (`n_estimators=100, max_depth=5, random_state=42`). Logistic Regression stays in the notebook as the interpretable baseline.
4. **Caveat**: the production pipeline should apply the 0.3 threshold (or a tuned one) at inference; with the registered model's default `.predict()` the recall collapses to 0.031 (see Decision 1).

---

## Observation 2: Feature signal interpretation

### Logistic Regression — SHAP (mean |SHAP|, top features, with coefficient sign)

| Feature | Mean \|SHAP\| | Direction | Interpretation |
|---|---|---|---|
| PageValues | 0.723 | + | Dominant signal. Sessions that touch pages with historical transaction value are far more likely to convert. |
| ExitRates | 0.578 | − | High average exit rate on visited pages → disengaged session → no purchase. |
| Month_May | 0.253 | − | May has the most traffic but below-average conversion — volume ≠ intent. |
| Month_Dec | 0.177 | − | December sessions convert less than the baseline month after controlling for other features. |
| Month_Mar | 0.165 | − | Same pattern as May/December. |
| TrafficType_2 | 0.110 | + | One acquisition channel converts notably above baseline. |
| Month_Nov | 0.109 | + | November (Black Friday season) is the only month with a meaningful positive effect. |
| ProductRelated | 0.095 | + | More product pages viewed → mildly higher purchase odds. |

### Random Forest — Gini importances (top features)

`PageValues` alone carries **55.4%** of the total importance, followed by ExitRates (10.1%), ProductRelated (8.5%), Administrative (4.4%) and Administrative_Duration (4.0%). Month_Nov is the strongest calendar dummy (3.8%).

### Key business takeaways

1. **PageValues is the business lever**: routing users toward high-value pages (and detecting when they arrive there organically) identifies buyers better than any demographic or technical attribute.
2. **High-intent profile**: session touching valued pages, low exit rates, many product pages, in November, from the high-converting traffic channels.
3. **OS/Browser/Region dummies carry almost no signal** — candidates for removal in a future iteration if feature count becomes a concern.
