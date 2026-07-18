# Methodology

## Decision problem

The project asks two related questions:

1. Can a model rank new account applications so that fraud is concentrated near the top of the queue?
2. Given a fixed investigation capacity, which alerts should be reviewed and which analyst assignments produce the strongest outcome?

## Validation design

Random train-test splits are not the primary evaluation because the dataset contains an explicit month field and fraud behaviour changes over time.

The planned primary split is:

| Role | Source months |
|---|---|
| Training | 0, 1, 2, 3 |
| Excluded from primary comparison | 4 |
| Validation | 5, 6 |
| Final test | 7 |

Month 4 is excluded because the supplied base file appears truncated during that period. The final test month remains untouched until model and threshold decisions are complete.

## Feature policy

`income`, `customer_age`, `employment_status` and `housing_status` are reserved for subgroup evaluation and are excluded from the primary model. This does not prove that all remaining proxy effects have been removed. The limitation is reported explicitly.

The supplied `model_score` is never used as an input when training a new fraud model. It is evaluated separately as an author-provided benchmark and used in the downstream alert-review analysis.

## Model sequence

1. Constant prevalence baseline.
2. Logistic regression with class weighting.
3. Histogram-based gradient boosting.
4. A tree-boosting model only if it provides a measurable and reproducible benefit.

Model selection is based on validation performance, not the final test month.

## Metrics

Accuracy is not a primary metric for this rare-event problem. The project reports:

- average precision (area under the precision-recall curve);
- ROC AUC as a secondary ranking measure;
- precision and recall at an explicitly selected threshold;
- precision and recall under fixed investigation capacities;
- fraud captured per 100 reviewed applications;
- alert volume and false-positive burden;
- monthly stability;
- subgroup performance for the reserved audit fields.

## Interpretation boundaries

The data and analysts are synthetic. Results demonstrate analytical and engineering methods; they do not establish production readiness, regulatory compliance or suitability for real lending decisions.

## Alert-review experiment

The alert-review stage is evaluated separately from application modelling. Historical analyst correctness is estimated on alert months 3–6. Month 7 is held out for the final comparison across 25 supplied capacity scenarios.

Three assignment policies are compared: random allocation, global historical skill and score-band specialisation. Each policy respects individual analyst limits; alerts not assigned to an analyst retain the supplied screening decision. The detailed design and final results are reported in [Alert review and assignment](ALERT_REVIEW.md).
