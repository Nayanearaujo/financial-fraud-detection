# Alert Review and Assignment

## Scope

The review experiment uses the 30,622 supplied alerts and decisions from 50 synthetic fraud analysts. It is separate from the application-ranking model: the alert table contains an author-provided screening score and a selected population that already crossed a review boundary.

All analysts, decisions and capacity scenarios are synthetic. The results describe an experimental workflow, not employee performance.

## Temporal design

Analyst behaviour is estimated from alert months 3–6 (26,165 alerts). Month 7 (4,457 alerts) is reserved for the final comparison. Score-band boundaries and analyst skill estimates are fitted on the historical period only.

The testbed supplies 25 team scenarios. Each scenario activates 10 analysts and provides individual maximum review capacities. Total capacity is 4,052 reviews per scenario, leaving 405 alerts with the screening decision when all capacity is used.

## Policies compared

| Policy | Assignment rule |
|---|---|
| Random capacity | Shuffle alerts and available analyst slots while respecting every individual limit |
| Global skill | Prioritise assignments where an analyst's historical overall correctness is expected to improve on the screening decision |
| Risk-band specialist | Estimate analyst correctness within ten historical model-score bands and assign cases where the expected advantage is greatest |

Risk-band estimates use smoothing towards each analyst's global result. This prevents a small number of historical cases in one band from producing an extreme skill estimate.

## Final-month results

Results are means across the 25 supplied team scenarios.

| Policy | Accuracy | Precision | Recall | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| Random capacity | 57.15% | 26.12% | **91.73%** | 1,850.84 | **58.80** |
| Global skill | 60.79% | 27.75% | 90.34% | 1,679.04 | 68.68 |
| Risk-band specialist | **60.94%** | **27.78%** | 90.06% | **1,670.40** | 70.68 |

Compared with random capacity allocation, risk-band assignment avoids about 180 false positives per scenario and improves precision by 1.66 percentage points. Recall falls by 1.67 percentage points, equivalent to about 12 additional missed fraud cases per scenario.

The difference between global skill and risk-band specialisation is small. The result supports capacity-aware assignment, but it does not justify a universal policy without explicit investigation and fraud-loss costs.

## Reproducibility

```bash
PYTHONPATH=src python scripts/evaluate_review_strategies.py \
  --source /path/to/FiFAR
```

The script verifies case identifiers, uses a historical/test separation, respects each analyst's capacity and writes both the scenario metrics and case-level assignments to `reports/`.

## Limitations

- Analyst predictions are simulated rather than observed in a live review operation.
- Correctness can change over time; the estimates cover a short synthetic history.
- Accuracy is used for the assignment objective, while precision and recall are reported to expose its operational consequences.
- No review-time, monetary-loss or escalation-cost field is supplied.
- The experiment must not be interpreted as a staff-ranking exercise.
