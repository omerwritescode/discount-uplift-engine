# 🎯 discount-uplift-engine

> *Who deserves a discount — and how much?*  
> A two-stage causal ML pipeline that identifies discount-responsive customers and assigns personalized, profit-maximizing offers using uplift modeling and expected utility optimization.

---

## 📌 Overview

E-commerce platforms routinely apply **uniform discounts** across all customers — ignoring the fact that some customers will buy regardless (wasted margin), some will never buy no matter the offer (wasted spend), and only a targeted subset genuinely change their behavior in response to a discount.

This project builds a **Personalized Discount Prediction Pipeline** that solves all three problems at once:

| Stage | Question | Method |
|---|---|---|
| **Stage 1** | *Who* should get a discount? | Uplift Modeling — T-Learner & X-Learner |
| **Stage 2** | *How much* discount? | Expected Profit Maximization |
| **Evaluation** | Does it actually work? | Qini Curve + Monte Carlo A/B Simulation |

**Dataset:** [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) — Sakar & Kastro (2019), 12,330 e-commerce sessions.

---

## 🏗️ Pipeline Architecture

```
Raw Sessions (12,330)
        │
        ▼
┌─────────────────────┐
│  0. Preprocessing   │  Label encoding, duplicate removal (125 rows), feature selection
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  1. Base Model      │  GBM → Isotonic Calibration (cv='prefit')
│                     │  Outputs calibrated P(buy) per customer
└─────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  2. Uplift Modeling (Stage 1)              │
│                                            │
│   ┌──────────────┐    ┌──────────────────┐ │
│   │  T-Learner   │    │    X-Learner     │ │
│   │  (baseline)  │    │  (imbalance fix) │ │
│   └──────┬───────┘    └────────┬─────────┘ │
│          └────────┬────────────┘           │
│                   ▼                        │
│           Qini Comparison → Winner         │
└────────────────────────────────────────────┘
        │
        ▼
  Customer Segmentation
  ├── Persuadables  → discount changes their behavior  → proceed to Stage 2
  ├── Sure Things   → buy anyway                       → no discount needed
  └── Lost Causes   → won't buy regardless             → no discount needed
        │
        ▼
┌─────────────────────────────────────────────┐
│  3. Profit Optimization (Stage 2)           │
│  Discount tiers: 0% / 5% / 10% / 15% / 20% │
│  E[Profit] = P(buy|disc) × (GM − disc$) − offer_cost │
│  Argmax over tiers → optimal discount/customer       │
└─────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  4. Evaluation           │
│  · Qini Curve Comparison │
│  · Monte Carlo A/B Sim   │
│    (1,000 runs)          │
└──────────────────────────┘
        │
        ▼
  customer_discount_recommendations.csv
```

---

## 🔬 Methods in Detail

### Uplift Modeling — The Core Idea

Standard classification models predict *P(buy)*. Uplift models predict *P(buy | discount) − P(buy | no discount)* — the **causal effect** of giving a discount to a specific customer. This is the Individual Treatment Effect (ITE).

**Treatment assignment** (observational proxy, since no A/B column exists in the dataset):
- Treated: `New_Visitor` OR `SpecialDay > 0` — customers who likely encountered a promotional signal
- Control: Returning visitors on non-special days

#### T-Learner (Künzel et al., 2019)
Train two separate GBMs — one on treated customers, one on control. Apply both to every customer and subtract predictions:

```
uplift = model_treated.predict(x) − model_control.predict(x)
```

Each model is trained on an 80% split and calibrated on the held-out 20% using **Isotonic Regression** (Niculescu-Mizil & Caruana, 2005) to correct GBM's known probability overconfidence.

#### X-Learner (Künzel et al., 2019)
Designed for imbalanced treatment/control splits (ours: 23.6% / 76.4%). Adds two extra steps:

1. Compute **imputed treatment effects** per customer using the opposite group's model:
   - `D_T = actual_outcome − model_control.predict(x)` for treated customers
   - `D_C = model_treated.predict(x) − actual_outcome` for control customers
2. Train two **tau regressors** (`GradientBoostingRegressor`) to predict these effects
3. Combine: `uplift_x = g × tau_t(x) + (1−g) × tau_c(x)`, where `g = P(treatment)` is the empirical propensity score

The weighted combination gives more influence to the larger control-side estimate — a principled correction for group imbalance.

**Winner is selected automatically** by Qini coefficient before Stage 2.

---

### Probability Calibration

Raw GBM outputs are not well-calibrated probabilities — the model optimizes ranking (AUC), not calibration. This causes the profit formula to multiply by inflated numbers.

Fix: `CalibratedClassifierCV` with `method='isotonic'` fits a monotone non-parametric correction on held-out data (`cv='prefit'`). After calibration:

```
Mean predicted probability : 0.1535
Actual purchase rate       : 0.1549
Mean Calibration Error     : ~0.024
```

---

### Customer Segmentation

Using `UPLIFT_THRESHOLD = 0.05`:

| Segment | Condition | Action |
|---|---|---|
| **Persuadable** | `uplift > 0.05` | Assign optimal discount |
| **Sure Thing** | `uplift ≤ 0.05` AND `p_control > 0.5` | No discount — buys anyway |
| **Lost Cause** | `uplift ≤ 0.05` AND `p_control ≤ 0.5` | No discount — won't convert |

---

### Expected Profit Optimization

For each Persuadable customer, all five discount tiers are evaluated:

```
E[Profit] = P(buy | discount) × (Gross Margin − Discount Amount) − Offer Cost

where:
  P(buy | discount) = min(p_base + 1.5 × discount, 0.99)
  Gross Margin      = revenue_proxy × (1 − 0.40)
  Offer Cost        = $0.50
```

Parameters grounded in literature:
- Elasticity = 1.5 → conservative mid-range (Neslin et al., 2006: online retail elasticities 1.5–3.0)
- Cost ratio = 40% → Shopify industry benchmarks (35–45%)
- Offer cost = $0.50 → Klaviyo (2023) email/push campaign unit cost
- Revenue proxy = PageValues scaled to $20–$200 (Statista 2023: global avg AOV ~$80–$120)

The optimization is **vectorized over all discount tiers** using NumPy broadcasting — no `iterrows` loops.

---

### Qini Curve Evaluation

Since ground-truth individual treatment effects are unobservable, standard metrics (AUC, F1) cannot evaluate uplift models. The **Qini curve** (Radcliffe, 2007) is the standard alternative:

- Sort customers by descending predicted uplift
- Walk down the ranked list, cumulatively tracking incremental conversions in treated vs. control
- Area between model curve and random baseline = **Qini coefficient**

Both T-Learner and X-Learner are evaluated. The pipeline prints both coefficients and automatically routes Stage 2 to the winner.

---

### Monte Carlo A/B Simulation

Since a live experiment is not feasible, the evaluation simulates 1,000 random targeting runs where the same proportion of customers as our model targets are selected *randomly* with a flat 10% discount. All three strategies are compared over the full dataset:

| Strategy | Scope |
|---|---|
| No Discount | All customers at natural purchase rate |
| Random Targeting | `n_target` random customers at 10% discount (1,000 runs, 95% CI reported) |
| Our Model | Persuadables at personalized optimal discount; rest at natural rate |

The percentage of simulation runs where our model outperforms random targeting is reported directly.

---

## 📊 Results

```
Base Model:
  ROC-AUC             : 0.9244
  F1 Score            : 0.6385
  Calibration Error   : 0.024  ✓ well calibrated

Uplift Model Comparison:
  T-Learner Qini      : (see notebook output)
  X-Learner Qini      : (see notebook output)
  Winner selected automatically

Profit Results (full dataset):
  No Discount         : $31,494
  Random Targeting    : $35,240  (avg)
  Our Model           : $39,843
  Gain vs Baseline    : +$8,349
  Model beats random  : reported per run
```

Outputs saved:
- `discount_pipeline_results.png` — 9-panel pipeline summary visualization
- `qini_curve.png` — T-Learner vs X-Learner Qini comparison
- `ab_simulation.png` — Monte Carlo strategy comparison
- `customer_discount_recommendations.csv` — per-customer optimal discount table

---

## 📁 Repository Structure

```
discount-uplift-engine/
│
├── test3_X-Learner.ipynb               # Main pipeline notebook
├── online_shoppers_intention.csv       # Dataset (Sakar & Kastro, 2019)
│
├── outputs/
│   ├── customer_discount_recommendations.csv
│   ├── discount_pipeline_results.png
│   ├── qini_curve.png
│   └── ab_simulation.png
│
└── README.md
```

---

## ⚙️ Requirements

```
python >= 3.9
pandas
numpy
scikit-learn >= 1.2
matplotlib
seaborn
```

Install:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run the pipeline by executing all cells in `test3_X-Learner.ipynb` in order. The dataset CSV must be in the same directory.

---

## 📚 References

| Citation | Used for |
|---|---|
| Rubin (1974) | Potential Outcomes framework / causal inference foundations |
| Rosenbaum & Rubin (1983) | Observational study treatment assignment |
| Radcliffe & Surry (1999) | Uplift modeling taxonomy (Persuadable / Sure Thing / Lost Cause) |
| Radcliffe (2007) | Qini curve formulation |
| Niculescu-Mizil & Caruana (2005) | Probability calibration (Platt/Isotonic) |
| Künzel et al. (2019) | T-Learner & X-Learner metalearners |
| Kohavi et al. (2020) | A/B simulation methodology |
| Neslin et al. (2006) | Price elasticity in online promotions |
| Sakar & Kastro (2019) | Online Shoppers Purchasing Intention dataset |

---

## 👥 Authors

Group project — MSc Data Science, NMIMS NSOMASA  
Subject: Research Discourse (RD), Semester II
