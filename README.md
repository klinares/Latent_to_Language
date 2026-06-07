# Latent-to-Language (L2L) Framework

**Can Psychometric-Informed Prompts Improve LLM-Based Survey Item Imputation?**

Kevin Linares \| Joint Program in Survey Methodology, University of Maryland

Paper -> [Latent-to-Language Manuscript](https://github.com/klinares/Latent_to_Language/blob/main/manuscript/final_paper_linares.pdf)

![](images/clipboard-3237513583.png)

## Overview

The L2L framework translates psychometric model *conclusions* — not raw parameters — into natural language prompts for Large Language Models, enabling psychometrically-informed imputation of missing survey item responses within a Total Survey Error framework.

The core innovation is the **model-then-adjust paradigm**: the IRT or CFA model provides a starting-point prediction for each masked item, and the LLM refines it using information the statistical model cannot capture — within-respondent consistency (residual patterns), correlated-item signals, demographic probability context, and domain knowledge about institutional trust. Prompts communicate model conclusions (predictions, confidence levels, adjustment directions) rather than raw psychometric parameters (discrimination values, thresholds, factor loadings), because LLMs are reasoners, not calculators.

Using two nationally representative surveys — **LAPOP Mexico 2023** (N = 1,430; 12-item institutional trust scale, 7-point) and **Latinobarómetro Colombia 2023** (N = 1,073; 6-item institutional distrust scale, 4-point) — we compare four imputation methods (two tree-based benchmarks, two LLMs) across three information tiers in a **5-fold cross-validation** experiment with per-fold CFA refitting, IRT/GRM estimation, demographic profiling, and m=5 multiple imputation replication.

## Key Findings

- **Item-level — psychometric context reduces item-level error monotonically**: Claude Sonnet weighted kappa rises .645 (S) → .667 (P1) → .675 (P2) on LAPOP and .550 → .559 → .579 on Latinobarómetro. At the fair comparison (S tier), Claude Sonnet (.645 LAPOP, .550 LB) outperforms mixgb (.615, .463) and miceRanger (.561, .504).
- **Item-level — the psychometric anchor helps most when the task is hardest**: under heavy masking (3–4 of 6 items) on the short Latinobarómetro scale, Claude Sonnet P2 declines only modestly (.612 → .567) while mixgb S collapses (.604 → .399, a 34% degradation).
- **Person-level — all methods preserve ordering, but only the LLM preserves scale**: correlations with ground-truth θ exceed .98 on LAPOP (Claude .989 at S, .988 at P2); on the shorter LB scale Claude (.930) beats mixgb (.895). Claude P2 holds the regression slope near 1.0 on both surveys, while mixgb S compresses the LB latent distribution by 11% (slope 0.89).
- **Instrument-level — both methods inflate IRT discrimination, with structurally different error**: every item shows positive Δa (LAPOP mean +0.09 Claude P2 / +0.05 mixgb S; LB +0.20 / +0.05). Absolute distortion is small (\< 5% relative on LAPOP), but Claude P2's error is predominantly systematic (89% LAPOP, 86% LB Bias²) while mixgb S is more balanced (75%, 33%); deletion adds near-zero bias but large variance.
- **Complementary error motivates stacking, not competition**: because LLM and tree-based methods carry structurally different error profiles, using L2L as a psychometric context layer over traditional imputation is more promising than choosing one method over the other.
- **Prior-attitude interference**: items where LLM world-knowledge priors conflict with population attitudes (armed_forces) show systematic directional bias — an LLM-specific bias mechanism and a caution for cross-national use.

## Research Questions

Within the Total Survey Error framework, imputation is a processing decision whose consequences register as **measurement error**. The three research questions trace that error across three levels of analysis:

1.  **RQ1 — Item-level measurement error**: To what extent does communicating progressively richer psychometric model information reduce item-level prediction error, and how does performance degrade as the proportion of masked items within a scale increases?
2.  **RQ2 — Person-level measurement error**: How well do imputed data preserve respondents' positions on the latent construct, as measured by the correlation and regression slope between ground-truth and imputed θ estimates?
3.  **RQ3 — Instrument-level measurement error**: To what extent does imputation alter the psychometric properties of the measurement instrument, and do LLM-based and tree-based methods introduce structurally different distortion profiles?

## Tier Design

### LLM Tiers

| Tier | Information | Paradigm |
|:-----------------|:-----------------------------|:-----------------------|
| **S** | Screened demographics + observed items (verbal labels) | World knowledge only |
| **P1** | S + CFA factor score (percentile) + CFA prediction anchor + 2 correlated items with CFA residuals | Model-then-adjust (CFA) |
| **P2** | S + IRT modal prediction with confidence + 2 correlated items with IRT residuals + demographic probability context P(high trust \| group) | Model-then-adjust (IRT) |

### Traditional Method Tiers

| Tier   | Features                                              |
|:-------|:------------------------------------------------------|
| **S**  | Screened demographics + observed items (masked as NA) |
| **P1** | S + CFA factor score (numeric column)                 |
| **P2** | S + IRT θ (numeric column)                            |

Note: Traditional methods at P1/P2 also see the complete training response matrix, giving them an information advantage over LLMs. The fair comparison is LLM tiers against traditional methods at S. Traditional P1/P2 results are reported in the appendix.

### Demographic Screening (Script 02)

Demographics screened using η² with both IRT theta and CFA factor score (threshold ≥ 0.005). Survivors per survey:

- **LAPOP**: age_cat, Edu, Urban, Employment (male dropped)
- **LB**: age_cat, Edu, Employment (male + Urban dropped)

## Repository Structure

```         
├── code/
│   ├── 01_data_preparation.qmd          # EFA/CFA, variable recoding, listwise deletion
│   ├── 02_kfold_cfa_irt.qmd             # 5-fold CV, CFA, IRT/GRM, demographic profiling,
│   │                                    #   masking table generation
│   ├── 03_traditional_item_imputation.R  # mixgb + miceRanger on S/P1/P2 tiers (m=5)
│   ├── 04_LLM_item_imputation.R         # LLM item imputation (model-then-adjust, m=5 MI,
│   │                                    #   graduated temperature, parallel batching)
│   ├── 05_item_results.qmd              # RQ evaluation, CFA/IRT visuals, TSE analysis
│   ├── item_prompt_functions.R           # Prompt construction + response parsing
│   ├── prompt_engineering.qmd            # Prompt design documentation + decision log
│   └── references.bib                    # Bibliography
│
├── results/                              # Generated by scripts (not tracked)
│   ├── survey_data.rds                   # Prepared data (list: lapop, lb)
│   ├── fold_data.RData                   # folds_list, cfa_list, irt_list, masking_tbl
│   ├── cfa_reporting.RData               # CFA/IRT params, demo_profile_tbl
│   ├── irt_models.RData                  # Fitted mirt model objects per fold
│   ├── results_trad_item.rds             # Traditional method results (S/P1/P2)
│   ├── results_item.rds                  # LLM majority-vote results (m=5)
│   ├── results_item_raw_runs.rds         # LLM per-run results (for MI variance)
│   ├── all_results_combined.rds          # Combined traditional + LLM results
│   └── evaluation_rq.RData              # Pre-computed RQ evaluation tables for manuscript
│
└── README.md
```

## Pipeline

```         
Script 01 (Data Preparation)
    │
    ▼
Script 02 (5-Fold CV + Psychometric Infrastructure)
    │   CFA, IRT/GRM, demographic profiling (η² screening,
    │   group means, model-based probabilities), masking table
    │
    ├──────────────────────────┐
    ▼                          ▼
Script 03 (Traditional)    Script 04 (LLM Imputation)
  mixgb + miceRanger         Claude Sonnet, Maverick
  S / P1 / P2 tiers          S / P1 / P2 tiers
  m=5, majority vote          m=5 MI (temp 0.01–0.40), majority vote
  Screened demographics       Model-then-adjust paradigm
    │                          │
    └──────────┬───────────────┘
               ▼
         Script 05 (Results)
           RQ1 (item-level): Weighted kappa, accuracy by tier
           RQ2 (person-level): θ correlation, regression slope
           RQ3 (instrument-level): Biemer decomposition of IRT discrimination (Δa)
           Item variance shrinkage, per-item analysis
```

## Prompt Design (V2)

The prompt architecture communicates model **conclusions**, not parameters:

| What the model computes | What the LLM sees |
|:---------------------------------------|:-------------------------------|
| IRT modal category + probabilities | "Model prediction: 5. Confidence: MODERATE (5 at 35%, 4 at 22%)" |
| IRT residuals for observed items | "police = 6 (model expected 5.2, +0.8 above → suggests this item should also be ABOVE model)" |
| Mean residual across items | "This respondent consistently rates ABOVE the model (+0.5 avg) → adjust predictions UPWARD" |
| P(high trust \| demographic group) | "For your group, P(high trust on armed_forces) = 48% vs. population 43%" |

### What was dropped from V1

- IRT discrimination/threshold tables
- CFA standardized loadings
- Q3 residual correlation matrices
- Conditional expectations E[X_miss \| X_obs]
- Chain-of-thought reasoning

### Psychometrician Persona

System prompt establishes a JPSM PhD survey psychometrician with expertise in institutional trust measurement. Key reasoning rules: residual pattern is the strongest signal; halo effects mean consistent respondents stay consistent; demographic context is a sanity check, not the primary signal.

## Multiple Imputation Design

LLM predictions are treated as m=5 multiple imputations with graduated temperature:

| Run | Temperature |
|-----|-------------|
| 1   | 0.01        |
| 2   | 0.10        |
| 3   | 0.20        |
| 4   | 0.30        |
| 5   | 0.40        |

Majority vote across 5 runs yields the final prediction, matching the m=5 majority-vote design used for traditional methods. Checkpointing per run enables resume after interruption.

## Models

| Model | Provider | Batch Size | Parallel | Role |
|:--------------|:--------------|:--------------|:--------------|:--------------|
| Claude Sonnet 4.6 | Anthropic API | 50 | 10 | Strongest LLM (production) |
| Llama 4 Maverick | OpenRouter | 25 | 2 | Second-tier LLM |
| mixgb | R package | — | — | Tree-based benchmark (XGBoost) |
| miceRanger | R package | — | — | Tree-based benchmark (Random forest) |

Gemma 3 27B was evaluated but dropped from the final analysis.

## Evaluation Framework

### RQ1 Metrics (Item-Level Measurement Error)

| Metric | Description |
|:---------------------------|:-------------------------------------------|
| Weighted kappa | Quadratic weighted kappa (primary); penalizes larger ordinal distances |
| Accuracy | Exact match rate (secondary) |

### RQ2 Metrics (Person-Level Measurement Error)

| Metric | Description |
|:---------------------------|:-------------------------------------------|
| Pearson r | Correlation between ground truth θ and imputed θ; captures rank preservation |
| Regression slope | Slope of θ_imputed on θ_truth; 1.0 = perfect, \< 1.0 = centripetal compression |

### RQ3 Metrics (Instrument-Level Measurement Error)

| Metric | Description |
|:---------------------------|:-------------------------------------------|
| Biemer decomposition of discrimination | MSE of Δa (= a_imputed − a_gold) = Bias² + Variance per item; ratio quantifies systematic vs. random distortion of the measurement model |
| Item variance shrinkage | Change in item response variance after imputation; negative = compression of the response spread |

Pairwise method comparisons use Nadeau-Bengio corrected t-tests.

## Requirements

- **R** ≥ 4.3 with `lavaan`, `mirt`, `semTools`, `tidySEM`, `mixgb`, `miceRanger`, `ellmer`, `furrr`, `viridis`, `gridExtra`, `tidyverse`, `tictoc`
- **Quarto** ≥ 1.4 + LaTeX distribution
- **API keys**: Anthropic (`ANTHROPIC_API_KEY`), OpenRouter (`OPENROUTER_API_KEY`) in `.Renviron`
- **Data**: [LAPOP 2023](https://www.vanderbilt.edu/lapop/) and [Latinobarómetro 2023](https://www.latinobarometro.org/)

## Citation

``` bibtex
@article{linares2026l2l,
  author  = {Linares, Kevin},
  title   = {Latent-to-Language: A Psychometric Context Engineering Framework
             for {LLM}-Based Survey Item Imputation},
  journal = {SURV 721 Course Paper, JPSM, University of Maryland},
  year    = {2026}
}
```

