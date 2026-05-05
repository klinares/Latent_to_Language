# ============================================================
# 03_traditional_item_imputation.R
# L2L Framework — Traditional Methods for Item Imputation
#
# Evaluates mixgb and miceRanger for item-level imputation
# using the SAME masking table as Script 04, enabling direct
# apples-to-apples comparison with LLM results.
#
# Tiers:
#   S  : screened demographics + observed trust items (masked NA)
#   P1 : S + CFA factor score (from partial observed items)
#   P2 : S + IRT theta (from partial observed items, no SE)
#
# Methods:
#   - mixgb: XGBoost multiple imputation (m=5, majority vote)
#   - miceRanger: Random forest MI (m=5, majority vote)
#
# Design:
#   - Trust items coded as factors for classification
#   - Combined data: complete training + masked test rows
#   - Psychometric features computed from partial data only
#   - CFA + IRT baselines computed for all conditions
#   - Demographics screened by Script 02 (eta² threshold)
#
# Prerequisites:
#   - fold_data.RData, cfa_reporting.RData, irt_models.RData
#   - masking_table.rds (from Script 02)
# ============================================================

# ── Load packages (tidyverse last per project convention) ────
library(lavaan)
library(mirt)
library(mixgb)
library(miceRanger)
library(tictoc)
library(tidyverse)

set.seed(721)


# ══════════════════════════════════════════════════════════════
# PATHS AND DATA
# ══════════════════════════════════════════════════════════════

results_dir <- "D:/repos/UMD_classes_code/TSE_II_SURV721/results"

# Load pre-computed objects from Script 02
load(file.path(results_dir, "fold_data.RData"))
load(file.path(results_dir, "cfa_reporting.RData"))
load(file.path(results_dir, "irt_models.RData"))

# Load shared masking table
masking_tbl <- readRDS(file.path(results_dir, "masking_table.rds"))
cat("Loaded masking table:", nrow(masking_tbl), "rows\n")
masking_tbl |> count(survey, n_masked) |> print()


# ══════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════

# Number of MI datasets per method
M_IMPS <- 5L

# Tiers to evaluate
TIERS <- c("S", "P1", "P2")


# ══════════════════════════════════════════════════════════════
# SURVEY CONFIGURATION
# ══════════════════════════════════════════════════════════════
#
# Demographics hardcoded from Script 02 screening results:
#   LAPOP: Edu, age_cat, Employment, Urban (male dropped)
#   LB:    age_cat, Edu, Employment (male + Urban dropped)
# ──────────────────────────────────────────────────────────────

config <- list(
  lapop = list(
    # Screened demographics (eta² >= 0.005 with theta or CFA fs)
    demo_vars       = c("age_cat", "Edu", "Urban", "Employment"),
    trust_items     = c(
      "justice_system", "electoral_tribunal", "armed_forces",
      "legislature", "public_ministry", "police", "auditor",
      "political_parties", "supreme_court", "municipality",
      "media", "elections"
    ),
    factor_names    = "Trust",
    trust_ascending = TRUE,
    scale_levels    = as.character(1:7),
    cfa_syntax = '
      Trust =~ justice_system + electoral_tribunal + armed_forces +
               legislature + public_ministry + police + auditor +
               political_parties + supreme_court + municipality +
               media + elections
      armed_forces ~~ legislature
      public_ministry ~~ auditor
    '
  ),
  lb = list(
    # Screened demographics (eta² >= 0.005 with theta or CFA fs)
    demo_vars       = c("age_cat", "Edu", "Employment"),
    trust_items     = c(
      "Congress", "Natl_Govt", "Judiciary", "Parties",
      "Electoral_Inst", "President"
    ),
    factor_names    = "Political",
    trust_ascending = FALSE,
    scale_levels    = as.character(1:4),
    cfa_syntax = '
      Political =~ Congress + Natl_Govt + Judiciary + Parties +
                   Electoral_Inst + President
      Natl_Govt ~~ President
    '
  )
)


# ══════════════════════════════════════════════════════════════
# HELPER: MAJORITY VOTE
# ══════════════════════════════════════════════════════════════
#
# Given a vector of m predictions (character or integer),
# returns the most frequent value as integer. Ties broken
# by which.max (first mode).
# ──────────────────────────────────────────────────────────────

majority_vote <- function(votes) {
  tab <- table(votes)
  as.integer(names(tab)[which.max(tab)])
}


# ══════════════════════════════════════════════════════════════
# COMPUTE FOLD CONTEXT
# ══════════════════════════════════════════════════════════════
#
# For a given fold: refit CFA (needed for partial lavPredict),
# load saved IRT model, and compute partial-data psychometric
# features for each test respondent's masking pattern.
#
# Returns per-respondent: CFA factor score (partial), IRT theta
# (partial, no SE), CFA baseline predictions, IRT baseline
# predictions (modal category from probtrace).
# ──────────────────────────────────────────────────────────────

compute_fold_context <- function(fold_i, survey_name, cfg,
                                 test_df, fold_mask) {

  train_df <- folds_list[[survey_name]] |>
    filter(fold_id == fold_i, split == "train")

  trust_train <- train_df |>
    select(all_of(cfg$trust_items)) |>
    mutate(across(everything(), as.numeric))

  item_min <- min(unlist(trust_train), na.rm = TRUE)
  item_max <- max(unlist(trust_train), na.rm = TRUE)

  # ── CFA: refit on training data for partial lavPredict ───
  cfa_fit <- cfa(
    cfg$cfa_syntax,
    data          = trust_train,
    estimator     = "ML",
    meanstructure = TRUE,
    missing       = "fiml"
  )

  params_df <- parameterEstimates(cfa_fit)
  load_df   <- params_df |> filter(op == "=~")
  intcpt_df <- params_df |> filter(op == "~1",
                                   lhs %in% cfg$trust_items)

  # CFA training factor scores for the combined dataset
  fs_cfa_train <- as.numeric(
    lavPredict(cfa_fit, newdata = trust_train,
               method = "regression")
  )

  # ── IRT: load saved model (not refit) ────────────────────
  irt_fit <- irt_models[[survey_name]][[fold_i]]

  # IRT training thetas for the combined dataset
  train_irt_theta <- irt_list[[survey_name]] |>
    filter(fold_id == fold_i, split == "train") |>
    pull(irt_theta)

  # ── Per-respondent psychometric profiles ─────────────────
  respondent_info <- map(seq_len(nrow(test_df)), function(i) {
    rid <- test_df$row_id[i]
    mask_info <- fold_mask |> filter(row_id == rid)
    masked_items   <- mask_info$masked_item[[1]]
    observed_items <- setdiff(cfg$trust_items, masked_items)

    # CFA factor score from partial data (masked items = NA)
    test_items_cfa <- test_df[i, ] |>
      select(all_of(cfg$trust_items)) |>
      mutate(across(everything(), as.numeric)) |>
      mutate(across(all_of(masked_items), function(x) NA_real_))

    fs_cfa <- as.numeric(
      lavPredict(cfa_fit, newdata = test_items_cfa,
                 method = "regression")
    )

    # CFA baseline predictions (expected value from loadings)
    cfa_preds <- map_int(masked_items, function(item) {
      ld  <- load_df$est[load_df$rhs == item]
      int <- intcpt_df$est[intcpt_df$lhs == item]
      raw <- int + ld * fs_cfa
      as.integer(round(pmax(item_min, pmin(item_max, raw))))
    }) |> set_names(masked_items)

    # IRT theta from partial data (reverse-code for LB)
    partial_vec <- as.numeric(test_df[i, cfg$trust_items])
    if (!cfg$trust_ascending) {
      partial_vec <- as.integer(item_max + 1L - partial_vec)
    }
    partial_vec[match(masked_items, cfg$trust_items)] <- NA

    theta_out <- fscores(
      irt_fit,
      response.pattern = matrix(partial_vec, nrow = 1),
      method = "EAP",
      full.scores.SE = TRUE
    )
    irt_theta <- as.numeric(theta_out[, 1])

    # IRT baseline: modal category from probtrace
    irt_probs <- map(masked_items, function(item) {
      item_obj  <- extract.item(irt_fit, item)
      raw_probs <- as.numeric(probtrace(item_obj, irt_theta))
      if (!cfg$trust_ascending) rev(raw_probs) else raw_probs
    }) |> set_names(masked_items)

    irt_preds <- map_int(masked_items, function(item) {
      probs     <- irt_probs[[item]]
      modal_idx <- which.max(probs)
      as.integer(modal_idx + item_min - 1L)
    }) |> set_names(masked_items)

    list(
      rid          = rid,
      masked_items = masked_items,
      cfa_fs       = fs_cfa,
      irt_theta    = irt_theta,
      cfa_preds    = cfa_preds,
      irt_preds    = irt_preds
    )
  })

  list(
    respondent_info = respondent_info,
    train_cfa_fs    = fs_cfa_train,
    train_irt_theta = train_irt_theta
  )
}


# ══════════════════════════════════════════════════════════════
# BUILD COMBINED DATA FOR IMPUTATION
# ══════════════════════════════════════════════════════════════
#
# Creates the train+test combined data frame with trust items
# as factors. Test rows have NAs for their masked items.
# Tier controls which psychometric features are added:
#   S  = demographics + trust items only
#   P1 = + CFA factor score
#   P2 = + IRT theta (no SE)
# ──────────────────────────────────────────────────────────────

build_imputation_data <- function(train_df, test_df, cfg, tier,
                                  fold_context, fold_mask) {

  trust_items  <- cfg$trust_items
  demo_vars    <- cfg$demo_vars
  scale_levels <- cfg$scale_levels

  # ── Training data: all trust items complete ────────────
  train_trust <- train_df |>
    select(all_of(trust_items)) |>
    mutate(across(everything(),
                  function(x) factor(as.integer(as.numeric(x)),
                                     levels = scale_levels)))

  train_demos <- train_df |> select(all_of(demo_vars))
  train_combined <- bind_cols(train_demos, train_trust)

  # ── Test data: per-respondent masking ──────────────────
  test_trust <- test_df |>
    select(all_of(trust_items)) |>
    mutate(across(everything(),
                  function(x) factor(as.integer(as.numeric(x)),
                                     levels = scale_levels)))

  # Apply masking: set masked items to NA per respondent
  resp_info <- fold_context$respondent_info
  walk(seq_len(nrow(test_df)), function(i) {
    masked <- resp_info[[i]]$masked_items
    walk(masked, function(item) {
      test_trust[[item]][i] <<- NA
    })
  })

  test_demos <- test_df |> select(all_of(demo_vars))
  test_combined <- bind_cols(test_demos, test_trust)

  # ── Tier-specific features ─────────────────────────────
  if (tier == "P1") {
    # CFA factor score: full-data for train, partial for test
    train_combined <- train_combined |>
      mutate(cfa_fs = fold_context$train_cfa_fs)

    test_cfa_fs <- map_dbl(resp_info, "cfa_fs")
    test_combined <- test_combined |>
      mutate(cfa_fs = test_cfa_fs)
  }

  if (tier == "P2") {
    # IRT theta only (no SE) — full-data for train, partial
    # for test
    train_combined <- train_combined |>
      mutate(irt_theta = fold_context$train_irt_theta)

    test_irt_theta <- map_dbl(resp_info, "irt_theta")
    test_combined <- test_combined |>
      mutate(irt_theta = test_irt_theta)
  }

  # ── Combine train + test ───────────────────────────────
  n_train <- nrow(train_combined)
  combined <- bind_rows(train_combined, test_combined)
  test_idx <- (n_train + 1):nrow(combined)

  list(combined = combined, test_idx = test_idx)
}


# ══════════════════════════════════════════════════════════════
# IMPUTATION RUNNERS
# ══════════════════════════════════════════════════════════════

# Run mixgb imputation and return list of m imputed test-row
# tibbles (trust items only)
run_mixgb_items <- function(combined, test_idx,
                            trust_items, m = 5L) {
  imputed_list <- tryCatch(
    mixgb(data = combined, m = m, verbose = FALSE),
    error = function(e) {
      cat("    mixgb ERROR:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(imputed_list)) return(NULL)

  # Extract imputed trust items for test rows from each dataset
  map(seq_len(m), function(i) {
    as_tibble(imputed_list[[i]])[test_idx, trust_items,
                                  drop = FALSE]
  })
}

# Run miceRanger imputation and return list of m imputed
# test-row tibbles (trust items only)
run_miceranger_items <- function(combined, test_idx,
                                 trust_items, m = 5L) {
  # Only impute columns that actually have NAs
  cols_with_na <- names(combined)[colSums(is.na(combined)) > 0]
  vars_to_impute <- intersect(trust_items, cols_with_na)

  if (length(vars_to_impute) == 0) return(NULL)

  miced <- tryCatch(
    miceRanger(combined, vars = vars_to_impute, m = m,
               verbose = FALSE),
    error = function(e) {
      cat("    miceRanger ERROR:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(miced)) return(NULL)

  # Extract imputed trust items for test rows from each dataset
  map(seq_len(m), function(i) {
    completed <- as_tibble(completeData(miced, datasets = i)[[1]])
    completed[test_idx, trust_items, drop = FALSE]
  })
}


# ══════════════════════════════════════════════════════════════
# EXTRACT PREDICTIONS (majority vote across m imputations)
# ══════════════════════════════════════════════════════════════

extract_item_predictions <- function(imputed_datasets, test_df,
                                     resp_info, method_name,
                                     survey_name, fold_i,
                                     tier) {

  # Handle imputation failure: return all-NA results
  if (is.null(imputed_datasets)) {
    return(map_dfr(seq_len(nrow(test_df)), function(j) {
      info <- resp_info[[j]]
      map_dfr(info$masked_items, function(item) {
        tibble(
          survey       = survey_name,
          fold_id      = fold_i,
          n_masked     = length(info$masked_items),
          tier         = tier,
          method       = method_name,
          row_id       = info$rid,
          item         = item,
          actual       = as.integer(as.numeric(test_df[[item]][j])),
          predicted    = NA_integer_,
          cfa_baseline = info$cfa_preds[[item]],
          irt_baseline = info$irt_preds[[item]]
        )
      })
    }))
  }

  m <- length(imputed_datasets)

  # Extract majority-vote prediction for each masked item
  map_dfr(seq_len(nrow(test_df)), function(j) {
    info <- resp_info[[j]]
    map_dfr(info$masked_items, function(item) {
      # Collect votes across m imputations
      votes <- map_int(seq_len(m), function(i) {
        val <- imputed_datasets[[i]][[item]][j]
        as.integer(as.character(val))
      })
      pred <- majority_vote(votes)

      tibble(
        survey       = survey_name,
        fold_id      = fold_i,
        n_masked     = length(info$masked_items),
        tier         = tier,
        method       = method_name,
        row_id       = info$rid,
        item         = item,
        actual       = as.integer(as.numeric(test_df[[item]][j])),
        predicted    = pred,
        cfa_baseline = info$cfa_preds[[item]],
        irt_baseline = info$irt_preds[[item]]
      )
    })
  })
}


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

cat("\n", strrep("=", 60), "\n")
cat("TRADITIONAL METHODS — ITEM IMPUTATION (S/P1/P2, 5-FOLD)")
cat("\n", strrep("=", 60), "\n\n")
tic("Total traditional")

results_tbl <- map_dfr(names(config), function(survey_name) {
  cfg <- config[[survey_name]]

  cat(sprintf("\n== %s ====================\n", survey_name))
  cat(sprintf("   Demographics: [%s]\n",
              str_c(cfg$demo_vars, collapse = ", ")))

  map_dfr(1:5, function(fold_i) {

    train_df <- folds_list[[survey_name]] |>
      filter(fold_id == fold_i, split == "train") |>
      arrange(row_id)

    test_df <- folds_list[[survey_name]] |>
      filter(fold_id == fold_i, split == "test") |>
      arrange(row_id)

    fold_mask <- masking_tbl |>
      filter(survey == survey_name, fold_id == fold_i,
             row_id %in% test_df$row_id)

    cat(sprintf("\n  Fold %d | n_train=%d, n_test=%d\n",
                fold_i, nrow(train_df), nrow(test_df)))

    # Compute psychometric baselines (shared across tiers)
    fold_context <- compute_fold_context(
      fold_i, survey_name, cfg, test_df, fold_mask)

    resp_info <- fold_context$respondent_info

    # ── Iterate over tiers ────────────────────────────────
    map_dfr(TIERS, function(tier) {

      cat(sprintf("    Tier %s: ", tier))
      tic()

      # Build combined train+test data for this tier
      imp_data <- build_imputation_data(
        train_df, test_df, cfg, tier,
        fold_context, fold_mask)

      # ── mixgb ──────────────────────────────────────────
      cat("mixgb...")
      mixgb_imp <- run_mixgb_items(
        imp_data$combined, imp_data$test_idx,
        cfg$trust_items, m = M_IMPS)

      mixgb_res <- extract_item_predictions(
        mixgb_imp, test_df, resp_info, "mixgb",
        survey_name, fold_i, tier)

      # ── miceRanger ─────────────────────────────────────
      cat(" miceRanger...")
      mr_imp <- run_miceranger_items(
        imp_data$combined, imp_data$test_idx,
        cfg$trust_items, m = M_IMPS)

      mr_res <- extract_item_predictions(
        mr_imp, test_df, resp_info, "miceRanger",
        survey_name, fold_i, tier)

      elapsed <- toc(quiet = TRUE)
      cat(sprintf(" done (%.1fs)\n",
                  elapsed$toc - elapsed$tic))

      bind_rows(mixgb_res, mr_res)
    })
  })
})

toc()


# ══════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════

cat("\n", strrep("=", 60), "\n")
cat("TRADITIONAL METHODS — RESULTS (S/P1/P2)\n")
cat(strrep("=", 60), "\n\n")

# ── Success rates ──
cat("-- Success Rate ------\n")
results_tbl |>
  mutate(parsed_ok = !is.na(predicted)) |>
  group_by(method, survey, tier) |>
  summarise(
    n_items      = n(),
    n_predicted  = sum(parsed_ok),
    success_rate = round(mean(parsed_ok), 3),
    .groups      = "drop"
  ) |>
  arrange(method, survey, tier) |>
  print(n = Inf)

# ── Accuracy by method × tier × survey ──
cat("\n-- Accuracy: Traditional vs CFA vs IRT Baselines ---\n")
results_tbl |>
  filter(!is.na(predicted)) |>
  group_by(method, survey, tier) |>
  summarise(
    accuracy     = round(mean(predicted == actual), 3),
    cfa_accuracy = round(mean(cfa_baseline == actual), 3),
    irt_accuracy = round(mean(irt_baseline == actual), 3),
    mae          = round(mean(abs(predicted - actual)), 3),
    n            = n(),
    .groups      = "drop"
  ) |>
  arrange(survey, tier, method) |>
  print(n = Inf)

# ── Degradation by number of masked items ──
cat("\n-- Degradation by n_masked -----\n")
results_tbl |>
  filter(!is.na(predicted)) |>
  group_by(method, survey, tier, n_masked) |>
  summarise(
    accuracy = round(mean(predicted == actual), 3),
    irt_acc  = round(mean(irt_baseline == actual), 3),
    n        = n(),
    .groups  = "drop"
  ) |>
  arrange(survey, tier, n_masked, method) |>
  print(n = Inf)

# ── Per-item accuracy ──
cat("\n-- Per-Item Accuracy --------\n")
results_tbl |>
  filter(!is.na(predicted)) |>
  group_by(method, survey, tier, item) |>
  summarise(
    accuracy = round(mean(predicted == actual), 3),
    irt_acc  = round(mean(irt_baseline == actual), 3),
    n        = n(),
    .groups  = "drop"
  ) |>
  arrange(survey, tier, item, method) |>
  print(n = Inf)


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

output_path <- file.path(results_dir, "results_trad_item.rds")
saveRDS(results_tbl, output_path)
cat("\nSaved:", output_path, "\n")

cat("\n", strrep("=", 60), "\n")
cat("TRADITIONAL METHODS COMPLETE (S/P1/P2)\n")
cat(strrep("=", 60), "\n")
