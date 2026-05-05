# ============================================================
# 04_LLM_item_imputation.R
# L2L Framework — LLM Item Imputation (Model-Then-Adjust)
#
# Paradigm: The psychometric model (CFA/IRT) provides a
# starting-point prediction. The LLM adjusts based on
# respondent consistency, demographic context, and domain
# knowledge — NOT raw psychometric parameters.
#
# Prompt design principles:
#   - Communicate model CONCLUSIONS, not parameters
#   - LLM is a reasoner, not a calculator
#   - Within-respondent consistency > demographic averages
#   - 2 correlated items per masked item (residual as signal)
#   - Demographic context as P(high trust | group) from IRT
#
# Models: Maverick, Gemma 27B, Claude Sonnet
# 5-fold CV, batch=25, parallel workers per batch
# ============================================================

# ── Load packages (tidyverse last) ───────────────────────────
library(ellmer)
library(furrr)
library(lavaan)
library(mirt)
library(tictoc)
library(tidyverse)

set.seed(721)


# ══════════════════════════════════════════════════════════════
# PATHS AND DATA
# ══════════════════════════════════════════════════════════════

code_dir    <- "D:/repos/UMD_classes_code/TSE_II_SURV721/code"
results_dir <- "D:/repos/UMD_classes_code/TSE_II_SURV721/results"

# Source prompt construction functions
source(file.path(code_dir, "item_prompt_functions.R"))

# Load Script 02 outputs
load(file.path(results_dir, "fold_data.RData"))
load(file.path(results_dir, "cfa_reporting.RData"))
load(file.path(results_dir, "irt_models.RData"))

cat("Loaded all Script 02 outputs.\n")
cat("Masking table:", nrow(masking_tbl), "rows\n")


# ══════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════

MAX_RETRIES   <- 3L
TIERS         <- c("S", "P1", "P2")

# ── MI replication: m=5 runs with graduated temperatures ─────
# Each fold × tier × model is run 5 times at increasing
# temperatures. Majority vote across runs yields the final
# prediction. This mirrors traditional MI (m=5) and recovers
# from parse failures across runs.
M_IMPS        <- 5L
TEMP_SCHEDULE <- c(0.01, 0.10, 0.20, 0.30, 0.40)

# ── Fold selection (set to 1 for testing, 1:5 for production) ─
RUN_FOLDS     <- 1:5

# ── Model configuration ─────────────────────────────────────
RUN_MAVERICK  <- FALSE
RUN_GEMMA_27B <- TRUE
RUN_CLAUDE    <- FALSE

LLM_MODELS <- list(
  llama4_maverick = list(
    provider     = "openrouter",
    model_id     = "meta-llama/llama-4-maverick",
    max_tokens   = 4000L,
    batch_size   = 25L,
    max_parallel = 6L,
    enabled      = RUN_MAVERICK
  ),
  gemma3_27b = list(
    provider     = "openrouter",
    model_id     = "google/gemma-4-26b-a4b-it",
    max_tokens   = 4000L,
    batch_size   = 25L,
    max_parallel = 6L,
    enabled      = RUN_GEMMA_27B
  ),
  claude_sonnet = list(
    provider     = "anthropic",
    model_id     = "claude-sonnet-4-6",
    max_tokens   = 4000L,
    batch_size   = 50L,
    max_parallel = 5L,
    enabled      = RUN_CLAUDE
  )
) |> keep(function(m) m$enabled)

cat("Active LLMs:\n")
walk(names(LLM_MODELS), function(nm) {
  m <- LLM_MODELS[[nm]]
  cat(sprintf("  %s: batch=%d, parallel=%d, max_tokens=%d\n",
              nm, m$batch_size, m$max_parallel, m$max_tokens))
})
cat("Folds:", str_c(RUN_FOLDS, collapse = ", "), "\n")


# ══════════════════════════════════════════════════════════════
# SURVEY CONFIGURATION
# ══════════════════════════════════════════════════════════════
#
# Demographics hardcoded from Script 02 screening results:
#   LAPOP: age_cat, Edu, Urban, Employment (male dropped)
#   LB:    age_cat, Edu, Employment (male + Urban dropped)
# ──────────────────────────────────────────────────────────────

config <- list(
  lapop = list(
    survey_label    = "a nationally representative survey conducted in Mexico in 2023",
    demo_vars       = c("age_cat", "Edu", "Urban", "Employment"),
    trust_items     = c(
      "justice_system", "electoral_tribunal", "armed_forces",
      "legislature", "public_ministry", "police", "auditor",
      "political_parties", "supreme_court", "municipality",
      "media", "elections"
    ),
    factor_names    = "Trust",
    trust_ascending = TRUE,
    item_scale_desc = "1 = no trust at all, 7 = a lot of trust",
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
    survey_label    = "a nationally representative survey conducted in Colombia in 2023",
    demo_vars       = c("age_cat", "Edu", "Employment"),
    trust_items     = c(
      "Congress", "Natl_Govt", "Judiciary", "Parties",
      "Electoral_Inst", "President"
    ),
    factor_names    = "Political",
    trust_ascending = FALSE,
    item_scale_desc = "1 = a lot of trust, 4 = no trust at all",
    cfa_syntax = '
      Political =~ Congress + Natl_Govt + Judiciary + Parties +
                   Electoral_Inst + President
      Natl_Govt ~~ President
    '
  )
)


# ══════════════════════════════════════════════════════════════
# CHECKPOINT INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════
# LLM calls are expensive — checkpointing enables resume

cp_dir <- file.path(results_dir, "checkpoints_llm_item")
dir.create(cp_dir, recursive = TRUE, showWarnings = FALSE)

cp_path   <- function(key) file.path(cp_dir, str_c(key, ".rds"))
cp_exists <- function(key) file.exists(cp_path(key))
cp_save   <- function(key, data) saveRDS(data, cp_path(key))
cp_load   <- function(key) readRDS(cp_path(key))


# ══════════════════════════════════════════════════════════════
# COMPUTE FOLD PROFILES
# ══════════════════════════════════════════════════════════════
#
# For each fold: refit CFA (for partial lavPredict), load IRT
# model, extract demographic probabilities from demo_profile_tbl,
# and compute per-respondent psychometric features.
# ──────────────────────────────────────────────────────────────

compute_fold_profiles <- function(fold_i, survey_name, cfg,
                                  test_df, fold_mask) {

  train_df <- folds_list[[survey_name]] |>
    filter(fold_id == fold_i, split == "train")

  trust_train <- train_df |>
    select(all_of(cfg$trust_items)) |>
    mutate(across(everything(), as.numeric))

  item_min <- min(unlist(trust_train), na.rm = TRUE)
  item_max <- max(unlist(trust_train), na.rm = TRUE)

  # ── CFA (for P1 tier) ─────────────────────────────────
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

  # CFA factor scores + ecdf for percentile labels
  fs_cfa_train <- as.numeric(
    lavPredict(cfa_fit, newdata = trust_train,
               method = "regression"))
  ecdf_cfa <- ecdf(fs_cfa_train)

  # Model-implied correlation matrix (for correlated items)
  implied_cor <- compute_implied_cor(cfa_fit, cfg$trust_items)

  # ── IRT (for P2 tier) ─────────────────────────────────
  irt_fit <- irt_models[[survey_name]][[fold_i]]

  # IRT training thetas + ecdf
  theta_irt_train <- irt_list[[survey_name]] |>
    filter(fold_id == fold_i, split == "train") |>
    pull(irt_theta)
  ecdf_irt <- ecdf(theta_irt_train)

  # ── Demographic probability context (from Script 02) ───
  # Filter model_probs for this survey × fold
  demo_probs_fold <- NULL
  if (!is.null(demo_profile_tbl$model_probs)) {
    demo_probs_fold <- demo_profile_tbl$model_probs |>
      filter(survey == survey_name, fold_id == fold_i)
  }

  # ── Few-shot examples (tier-specific) ──────────────────
  few_shot_blocks <- map(TIERS, function(tier) {
    build_few_shot_examples(
      train_df    = train_df,
      cfg         = cfg,
      tier        = tier,
      irt_fit     = irt_fit,
      theta_train = theta_irt_train,
      cfa_fit     = cfa_fit,
      fs_train    = fs_cfa_train,
      implied_cor = implied_cor,
      load_df     = load_df,
      intcpt_df   = intcpt_df,
      item_min    = item_min,
      item_max    = item_max
    )
  }) |> set_names(TIERS)

  # ── Per-respondent profiles ────────────────────────────
  respondent_info <- map(seq_len(nrow(test_df)), function(i) {
    rid <- test_df$row_id[i]
    mask_info <- fold_mask |> filter(row_id == rid)

    if (nrow(mask_info) == 0) return(NULL)

    masked_items   <- mask_info$masked_item[[1]]
    observed_items <- setdiff(cfg$trust_items, masked_items)

    # ── CFA: factor score from partial data ──────────
    test_items_cfa <- test_df[i, ] |>
      select(all_of(cfg$trust_items)) |>
      mutate(across(everything(), as.numeric)) |>
      mutate(across(all_of(masked_items),
                    function(x) NA_real_))

    fs_cfa <- as.numeric(
      lavPredict(cfa_fit, newdata = test_items_cfa,
                 method = "regression"))
    cfa_pctile <- round(ecdf_cfa(fs_cfa) * 100)
    cfa_label  <- label_trust(fs_cfa, fs_cfa_train)
    cfa_pctile_verbal <- str_c(
      cfa_pctile, ordinal_suffix(cfa_pctile),
      " percentile (", cfa_label, " trust)")

    # CFA predictions for masked items
    cfa_preds <- map_int(masked_items, function(it) {
      raw <- compute_cfa_expected(it, fs_cfa, load_df, intcpt_df)
      as.integer(round(pmax(item_min, pmin(item_max, raw))))
    }) |> set_names(masked_items)

    # CFA residuals for observed items
    cfa_resids <- compute_cfa_residuals(
      test_df[i, ], observed_items, fs_cfa, load_df, intcpt_df)
    cfa_mean_resid <- round(mean(cfa_resids), 2)

    # ── IRT: theta from partial data ─────────────────
    partial_vec <- as.numeric(test_df[i, cfg$trust_items])
    if (!cfg$trust_ascending) {
      partial_vec <- as.integer(item_max + 1L - partial_vec)
    }
    partial_vec[match(masked_items, cfg$trust_items)] <- NA

    theta_out <- fscores(
      irt_fit,
      response.pattern = matrix(partial_vec, nrow = 1),
      method = "EAP",
      full.scores.SE = TRUE)
    irt_theta <- as.numeric(theta_out[, 1])

    # IRT probabilities + modal predictions per masked item
    irt_probs <- map(masked_items, function(it) {
      item_obj  <- extract.item(irt_fit, it)
      raw_probs <- as.numeric(probtrace(item_obj, irt_theta))
      if (!cfg$trust_ascending) rev(raw_probs) else raw_probs
    }) |> set_names(masked_items)

    irt_preds <- map_int(masked_items, function(it) {
      probs     <- irt_probs[[it]]
      modal_idx <- which.max(probs)
      as.integer(modal_idx + item_min - 1L)
    }) |> set_names(masked_items)

    # IRT residuals for observed items
    irt_resids <- compute_irt_residuals(
      test_df[i, ], observed_items, irt_fit, irt_theta,
      item_min, cfg$trust_ascending)
    irt_mean_resid <- round(mean(irt_resids), 2)

    # ── Correlated items (top 2 per masked item) ─────
    # Two sets: CFA residuals for P1, IRT residuals for P2
    correlated_items_cfa <- map(masked_items, function(it) {
      find_correlated_items(
        it, observed_items, implied_cor,
        test_df[i, ], cfa_resids, k = 2)
    }) |> set_names(masked_items)

    correlated_items_irt <- map(masked_items, function(it) {
      find_correlated_items(
        it, observed_items, implied_cor,
        test_df[i, ], irt_resids, k = 2)
    }) |> set_names(masked_items)

    list(
      rid               = rid,
      masked_items      = masked_items,
      observed_items    = observed_items,
      item_min          = item_min,
      item_max          = item_max,
      # CFA (P1)
      cfa_pctile_verbal = cfa_pctile_verbal,
      cfa_preds         = cfa_preds,
      cfa_mean_resid    = cfa_mean_resid,
      correlated_items_cfa = correlated_items_cfa,
      # IRT (P2)
      irt_theta         = irt_theta,
      irt_preds         = irt_preds,
      irt_probs         = irt_probs,
      irt_mean_resid    = irt_mean_resid,
      correlated_items_irt = correlated_items_irt
    )
  }) |> compact()

  list(
    respondent_info  = respondent_info,
    few_shot_blocks  = few_shot_blocks,
    demo_probs_fold  = demo_probs_fold
  )
}


# ══════════════════════════════════════════════════════════════
# LLM CALL (with retry logic)
# ══════════════════════════════════════════════════════════════

call_llm <- function(user_prompt, system_prompt, model_cfg,
                     temperature) {
  t0 <- proc.time()[["elapsed"]]

  result <- reduce(seq_len(MAX_RETRIES), function(prev, attempt) {
    if (!is.na(prev)) return(prev)
    Sys.sleep(runif(1, 0.5, 2.0))

    response <- tryCatch({
      chat <- switch(model_cfg$provider,
        openrouter = chat_openrouter(
          model         = model_cfg$model_id,
          system_prompt = system_prompt,
          params        = params(
            temperature = temperature,
            max_tokens  = model_cfg$max_tokens)),
        anthropic = chat_anthropic(
          model         = model_cfg$model_id,
          system_prompt = system_prompt,
          params        = params(
            temperature = temperature,
            max_tokens  = model_cfg$max_tokens))
      )
      as.character(chat$chat(user_prompt, echo = FALSE))
    }, error = function(e) str_c("ERROR: ", e$message))

    # Check if response contains parseable predictions
    if (str_detect(response, "\\d+\\s*:.*=")) return(response)

    cat(sprintf("    Retry %d/%d\n", attempt, MAX_RETRIES))
    NA_character_
  }, .init = NA_character_)

  elapsed <- proc.time()[["elapsed"]] - t0
  list(text = if (is.na(result)) "" else result, time = elapsed)
}


# ══════════════════════════════════════════════════════════════
# BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════

process_batch <- function(batch_test, batch_info, cfg, tier,
                          few_shot_block, demo_probs_fold,
                          survey_name, model_cfg, fold_i,
                          temperature) {

  masked_items_list <- map(batch_info, "masked_items")

  # Build system prompt (tier-invariant)
  system_prompt <- build_system_prompt(cfg)

  # Build user prompt (tier-specific content)
  user_prompt <- build_user_prompt(
    batch_rows      = batch_test,
    batch_info      = batch_info,
    cfg             = cfg,
    tier            = tier,
    few_shot_block  = few_shot_block,
    demo_probs_fold = demo_probs_fold
  )

  # Call LLM
  llm_out <- call_llm(user_prompt, system_prompt, model_cfg,
                       temperature)

  # Parse response
  parsed <- parse_item_predictions(
    llm_out$text,
    batch_test$row_id,
    masked_items_list,
    survey_name)

  # Assemble results for this batch
  map2_dfr(seq_len(nrow(batch_test)), batch_info,
    function(j, info) {
      rid <- batch_test$row_id[j]
      row <- batch_test[j, ]

      map_dfr(info$masked_items, function(it) {
        actual_val <- as.integer(as.numeric(row[[it]]))
        pred_row   <- parsed |>
          filter(row_id == rid, item == it)

        llm_pred <- if (nrow(pred_row) > 0) {
          pred_row$predicted[1]
        } else NA_integer_

        tibble(
          survey       = survey_name,
          fold_id      = fold_i,
          method       = NA_character_,
          tier         = tier,
          row_id       = rid,
          item         = it,
          n_masked     = length(info$masked_items),
          actual       = actual_val,
          predicted    = llm_pred,
          cfa_baseline = info$cfa_preds[[it]],
          irt_baseline = info$irt_preds[[it]],
          call_time    = llm_out$time
        )
      })
    })
}


# ══════════════════════════════════════════════════════════════
# HELPER: MAJORITY VOTE
# ══════════════════════════════════════════════════════════════
# Given a vector of m predictions (possibly with NAs),
# returns the most frequent non-NA value as integer.
# Returns NA if all inputs are NA.

majority_vote <- function(votes) {
  valid <- votes[!is.na(votes)]
  if (length(valid) == 0) return(NA_integer_)
  tab <- table(valid)
  as.integer(names(tab)[which.max(tab)])
}


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════
#
# Structure: survey → fold → model → tier → run (m=5)
# Each run uses a different temperature from TEMP_SCHEDULE.
# After all runs complete, majority vote collapses m=5 into
# a single prediction per respondent × item.
# ──────────────────────────────────────────────────────────────

cat("\n", strrep("=", 60), "\n")
cat("LLM ITEM IMPUTATION (MODEL-THEN-ADJUST, m=5 MI)")
cat("\n", strrep("=", 60), "\n\n")
tic("Total LLM imputation")

# Collect all per-run results (before majority vote)
all_runs_tbl <- map_dfr(names(config), function(survey_name) {
  cfg <- config[[survey_name]]

  cat(sprintf("\n== %s ====================\n", survey_name))
  cat(sprintf("   Demographics: [%s]\n",
              str_c(cfg$demo_vars, collapse = ", ")))

  map_dfr(RUN_FOLDS, function(fold_i) {

    test_df <- folds_list[[survey_name]] |>
      filter(fold_id == fold_i, split == "test") |>
      arrange(row_id)

    fold_mask <- masking_tbl |>
      filter(survey == survey_name, fold_id == fold_i,
             row_id %in% test_df$row_id)

    cat(sprintf("\n  Profiles: %s fold %d | n=%d\n",
                survey_name, fold_i, nrow(test_df)))

    # Compute psychometric profiles once per fold (shared)
    fold_profiles <- compute_fold_profiles(
      fold_i, survey_name, cfg, test_df, fold_mask)

    resp_info       <- fold_profiles$respondent_info
    few_shot_blocks <- fold_profiles$few_shot_blocks
    demo_probs_fold <- fold_profiles$demo_probs_fold

    info_rids <- map_int(resp_info, "rid")

    # ── Loop: model × tier × run ─────────────────────────
    map_dfr(names(LLM_MODELS), function(model_name) {
      model_cfg <- LLM_MODELS[[model_name]]

      map_dfr(TIERS, function(tier) {

        map_dfr(seq_len(M_IMPS), function(run_i) {

          temp <- TEMP_SCHEDULE[run_i]

          # Checkpoint per run
          cp_key <- str_c("llm", model_name, survey_name,
                          fold_i, tier, run_i, sep = "_")
          if (cp_exists(cp_key)) {
            cat(sprintf("  [skip] %s\n", cp_key))
            return(cp_load(cp_key))
          }

          cat(sprintf("  %s | %s fold %d | %s | run %d (temp=%.2f)\n",
                      model_name, survey_name, fold_i,
                      tier, run_i, temp))

          # Filter to respondents with valid profiles
          valid_idx  <- which(test_df$row_id %in% info_rids)
          valid_test <- test_df[valid_idx, ]
          valid_info <- resp_info[match(valid_test$row_id,
                                        info_rids)]

          # Split into batches (model-specific)
          n_valid   <- nrow(valid_test)
          batch_sz  <- model_cfg$batch_size
          batch_idx <- split(seq_len(n_valid),
                             ceiling(seq_len(n_valid) / batch_sz))

          n_batches <- length(batch_idx)
          n_workers <- min(model_cfg$max_parallel, n_batches)

          cat(sprintf("    %d cases | %d batches (size %d) | %d workers\n",
                      n_valid, n_batches, batch_sz, n_workers))

          # Run batches in parallel
          plan(multisession, workers = n_workers)

          run_results <- future_map_dfr(batch_idx, function(idx) {
            process_batch(
              batch_test      = valid_test[idx, ],
              batch_info      = valid_info[idx],
              cfg             = cfg,
              tier            = tier,
              few_shot_block  = few_shot_blocks[[tier]],
              demo_probs_fold = demo_probs_fold,
              survey_name     = survey_name,
              model_cfg       = model_cfg,
              fold_i          = fold_i,
              temperature     = temp)
          }, .options = furrr_options(seed = TRUE))

          plan(sequential)

          # Stamp model name, run_id, and temperature
          run_results <- run_results |>
            mutate(method = model_name,
                   run_id = run_i,
                   temperature = temp)

          parse_ok <- sum(!is.na(run_results$predicted))
          cat(sprintf("    parsed: %d/%d (%.1f%%)\n",
                      parse_ok, nrow(run_results),
                      parse_ok / nrow(run_results) * 100))

          # Checkpoint this run
          cp_save(cp_key, run_results)
          run_results
        })
      })
    })
  })
})

toc()


# ══════════════════════════════════════════════════════════════
# MAJORITY VOTE ACROSS m=5 RUNS
# ══════════════════════════════════════════════════════════════
#
# For each respondent × item × method × tier × fold, take
# the majority vote across the m runs. This parallels the
# m=5 majority vote used for traditional methods in Script 03.
# ──────────────────────────────────────────────────────────────

cat("\n-- Collapsing m=5 runs via majority vote --\n")

results_tbl <- all_runs_tbl |>
  group_by(survey, fold_id, method, tier, row_id, item,
           n_masked, actual, cfa_baseline, irt_baseline) |>
  summarise(
    predicted  = majority_vote(predicted),
    n_runs     = n(),
    n_parsed   = sum(!is.na(predicted)),
    .groups    = "drop"
  )

cat(sprintf("  Raw runs: %d rows\n", nrow(all_runs_tbl)))
cat(sprintf("  After majority vote: %d rows\n", nrow(results_tbl)))
cat(sprintf("  Items with ≥1 parse across runs: %d/%d (%.1f%%)\n",
            sum(!is.na(results_tbl$predicted)),
            nrow(results_tbl),
            mean(!is.na(results_tbl$predicted)) * 100))


# ══════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════

cat("\n", strrep("=", 60), "\n")
cat("LLM ITEM IMPUTATION — RESULTS\n")
cat(strrep("=", 60), "\n\n")

# ── Parse rates ──
cat("-- Parse Rates --\n")
results_tbl |>
  mutate(parsed_ok = !is.na(predicted)) |>
  group_by(method, survey, tier) |>
  summarise(
    n_items    = n(),
    n_parsed   = sum(parsed_ok),
    parse_rate = round(mean(parsed_ok), 3),
    .groups    = "drop"
  ) |>
  arrange(method, survey, tier) |>
  print(n = Inf)

# ── Accuracy by method × tier × survey ──
cat("\n-- Accuracy by Method × Tier × Survey --\n")
results_tbl |>
  filter(!is.na(predicted)) |>
  group_by(method, survey, tier) |>
  summarise(
    accuracy     = round(mean(predicted == actual), 3),
    cfa_accuracy = round(mean(cfa_baseline == actual), 3),
    irt_accuracy = round(mean(irt_baseline == actual), 3),
    marginal_irt = round(
      mean(predicted == actual) -
        mean(irt_baseline == actual), 3),
    mae = round(mean(abs(predicted - actual)), 3),
    n   = n(),
    .groups = "drop"
  ) |>
  arrange(survey, tier, method) |>
  print(n = Inf)

# ── Degradation by n_masked ──
cat("\n-- Accuracy by n_masked --\n")
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

# ── Per-item accuracy at P2 ──
cat("\n-- Per-Item Accuracy (P2 tier) --\n")
results_tbl |>
  filter(!is.na(predicted), tier == "P2") |>
  group_by(method, survey, item) |>
  summarise(
    accuracy = round(mean(predicted == actual), 3),
    irt_acc  = round(mean(irt_baseline == actual), 3),
    marginal = round(
      mean(predicted == actual) -
        mean(irt_baseline == actual), 3),
    n = n(),
    .groups = "drop"
  ) |>
  arrange(survey, item, method) |>
  print(n = Inf)

# ── Per-fold accuracy ──
cat("\n-- Per-Fold Accuracy --\n")
results_tbl |>
  filter(!is.na(predicted)) |>
  group_by(method, survey, tier, fold_id) |>
  summarise(
    accuracy = round(mean(predicted == actual), 3),
    .groups  = "drop"
  ) |>
  pivot_wider(names_from = fold_id,
              values_from = accuracy,
              names_prefix = "f") |>
  arrange(survey, tier, method) |>
  print(n = Inf)


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

# Save majority-vote results (primary output for Script 05)
output_path <- file.path(results_dir, "results_item.rds")
saveRDS(results_tbl, output_path)
cat("\nSaved majority-vote results:", output_path, "\n")

# Save raw per-run results (for MI variance analysis)
raw_path <- file.path(results_dir, "results_item_raw_runs.rds")
saveRDS(all_runs_tbl, raw_path)
cat("Saved raw per-run results:", raw_path, "\n")

cat("\n", strrep("=", 60), "\n")
cat("LLM ITEM IMPUTATION COMPLETE (m=5 MI)\n")
cat(strrep("=", 60), "\n")
