# ═══════════════════════════════════════════════════════════════
# RQ3: Item-Level Measurement Error — Claude P2 Focus
# IRT discrimination, thresholds, item variance
# Gold vs Deletion vs Claude P2
# ═══════════════════════════════════════════════════════════════

library(lavaan)
library(mirt)

set.seed(721)

data_dir <- "D:/repos/UMD_classes_code/TSE_II_SURV721/results"
load(file.path(data_dir, "evaluation_rq.RData"))
load(file.path(data_dir, "fold_data.RData"))

library(tidyverse)

survey_config <- list(
  lapop = list(
    trust_items = c(
      "justice_system", "electoral_tribunal", "armed_forces",
      "legislature", "public_ministry", "police", "auditor",
      "political_parties", "supreme_court", "municipality",
      "media", "elections"),
    trust_ascending = TRUE),
  lb = list(
    trust_items = c(
      "Congress", "Natl_Govt", "Judiciary", "Parties",
      "Electoral_Inst", "President"),
    trust_ascending = FALSE)
)

# ── Load results for reconstruction ──────────────────────────
llm_results  <- readRDS(file.path(data_dir, "results_item.rds"))
trad_results <- readRDS(file.path(data_dir,
                                  "results_trad_item.rds"))

shared_cols <- c("survey", "fold_id", "n_masked", "tier",
                 "method", "row_id", "item", "actual",
                 "predicted", "cfa_baseline", "irt_baseline")

all_results <- bind_rows(
  trad_results |> filter(!is.na(predicted)) |>
    select(all_of(shared_cols)) |>
    mutate(method_type = "Traditional"),
  llm_results |> filter(!is.na(predicted)) |>
    select(all_of(shared_cols)) |>
    mutate(method_type = "LLM")
) |>
  filter(method != "gemma3_27b") |>
  mutate(
    method_label = recode(method,
                          llama4_maverick = "Maverick",
                          claude_sonnet   = "Claude Sonnet",
                          mixgb           = "mixgb",
                          miceRanger      = "miceRanger"),
    tier = factor(tier, levels = c("S", "P1", "P2"))
  )

# ── Reconstruct training + imputed test ──────────────────────
reconstruct_full_data <- function(survey_name, fold_i,
                                  method_name, tier_name) {
  cfg <- survey_config[[survey_name]]
  train_df <- folds_list[[survey_name]] |>
    filter(fold_id == fold_i, split == "train") |>
    select(row_id, all_of(cfg$trust_items)) |>
    mutate(across(all_of(cfg$trust_items), as.numeric))
  test_df <- folds_list[[survey_name]] |>
    filter(fold_id == fold_i, split == "test") |>
    select(row_id, all_of(cfg$trust_items)) |>
    mutate(across(all_of(cfg$trust_items), as.numeric))
  preds <- all_results |>
    filter(survey == survey_name, fold_id == fold_i,
           method == method_name, tier == tier_name) |>
    select(row_id, item, predicted) |>
    mutate(predicted = as.numeric(predicted))
  if (nrow(preds) == 0L) return(NULL)
  preds_wide <- preds |>
    pivot_wider(names_from = item, values_from = predicted,
                names_prefix = "imp_")
  test_merged <- test_df |>
    left_join(preds_wide, by = "row_id")
  test_imputed <- map_dfc(cfg$trust_items, function(it) {
    actual_val <- test_merged[[it]]
    imp_col <- paste0("imp_", it)
    imp_val <- if (imp_col %in% names(test_merged)) {
      test_merged[[imp_col]]
    } else { NA_real_ }
    tibble(!!it := if_else(!is.na(imp_val),
                           imp_val, actual_val))
  }) |>
    mutate(row_id = test_merged$row_id)
  bind_rows(train_df, test_imputed)
}

# ── Fit IRT and extract item params ──────────────────────────
# Uses the same coef(fit, printSE = TRUE) pattern from Script 02
fit_irt_extract <- function(data_df, cfg) {
  items <- cfg$trust_items
  data_mat <- data_df |>
    select(all_of(items)) |>
    mutate(across(everything(), as.numeric))
  item_max <- max(data_mat, na.rm = TRUE)
  irt_data <- if (!cfg$trust_ascending) {
    data_mat |>
      mutate(across(everything(),
                    function(x) as.integer(item_max + 1L - x)))
  } else {
    data_mat |>
      mutate(across(everything(), as.integer))
  }
  
  fit <- mirt(irt_data, 1, itemtype = "graded",
              verbose = FALSE)
  
  # Extract per-item using coef(fit, printSE = TRUE)
  # Returns named list: one element per item, each a matrix
  # with rows "par" and "SE", columns "a1", "d1", "d2", ...
  item_coefs <- coef(fit, printSE = TRUE)
  
  map_dfr(items, function(item_name) {
    mat <- item_coefs[[item_name]]
    par_row <- mat["par", ]
    param_names <- names(par_row)
    
    # a1 = discrimination
    a_val <- par_row[["a1"]]
    
    # d params = intercepts (mirt default parameterization)
    # Convert to IRT thresholds: b_k = -d_k / a
    d_cols <- param_names[grepl("^d", param_names)]
    
    bind_rows(
      tibble(item = item_name, param = "a", value = a_val),
      map_dfr(d_cols, function(dc) {
        d_val <- par_row[[dc]]
        b_val <- -d_val / a_val
        # Label as b1, b2, etc.
        b_label <- gsub("^d", "b", dc)
        tibble(item = item_name, param = b_label,
               value = b_val)
      })
    )
  })
}

# ═══════════════════════════════════════════════════════════════
# EXTRACT: 3 conditions × 5 folds × 2 surveys
# ═══════════════════════════════════════════════════════════════

cat("Extracting IRT item parameters...\n\n")

irt_params <- map_dfr(c("lapop", "lb"), function(s) {
  cfg <- survey_config[[s]]
  map_dfr(1:5, function(fi) {
    cat("  ", s, "fold", fi, "\n")
    
    # Gold
    gold_df <- folds_list[[s]] |>
      filter(fold_id == fi) |>
      select(row_id, all_of(cfg$trust_items)) |>
      mutate(across(all_of(cfg$trust_items), as.numeric))
    gold_p <- fit_irt_extract(gold_df, cfg) |>
      mutate(condition = "Gold")
    
    # Deletion
    del_df <- folds_list[[s]] |>
      filter(fold_id == fi, split == "train") |>
      select(row_id, all_of(cfg$trust_items)) |>
      mutate(across(all_of(cfg$trust_items), as.numeric))
    del_p <- fit_irt_extract(del_df, cfg) |>
      mutate(condition = "Deletion")
    
    # Claude P2
    claude_df <- reconstruct_full_data(s, fi,
                                       "claude_sonnet", "P2")
    claude_p <- fit_irt_extract(claude_df, cfg) |>
      mutate(condition = "Claude P2")
    
    bind_rows(gold_p, del_p, claude_p) |>
      mutate(survey = s, fold_id = fi)
  })
}) |>
  mutate(
    survey_label = if_else(survey == "lapop",
                           "LAPOP Mexico", "Latinobarómetro Colombia"),
    condition = factor(condition,
                       levels = c("Gold", "Deletion", "Claude P2"))
  )

# ═══════════════════════════════════════════════════════════════
# TABLE 1: IRT DISCRIMINATION (a)
# ═══════════════════════════════════════════════════════════════

cat("\n══════════════════════════════════════════════════\n")
cat("TABLE 1: IRT Discrimination (a) by Item x Condition\n")
cat("         Averaged across 5 folds\n")
cat("══════════════════════════════════════════════════\n\n")

discrim <- irt_params |>
  filter(param == "a") |>
  group_by(survey_label, item, condition) |>
  summarise(mean_a = round(mean(value, na.rm = TRUE), 3),
            .groups = "drop") |>
  pivot_wider(names_from = condition,
              values_from = mean_a) |>
  mutate(
    delta_claude = round(`Claude P2` - Gold, 4),
    delta_del    = round(Deletion - Gold, 4)
  ) |>
  arrange(survey_label, item)

discrim |> print(n = 30)

cat("\n-- Summary --\n\n")
discrim |>
  group_by(survey_label) |>
  summarise(
    mean_delta_claude = round(mean(delta_claude, na.rm = TRUE), 4),
    mean_delta_del    = round(mean(delta_del, na.rm = TRUE), 4),
    mean_abs_claude   = round(mean(abs(delta_claude), na.rm = TRUE), 4),
    mean_abs_del      = round(mean(abs(delta_del), na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  print()

# ═══════════════════════════════════════════════════════════════
# TABLE 2: IRT THRESHOLDS (b params)
# ═══════════════════════════════════════════════════════════════

cat("\n══════════════════════════════════════════════════\n")
cat("TABLE 2: IRT Thresholds -- delta from Gold\n")
cat("         Averaged across 5 folds\n")
cat("══════════════════════════════════════════════════\n\n")

thresholds <- irt_params |>
  filter(param != "a") |>
  group_by(survey_label, item, param, condition) |>
  summarise(mean_val = mean(value, na.rm = TRUE),
            .groups = "drop") |>
  pivot_wider(names_from = condition,
              values_from = mean_val) |>
  mutate(
    delta_claude = round(`Claude P2` - Gold, 4),
    delta_del    = round(Deletion - Gold, 4)
  ) |>
  arrange(survey_label, item, param)

thresholds |> print(n = 100)

cat("\n-- Threshold Shift Summary --\n\n")
thresholds |>
  group_by(survey_label) |>
  summarise(
    mean_delta_claude = round(mean(delta_claude, na.rm = TRUE), 4),
    mean_delta_del    = round(mean(delta_del, na.rm = TRUE), 4),
    mean_abs_claude   = round(mean(abs(delta_claude), na.rm = TRUE), 4),
    mean_abs_del      = round(mean(abs(delta_del), na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  print()

# ═══════════════════════════════════════════════════════════════
# TABLE 3: ITEM-LEVEL RESPONSE VARIANCE
# ═══════════════════════════════════════════════════════════════

cat("\n══════════════════════════════════════════════════\n")
cat("TABLE 3: Item-Level Response Variance\n")
cat("         Gold vs Deletion vs Claude P2\n")
cat("══════════════════════════════════════════════════\n\n")

item_var <- map_dfr(c("lapop", "lb"), function(s) {
  cfg <- survey_config[[s]]
  items <- cfg$trust_items
  map_dfr(1:5, function(fi) {
    gold_data <- folds_list[[s]] |>
      filter(fold_id == fi) |>
      select(all_of(items)) |>
      mutate(across(everything(), as.numeric))
    del_data <- folds_list[[s]] |>
      filter(fold_id == fi, split == "train") |>
      select(all_of(items)) |>
      mutate(across(everything(), as.numeric))
    claude_data <- reconstruct_full_data(
      s, fi, "claude_sonnet", "P2") |>
      select(all_of(items))
    map_dfr(items, function(it) {
      tibble(
        survey = s, fold_id = fi, item = it,
        var_gold   = var(gold_data[[it]], na.rm = TRUE),
        var_del    = var(del_data[[it]], na.rm = TRUE),
        var_claude = var(claude_data[[it]], na.rm = TRUE)
      )
    })
  })
}) |>
  mutate(
    survey_label = if_else(survey == "lapop",
                           "LAPOP Mexico", "Latinobarómetro Colombia"),
    pct_claude = round((var_claude - var_gold) /
                         var_gold * 100, 2),
    pct_del    = round((var_del - var_gold) /
                         var_gold * 100, 2)
  )

item_var |>
  group_by(survey_label, item) |>
  summarise(
    var_gold   = round(mean(var_gold), 3),
    var_claude = round(mean(var_claude), 3),
    var_del    = round(mean(var_del), 3),
    pct_claude = round(mean(pct_claude), 2),
    pct_del    = round(mean(pct_del), 2),
    .groups = "drop"
  ) |>
  arrange(survey_label, item) |>
  print(n = 30)

cat("\npct = % change from gold. Negative = variance shrinkage.\n")

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

save(irt_params, discrim, thresholds, item_var,
     file = file.path(data_dir, "rq3_item_params.RData"))
cat("\nSaved: rq3_item_params.RData\n")