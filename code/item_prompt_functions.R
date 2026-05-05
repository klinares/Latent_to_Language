# ============================================================
# item_prompt_functions.R
# L2L Framework — Item Imputation Prompt Functions (v2)
#
# "Model-then-adjust" paradigm:
#   The psychometric model (CFA/IRT) provides a starting-point
#   prediction. The LLM adjusts using respondent consistency,
#   demographic context, and domain knowledge.
#
# Key design principle: prompts communicate model CONCLUSIONS
# (predictions, confidence, residual patterns) in natural
# language — NOT raw parameters (a, b, Q3, loadings). The LLM
# is a reasoner, not a calculator.
#
# Tiers:
#   S  : Demographics + observed items only (world knowledge)
#   P1 : + CFA factor score percentile + CFA prediction anchor
#   P2 : + IRT prediction with confidence + residual pattern +
#        top-2 correlated items + demographic probability context
#
# Prompt budget per respondent (approximate):
#   S  : ~120 tokens
#   P1 : ~160 tokens
#   P2 : ~220 tokens (× 25 respondents = ~5500 + system + fewshot)
# ============================================================

library(tidyverse)


# ══════════════════════════════════════════════════════════════
# LABEL LOOKUPS
# ══════════════════════════════════════════════════════════════

lapop_trust_labels <- c(
  "1" = "No trust at all", "2" = "Very little trust",
  "3" = "Little trust",    "4" = "Some trust",
  "5" = "Moderate trust",  "6" = "High trust",
  "7" = "A lot of trust"
)

lb_trust_labels <- c(
  "1" = "A lot of trust", "2" = "Some trust",
  "3" = "Little trust",   "4" = "No trust"
)

# Reverse lookups for parser fallback (verbal → integer)
lapop_label_to_int <- set_names(
  as.integer(names(lapop_trust_labels)),
  unname(lapop_trust_labels)
)
lb_label_to_int <- set_names(
  as.integer(names(lb_trust_labels)),
  unname(lb_trust_labels)
)

# Display labels for demographic values
value_display_labels <- list(
  Edu   = c("none" = "No formal education",
            "primary" = "Primary",
            "secondary" = "Secondary",
            "higher_ed" = "Higher education"),
  Urban = c("urban" = "Urban", "rural" = "Rural"),
  Employment = c("Employed" = "Employed",
                 "Unemployed" = "Unemployed",
                 "Student" = "Student",
                 "Retired" = "Retired",
                 "house_wife" = "Homemaker")
)


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

# Classify a latent score relative to the training distribution
label_trust <- function(fs_test, fs_train) {
  m <- mean(fs_train)
  s <- sd(fs_train)
  case_when(
    fs_test < (m - 1.0 * s) ~ "VERY LOW",
    fs_test < (m - 0.5 * s) ~ "LOW",
    fs_test < (m + 0.5 * s) ~ "MODERATE",
    fs_test < (m + 1.0 * s) ~ "HIGH",
    TRUE                     ~ "VERY HIGH"
  )
}

# Ordinal suffix for percentiles (1st, 2nd, 3rd, 4th...)
ordinal_suffix <- function(n) {
  if (n %% 100 %in% 11:13) return("th")
  switch(as.character(n %% 10),
         "1" = "st", "2" = "nd", "3" = "rd", "th")
}

# Verbal confidence label from max probability
confidence_label <- function(max_p) {
  if (max_p >= 0.50) "HIGH"
  else if (max_p >= 0.30) "MODERATE"
  else "LOW"
}


# ══════════════════════════════════════════════════════════════
# MODEL-LEVEL COMPUTATIONS
# ══════════════════════════════════════════════════════════════

# CFA model-implied correlation matrix (named matrix)
compute_implied_cor <- function(cfa_fit, trust_items) {
  implied_cov <- lavInspect(cfa_fit, "fitted")$cov
  d <- sqrt(diag(implied_cov))
  cor_mat <- implied_cov / outer(d, d)
  cor_mat[trust_items, trust_items]
}

# CFA expected value: E[X_j | fs] = intercept + loading * fs
compute_cfa_expected <- function(item_name, fs,
                                 load_df, intcpt_df) {
  ld  <- load_df$est[load_df$rhs == item_name]
  int <- intcpt_df$est[intcpt_df$lhs == item_name]
  int + ld * fs
}

# IRT expected value: E[X_j | theta] = sum(k * P(k | theta))
compute_irt_expected <- function(irt_fit, item_name, theta,
                                 item_min, trust_ascending) {
  item_obj  <- extract.item(irt_fit, item_name)
  raw_probs <- as.numeric(probtrace(item_obj, theta))
  if (!trust_ascending) raw_probs <- rev(raw_probs)
  n_cats <- length(raw_probs)
  cats   <- seq(item_min, item_min + n_cats - 1L)
  sum(cats * raw_probs)
}


# ══════════════════════════════════════════════════════════════
# PER-RESPONDENT COMPUTATION HELPERS
# ══════════════════════════════════════════════════════════════

# CFA residuals: observed - expected for each observed item
compute_cfa_residuals <- function(row, observed_items, fs,
                                  load_df, intcpt_df) {
  map_dbl(observed_items, function(item_name) {
    observed <- as.numeric(row[[item_name]])
    expected <- compute_cfa_expected(
      item_name, fs, load_df, intcpt_df)
    round(observed - expected, 2)
  }) |> set_names(observed_items)
}

# IRT residuals: observed - E[X|theta] for each observed item
compute_irt_residuals <- function(row, observed_items,
                                  irt_fit, theta, item_min,
                                  trust_ascending) {
  map_dbl(observed_items, function(item_name) {
    observed <- as.numeric(row[[item_name]])
    expected <- compute_irt_expected(
      irt_fit, item_name, theta, item_min, trust_ascending)
    round(observed - expected, 2)
  }) |> set_names(observed_items)
}

# Find top-k most correlated observed items for a masked item
# Returns tibble with observed values and residuals
find_correlated_items <- function(masked_item, observed_items,
                                  implied_cor, row, residuals,
                                  k = 2) {
  cors <- implied_cor[masked_item, observed_items]
  top_k <- min(k, length(cors))
  top_idx <- order(abs(cors), decreasing = TRUE)[1:top_k]
  top_items <- observed_items[top_idx]

  tibble(
    obs_item     = top_items,
    correlation  = round(cors[top_items], 2),
    observed_val = map_int(top_items, function(it) {
      as.integer(as.numeric(row[[it]]))
    }),
    expected_val = map_dbl(top_items, function(it) {
      # Expected value comes from the residuals vector
      as.numeric(row[[it]]) - residuals[it]
    }),
    residual     = round(residuals[top_items], 1)
  )
}


# ══════════════════════════════════════════════════════════════
# FORMAT HELPERS
# ══════════════════════════════════════════════════════════════

# Format demographics as readable text
format_demographics <- function(row, cfg) {
  map_chr(cfg$demo_vars, function(v) {
    val <- as.character(row[[v]])
    # Look up display labels if available
    if (v %in% names(value_display_labels)) {
      lookup <- value_display_labels[[v]]
      if (val %in% names(lookup)) return(unname(lookup[val]))
    }
    val
  }) |> str_c(collapse = " | ")
}

# Format observed items with verbal labels
format_observed_items <- function(row, cfg, masked_items) {
  label_lookup <- if (cfg$factor_names == "Trust") {
    lapop_trust_labels
  } else {
    lb_trust_labels
  }

  map_chr(cfg$trust_items, function(item_name) {
    if (item_name %in% masked_items) {
      str_c("    ", item_name, " = [MISSING]")
    } else {
      raw_val <- as.character(
        as.integer(as.numeric(row[[item_name]])))
      verbal  <- unname(label_lookup[raw_val])
      if (is.na(verbal)) verbal <- raw_val
      str_c("    ", item_name, " = ", raw_val,
            " (", verbal, ")")
    }
  }) |> str_c(collapse = "\n")
}

# Format residual pattern as a prescriptive instruction
format_residual_summary <- function(mean_resid) {
  if (mean_resid > 0.3) {
    sprintf("This respondent consistently rates ABOVE the model (+%.1f avg) → adjust predictions UPWARD",
            mean_resid)
  } else if (mean_resid < -0.3) {
    sprintf("This respondent consistently rates BELOW the model (%.1f avg) → adjust predictions DOWNWARD",
            mean_resid)
  } else {
    sprintf("This respondent is close to model expectations (%+.1f avg) → model predictions are likely accurate",
            mean_resid)
  }
}

# Format correlated items with explicit implication
format_correlated_items_text <- function(cor_tbl) {
  lines <- map_chr(seq_len(nrow(cor_tbl)), function(j) {
    r <- cor_tbl[j, ]
    if (r$residual > 0.2) {
      sprintf("      %s = %d (model expected %.1f, +%.1f above → suggests this item should also be ABOVE model)",
              r$obs_item, r$observed_val, r$expected_val,
              r$residual)
    } else if (r$residual < -0.2) {
      sprintf("      %s = %d (model expected %.1f, %.1f below → suggests this item should also be BELOW model)",
              r$obs_item, r$observed_val, r$expected_val,
              r$residual)
    } else {
      sprintf("      %s = %d (model expected %.1f, %+.1f — consistent with model)",
              r$obs_item, r$observed_val, r$expected_val,
              r$residual)
    }
  })
  str_c(lines, collapse = "\n")
}


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

build_system_prompt <- function(cfg) {

  str_c(
    "You are a survey psychometrician with a PhD from the ",
    "Joint Program in Survey Methodology (JPSM) at the ",
    "University of Maryland, specializing in survey data ",
    "science. You are reviewing an institutional trust scale ",
    "from ", cfg$survey_label, " where some item responses ",
    "are missing and need to be imputed.\n\n",

    "Scale: ", cfg$item_scale_desc, "\n\n",

    "Your task: For each respondent, a statistical model ",
    "provides a starting-point prediction for each missing ",
    "item. Your job is to REFINE these predictions using ",
    "information the model cannot capture. The model only ",
    "sees numerical patterns — you can see response ",
    "consistency, demographic context, and institutional ",
    "meaning.\n\n",

    "HOW TO REASON:\n",
    "1. Start from the model prediction, then look at the ",
    "RESIDUAL PATTERN. If this respondent consistently rates ",
    "ABOVE or BELOW the model on observed items, the missing ",
    "items almost certainly follow the same direction. Adjust ",
    "accordingly — this is the strongest signal.\n",
    "2. Check the CORRELATED ITEMS. If the two most related ",
    "observed items are both above (or below) the model, that ",
    "is direct evidence to adjust in the same direction.\n",
    "3. Consider DEMOGRAPHIC CONTEXT as a sanity check. If ",
    "the model predicts a value that is unusual for this ",
    "respondent's demographic group, that may warrant a small ",
    "adjustment — but respondent-level patterns always ",
    "outweigh group averages.\n",
    "4. Use DOMAIN KNOWLEDGE. Trust in armed forces is ",
    "substantively different from trust in the judiciary. ",
    "Consider what you know about how these specific ",
    "institutions are perceived in this country.\n",
    "5. When model confidence is LOW (the top category has ",
    "<30% probability), the model is genuinely uncertain — ",
    "your contextual reasoning adds the most value here. ",
    "Do not simply repeat the model prediction.\n",
    "6. Trust scales exhibit HALO EFFECTS. If a respondent ",
    "rates all observed institutions consistently high or ",
    "low, the missing items almost certainly follow suit.\n",
    "7. Your prediction must be an INTEGER on the scale ",
    "(", cfg$item_scale_desc, ").\n\n",

    "OUTPUT: One line per respondent. Format:\n",
    "row_id: item1=value, item2=value\n",
    "No explanation. Only prediction lines."
  )
}


# ══════════════════════════════════════════════════════════════
# PER-RESPONDENT BLOCK BUILDERS (by tier)
# ══════════════════════════════════════════════════════════════

# Tier S: demographics + observed items only
build_respondent_block_s <- function(info, row, cfg) {
  demo_str  <- format_demographics(row, cfg)
  items_str <- format_observed_items(row, cfg, info$masked_items)

  missing_str <- map_chr(info$masked_items, function(it) {
    str_c("    ", it, ": [MISSING]")
  }) |> str_c(collapse = "\n")

  str_c(
    "RESPONDENT ", info$rid, ":\n",
    "  Demographics: ", demo_str, "\n",
    "  Observed trust ratings:\n", items_str, "\n",
    "  Predict:\n", missing_str
  )
}

# Tier P1: + CFA factor score percentile + CFA prediction anchor
build_respondent_block_p1 <- function(info, row, cfg) {
  demo_str  <- format_demographics(row, cfg)
  items_str <- format_observed_items(row, cfg, info$masked_items)
  resid_str <- format_residual_summary(info$cfa_mean_resid)

  # Per-masked-item blocks with CFA anchor
  missing_blocks <- map_chr(info$masked_items, function(it) {
    str_c(
      "    ", it, ": [MISSING]\n",
      "      Model prediction: ", info$cfa_preds[[it]], "\n",
      "      Most informative observed items:\n",
      format_correlated_items_text(info$correlated_items_cfa[[it]])
    )
  }) |> str_c(collapse = "\n")

  str_c(
    "RESPONDENT ", info$rid, ":\n",
    "  Demographics: ", demo_str, "\n",
    "  Latent trust: ", info$cfa_pctile_verbal, "\n",
    "  Observed trust ratings:\n", items_str, "\n",
    "  Overall pattern: ", resid_str, "\n",
    "  MISSING ITEMS — predict each:\n", missing_blocks
  )
}

# Tier P2: + IRT prediction with confidence + demographic context
build_respondent_block_p2 <- function(info, row, cfg,
                                       demo_probs_fold) {
  demo_str  <- format_demographics(row, cfg)
  items_str <- format_observed_items(row, cfg, info$masked_items)
  resid_str <- format_residual_summary(info$irt_mean_resid)

  # Per-masked-item blocks with IRT anchor + confidence +
  # correlated items + demographic probability
  missing_blocks <- map_chr(info$masked_items, function(it) {

    # IRT prediction and confidence
    probs   <- info$irt_probs[[it]]
    modal   <- info$irt_preds[[it]]
    max_p   <- max(probs)
    conf    <- confidence_label(max_p)

    # Top-2 category probabilities for context
    cats    <- seq(info$item_min, info$item_max)
    sorted  <- sort(probs, decreasing = TRUE)
    top2_idx <- order(probs, decreasing = TRUE)[1:2]
    top2_str <- sprintf("%d at %d%%, %d at %d%%",
                        cats[top2_idx[1]],
                        round(sorted[1] * 100),
                        cats[top2_idx[2]],
                        round(sorted[2] * 100))

    # Demographic probability context (if available)
    demo_line <- ""
    if (!is.null(demo_probs_fold)) {
      # Get ALL groups for this item (for population reference)
      all_item_probs <- demo_probs_fold |>
        filter(item == it)

      # Get this respondent's specific group probability
      resp_item_probs <- get_respondent_demo_probs_item(
        row, demo_probs_fold, cfg, it)

      if (!is.null(resp_item_probs) &&
          nrow(all_item_probs) > 1) {
        resp_p <- resp_item_probs$p_high_trust[1]
        # Population reference: mean across all demographic groups
        pop_p <- mean(all_item_probs$p_high_trust)
        demo_line <- sprintf(
          "\n      Demographic context: For your group, P(high trust on %s) = %.0f%% vs. population %.0f%%",
          it, resp_p * 100, pop_p * 100)
      }
    }

    # Confidence framing — directive
    conf_str <- if (conf == "HIGH") {
      sprintf("Confidence: %s (%s) — model is fairly certain",
              conf, top2_str)
    } else if (conf == "MODERATE") {
      sprintf("Confidence: %s (%s) — use residual pattern to refine",
              conf, top2_str)
    } else {
      sprintf("Confidence: %s (%s) — model is uncertain, your reasoning matters most",
              conf, top2_str)
    }

    str_c(
      "    ", it, ": [MISSING]\n",
      "      Model prediction: ", modal, "\n",
      "      ", conf_str, "\n",
      "      Most informative observed items:\n",
      format_correlated_items_text(info$correlated_items_irt[[it]]),
      demo_line
    )
  }) |> str_c(collapse = "\n")

  str_c(
    "RESPONDENT ", info$rid, ":\n",
    "  Demographics: ", demo_str, "\n",
    "  Observed trust ratings:\n", items_str, "\n",
    "  Overall pattern: ", resid_str, "\n",
    "  MISSING ITEMS — predict each:\n", missing_blocks
  )
}


# ══════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLE BUILDER
# ══════════════════════════════════════════════════════════════
#
# 3 examples spanning the latent trust spectrum:
#   1. LOW trust respondent — model is confident, pattern
#      consistent → keep prediction
#   2. MODERATE trust respondent — model uncertain, strong
#      positive residual → adjust UP
#   3. HIGH trust respondent — model confident but demographic
#      context diverges → small adjustment
# ──────────────────────────────────────────────────────────────

build_few_shot_examples <- function(train_df, cfg, tier,
                                    irt_fit, theta_train,
                                    cfa_fit, fs_train,
                                    implied_cor, load_df,
                                    intcpt_df, item_min,
                                    item_max) {

  # No few-shot for S tier (pure world knowledge)
  if (tier == "S") return("")

  n_train   <- nrow(train_df)
  all_items <- cfg$trust_items

  # Choose the latent score vector for this tier
  latent_scores <- if (tier == "P1") fs_train else theta_train

  # Compute residuals for all training respondents
  train_resids <- map_dbl(seq_len(n_train), function(i) {
    row <- train_df[i, ]
    if (tier == "P1") {
      resids <- compute_cfa_residuals(
        row, all_items, fs_train[i], load_df, intcpt_df)
    } else {
      resids <- compute_irt_residuals(
        row, all_items, irt_fit, theta_train[i],
        item_min, cfg$trust_ascending)
    }
    mean(resids)
  })

  # ── Select 3 diverse exemplars ─────────────────────────
  # Tertiles of latent score distribution
  q33 <- quantile(latent_scores, 1/3, na.rm = TRUE)
  q66 <- quantile(latent_scores, 2/3, na.rm = TRUE)

  low_idx  <- which(latent_scores <= q33)
  mid_idx  <- which(latent_scores > q33 & latent_scores <= q66)
  high_idx <- which(latent_scores > q66)

  # 1. Low trust, low residual → keep model
  idx_1 <- low_idx[which.min(abs(train_resids[low_idx]))]

  # 2. Moderate trust, strong positive residual → adjust up
  idx_2 <- mid_idx[which.max(train_resids[mid_idx])]

  # 3. High trust, moderate negative residual → adjust down
  idx_3 <- high_idx[which.min(train_resids[high_idx])]

  exemplar_indices <- c(idx_1, idx_2, idx_3)
  exemplar_decisions <- c(
    "KEEP — model confidence is high and respondent pattern is consistent",
    "ADJUST UP — respondent consistently rates above model, uncertainty is high",
    "ADJUST DOWN — despite high trust overall, this item's pattern suggests lower"
  )

  # ── Build example text for each exemplar ───────────────
  examples <- map2_chr(exemplar_indices, exemplar_decisions,
    function(idx, decision) {

      row <- train_df[idx, ]
      # Mask 2 random items for the example
      masked_items <- sample(all_items, 2)
      obs_items    <- setdiff(all_items, masked_items)

      demo_str  <- format_demographics(row, cfg)
      items_str <- format_observed_items(row, cfg, masked_items)

      if (tier == "P1") {
        fs <- fs_train[idx]
        resids <- compute_cfa_residuals(
          row, obs_items, fs, load_df, intcpt_df)
        mean_resid <- round(mean(resids), 1)
        resid_str  <- format_residual_summary(mean_resid)

        item_lines <- map_chr(masked_items, function(it) {
          cfa_exp  <- compute_cfa_expected(
            it, fs, load_df, intcpt_df)
          cfa_pred <- as.integer(round(
            pmax(item_min, pmin(item_max, cfa_exp))))
          actual   <- as.integer(as.numeric(row[[it]]))

          cor_tbl <- find_correlated_items(
            it, obs_items, implied_cor, row, resids, k = 2)

          str_c(
            "    ", it, ": [MISSING]\n",
            "      Model prediction: ", cfa_pred, "\n",
            "      Most informative observed items:\n",
            format_correlated_items_text(cor_tbl), "\n",
            "      → Correct answer: ", actual)
        }) |> str_c(collapse = "\n")

      } else {
        # P2 tier
        theta <- theta_train[idx]
        resids <- compute_irt_residuals(
          row, obs_items, irt_fit, theta,
          item_min, cfg$trust_ascending)
        mean_resid <- round(mean(resids), 1)
        resid_str  <- format_residual_summary(mean_resid)

        item_lines <- map_chr(masked_items, function(it) {
          item_obj <- extract.item(irt_fit, it)
          raw_probs <- as.numeric(probtrace(item_obj, theta))
          probs <- if (!cfg$trust_ascending) {
            rev(raw_probs)
          } else raw_probs

          cats    <- seq(item_min, item_max)
          modal   <- cats[which.max(probs)]
          max_p   <- max(probs)
          conf    <- confidence_label(max_p)
          top2_idx <- order(probs, decreasing = TRUE)[1:2]
          sorted   <- sort(probs, decreasing = TRUE)
          top2_str <- sprintf("%d at %d%%, %d at %d%%",
                              cats[top2_idx[1]],
                              round(sorted[1] * 100),
                              cats[top2_idx[2]],
                              round(sorted[2] * 100))

          actual <- as.integer(as.numeric(row[[it]]))

          cor_tbl <- find_correlated_items(
            it, obs_items, implied_cor, row, resids, k = 2)

          str_c(
            "    ", it, ": [MISSING]\n",
            "      Model prediction: ", modal, "\n",
            "      Confidence: ", conf,
            " (", top2_str, ")\n",
            "      Most informative observed items:\n",
            format_correlated_items_text(cor_tbl), "\n",
            "      → Correct answer: ", actual)
        }) |> str_c(collapse = "\n")
      }

      # The output line (what the LLM should produce)
      answer_line <- map_chr(masked_items, function(it) {
        str_c(it, "=",
              as.integer(as.numeric(row[[it]])))
      }) |> str_c(collapse = ", ")

      str_c(
        "Example (", decision, "):\n",
        "  Demographics: ", demo_str, "\n",
        "  Observed:\n", items_str, "\n",
        "  Overall pattern: ", resid_str, "\n",
        "  Predict:\n", item_lines, "\n",
        "  Output: ", row$row_id, ": ", answer_line)
    })

  str_c(
    "CALIBRATION EXAMPLES (from training data):\n",
    "These show when to keep the model vs. when to adjust.\n\n",
    str_c(examples, collapse = "\n\n"),
    "\n\nNow predict for the test respondents below.\n\n")
}


# ══════════════════════════════════════════════════════════════
# USER PROMPT BUILDER
# ══════════════════════════════════════════════════════════════
#
# Assembles the complete user prompt for a batch of respondents.
# Structure: [few-shot examples] + [test respondent blocks]
# ──────────────────────────────────────────────────────────────

build_user_prompt <- function(batch_rows, batch_info, cfg, tier,
                               few_shot_block, demo_probs_fold) {

  # Build per-respondent blocks
  respondent_blocks <- map2_chr(
    seq_len(nrow(batch_rows)), batch_info,
    function(j, info) {
      row <- batch_rows[j, ]
      switch(tier,
        S  = build_respondent_block_s(info, row, cfg),
        P1 = build_respondent_block_p1(info, row, cfg),
        P2 = build_respondent_block_p2(
               info, row, cfg, demo_probs_fold)
      )
    })

  str_c(few_shot_block,
        "TEST RESPONDENTS:\n\n",
        str_c(respondent_blocks, collapse = "\n\n"))
}


# ══════════════════════════════════════════════════════════════
# DEMOGRAPHIC PROBABILITY LOOKUP
# ══════════════════════════════════════════════════════════════
#
# For a given respondent, find the model-based probabilities
# from demo_profile_tbl$model_probs that match their
# demographic group. Returns a filtered tibble or NULL if
# no demographic context is available.
# ──────────────────────────────────────────────────────────────

# For a given respondent and item, find the best-matching
# demographic probability from the model_probs table.
# Returns a one-row tibble or NULL.
get_respondent_demo_probs_item <- function(row, demo_probs_fold,
                                            cfg, item_name) {
  if (is.null(demo_probs_fold) || nrow(demo_probs_fold) == 0) {
    return(NULL)
  }

  demo_vars <- cfg$demo_vars

  # For each demographic, check if this respondent's group
  # has a matching row for this item
  matched <- map_dfr(demo_vars, function(dv) {
    resp_group <- as.character(row[[dv]])
    demo_probs_fold |>
      filter(demo_var == dv, group == resp_group, item == item_name)
  })

  if (nrow(matched) == 0) return(NULL)

  # Return the demographic with the largest deviation from 0.5
  # (strongest signal for this item)
  matched |>
    slice_max(abs(p_high_trust - 0.5), n = 1,
              with_ties = FALSE)
}


# ══════════════════════════════════════════════════════════════
# RESPONSE PARSER
# ══════════════════════════════════════════════════════════════
#
# Parses LLM output text into a tibble of predictions.
# Expected format: "row_id: item1=value, item2=value"
# Handles both integer and verbal label responses.
# ──────────────────────────────────────────────────────────────

parse_item_predictions <- function(response_text,
                                   expected_row_ids,
                                   masked_items_list,
                                   survey_name) {

  label_to_int <- if (survey_name == "lapop") {
    lapop_label_to_int
  } else {
    lb_label_to_int
  }
  valid_labels <- names(label_to_int)

  # Strip any reasoning tags
  clean_text <- str_replace_all(
    response_text,
    "<reasoning>[\\s\\S]*?</reasoning>",
    ""
  )

  # Extract prediction lines (start with a number followed by :)
  lines <- str_split(clean_text, "\n")[[1]] |>
    str_trim() |>
    keep(function(l) str_detect(l, "^(row_id[:\\s]\\s*)?\\d+\\s*:"))

  # Parse each expected respondent
  map2_dfr(expected_row_ids, masked_items_list,
    function(rid, masked_items) {
      pattern <- str_c("^(row_id[:\\s]\\s*)?", rid, "\\s*:")
      line <- detect(lines,
                     function(l) str_detect(l, pattern))

      if (is.null(line)) {
        return(map_dfr(masked_items, function(it) {
          tibble(row_id = rid, item = it,
                 predicted = NA_integer_)
        }))
      }

      map_dfr(masked_items, function(it) {
        # Try integer match first
        item_pat <- str_c(it, "\\s*=\\s*(\\d+)")
        m <- str_match(line, item_pat)

        pred_int <- NA_integer_

        if (!is.na(m[1, 1])) {
          pred_int <- as.integer(m[1, 2])
        } else {
          # Try verbal label match
          item_pat2 <- str_c(
            it, "\\s*=\\s*(.+?)(?:,|\\s*$)")
          m2 <- str_match(line, item_pat2)
          if (!is.na(m2[1, 1])) {
            raw_pred <- str_trim(m2[1, 2])
            match_idx <- which(
              str_to_lower(valid_labels) ==
                str_to_lower(raw_pred))
            if (length(match_idx) > 0) {
              pred_int <- unname(
                label_to_int[valid_labels[match_idx[1]]])
            }
          }
        }

        tibble(row_id = rid, item = it,
               predicted = pred_int)
      })
    })
}
