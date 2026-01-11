# =============================================================================
# TOTAL DEPTH INDEX (TDI) IN THE NHL: COMPLETE ANALYSIS SCRIPT
# =============================================================================
# 
# This script reproduces all analyses, figures, and tables from:
# "Latent Variable Modeling for Intangible Constructs in Sports: 
#  The case of team depth in the National Hockey League"
# 
# Dylan Wiwad, PhD
# Hockey Decoded
# 
# =============================================================================

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Set working directory
setwd("~/dev/hockey_site/data/total-depth-index/all_seasons")

# Load required packages
library(lavaan)          # SEM and measurement models
library(lavaanPlot)      # Path diagrams
library(lme4)            # Mixed-effects models
library(broom.mixed)     # Tidy mixed model output
library(dplyr)           # Data manipulation
library(tidyr)           # Data tidying
library(purrr)           # Functional programming
library(ggplot2)         # Visualization
library(patchwork)       # Combining plots
library(slider)          # Rolling windows
library(psych)           # Correlations with significance
library(knitr)           # Tables
library(kableExtra)      # Enhanced tables
library(gt)              # Modern tables
library(semTools)        # Measurement invariance
library(yardstick)       # Model metrics
library(scales)          # Scaling functions

# Optional: showtext for custom fonts
.use_showtext <- requireNamespace("showtext", quietly = TRUE)
if (.use_showtext) {
  library(showtext)
  font_add_google("Charter", "HD-Serif")
  showtext_auto()
}

# Load data
data <- read.csv("final_data_20102025.csv", header = TRUE)

# Brand colors (matching Hockey Decoded style)
col_blue_pale   <- "#3B4B64"  # Site pale blue
col_orange_pale <- "#E76F2B"  # Site pale orange
col_blue_dark   <- "#041E42"  # Oilers dark blue
col_oil_orange  <- "#FF4C00"  # Oilers orange
col_bg_cream    <- "#FBF8F1"  # Warm background
col_white       <- "#FFFFFF"

# Custom theme function
theme_hockey_decoded <- function(base_size = 12,
                                 base_family = if (.use_showtext) "HD-Serif" else "") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      text = element_text(color = col_blue_dark),
      plot.title = element_text(face = "bold", size = base_size * 1.3, margin = margin(b = 6)),
      plot.subtitle = element_text(size = base_size * 1.05, margin = margin(b = 8)),
      plot.caption = element_text(size = base_size * 0.85, color = col_blue_pale),
      axis.title = element_text(face = "bold"),
      panel.grid.major = element_line(color = scales::alpha(col_blue_pale, 0.25), linewidth = 0.4),
      panel.grid.minor = element_line(color = scales::alpha(col_blue_pale, 0.15), linewidth = 0.25),
      plot.background = element_rect(fill = col_white, color = NA),
      legend.position = "top",
      legend.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold")
    )
}
theme_set(theme_hockey_decoded())

# Helper function for ordering games
.order_cols <- function(df) {
  if ("game_number" %in% names(df)) return(arrange(df, teamAbbrev, season, game_number))
  if ("date" %in% names(df)) return(arrange(df, teamAbbrev, season, date))
  df %>%
    mutate(game_id_num = suppressWarnings(as.numeric(game_id))) %>%
    arrange(teamAbbrev, season, game_id_num)
}

# Data cleaning
stopifnot(all(c("teamAbbrev", "season", "game_id", "tdi_factor") %in% names(data)))
data_clean <- data %>%
  mutate(season = as.character(season)) %>%
  .order_cols() %>%
  mutate(team_season = interaction(teamAbbrev, season, drop = TRUE, lex.order = TRUE))


# =============================================================================
# CONSTRUCT VALIDATION
# =============================================================================

# -----------------------------------------------------------------------------
# Table 1: Inter-indicator Correlations
# -----------------------------------------------------------------------------
# "I calculated pearson-product moment correlations at the game level between 
#  total shots on goal, xG, CF, shot depth, xG depth, CF depth, assist depth, 
#  and TOI depth (Table 1)."

cat("\n=== TABLE 1: INTER-INDICATOR CORRELATIONS ===\n")

# Calculate raw depth values (complement of gini)
data <- data %>%
  mutate(
    sog_depth_raw = 1 - sog_gini,
    xgoal_depth_raw = 1 - xgoal_gini,
    cf_depth_raw = 1 - cf_gini,
    assist_depth_raw = 1 - assist_gini,
    toi_depth_raw = 1 - toi_gini
  )

# Select variables for correlation table
cor_vars <- c("total_sogs", "xgoal", "corsi_for",
              "sog_depth_raw", "xgoal_depth_raw", "cf_depth_raw",
              "assist_depth_raw", "toi_depth_raw")

# Compute correlations with significance tests
cor_results <- psych::corr.test(data[cor_vars], method = "pearson")

# Display correlation matrix
print(cor_results$r, digits = 2)

# Create publication-ready table
cor_table <- as.data.frame(cor_results$r) %>%
  mutate(Variable = rownames(.)) %>%
  select(Variable, everything())

# Format with significance stars
cor_matrix_formatted <- cor_results$r
p_matrix <- cor_results$p

for(i in 1:nrow(cor_matrix_formatted)) {
  for(j in 1:ncol(cor_matrix_formatted)) {
    if(i >= j) {
      if(i == j) {
        cor_matrix_formatted[i,j] <- "—"
      } else {
        cor_val <- round(cor_results$r[i,j], 2)
        p_val <- p_matrix[i,j]
        stars <- ifelse(p_val < .001, "***",
                        ifelse(p_val < .01, "**",
                               ifelse(p_val < .05, "*",
                                      ifelse(p_val < .10, "†", ""))))
        cor_matrix_formatted[i,j] <- paste0(sprintf("%.2f", cor_val), stars)
      }
    } else {
      cor_matrix_formatted[i,j] <- ""
    }
  }
}

# Print formatted table
kable(cor_matrix_formatted,
      caption = "Table 1. Pearson product-moment correlations among performance outcomes and depth indicators",
      align = "c") %>%
  kable_styling() %>%
  footnote(general = "N = 38,394. †p < .10, *p < .05, **p < .01, ***p < .001.",
           footnote_as_chunk = TRUE)


# -----------------------------------------------------------------------------
# Figure 1 & Measurement Model: CFA for Depth
# -----------------------------------------------------------------------------
# "I specified a structural equation model with each indicator loading onto 
#  a single latent variable (Figure 1)"

cat("\n=== FIGURE 1: MEASUREMENT MODEL ===\n")

# Define the measurement model
model_depth <- '
    depth =~ cf_depth_z + sog_depth_z + toi_depth_z + xgoal_depth_z
  '

# Fit the model with standardized latent variable
fit_depth <- sem(model_depth, data = data, std.lv = TRUE)

# Model fit
fit_measures <- fitMeasures(fit_depth, c("chisq", "df", "pvalue", "cfi", "rmsea", "srmr"))
cat("\nModel Fit:\n")
cat(sprintf("χ²(%d) = %.3f, p < .001\n", fit_measures["df"], fit_measures["chisq"]))
cat(sprintf("CFI = %.3f, RMSEA = %.3f, SRMR = %.3f\n",
            fit_measures["cfi"], fit_measures["rmsea"], fit_measures["srmr"]))

# Parameter estimates
params_depth <- parameterEstimates(fit_depth, standardized = TRUE, ci = TRUE)
loadings <- params_depth %>%
  filter(op == "=~") %>%
  select(Indicator = rhs, β = std.all, SE = se, p = pvalue, CI_lower = ci.lower, CI_upper = ci.upper)

cat("\nFactor Loadings:\n")
print(loadings, digits = 3)

# Create path diagram
lavaanPlot(
  model = fit_depth,
  node_options = list(shape = "box", fontname = "Helvetica"),
  edge_options = list(color = "grey"),
  coefs = TRUE,
  sig = .05,
  stand = TRUE
)

# Save factor scores for later analyses
#data$tdi_factor <- lavPredict(fit_depth)[, "depth"]


# -----------------------------------------------------------------------------
# Table 2: Measurement Invariance Across Seasons
# -----------------------------------------------------------------------------
# "To assess measurement invariance, I conducted a standard sequence of 
#  increasingly strict tests"

cat("\n=== TABLE 2: MEASUREMENT INVARIANCE ACROSS SEASONS ===\n")

# Configural invariance (same structure across seasons)
fit.config_season <- sem(model_depth, data = data, group = "season", std.lv = TRUE)

# Metric invariance (equal loadings)
fit.metric_season <- sem(model_depth, data = data, group = "season",
                         group.equal = c("loadings"), std.lv = TRUE)

# Scalar invariance (equal loadings + intercepts)
fit.scalar_season <- sem(model_depth, data = data, group = "season",
                         group.equal = c("loadings", "intercepts"), std.lv = TRUE)

# Partial scalar (free TOI intercept)
fit.partial_season <- sem(
  model_depth,
  data = data,
  std.lv = TRUE,
  group = "season",
  group.equal = c("loadings", "intercepts"),
  group.partial = c("toi_depth_z~1")
)

# Compare models
invariance_season <- anova(fit.config_season, fit.metric_season, fit.scalar_season, fit.partial_season)

# Extract fit measures
inv_table_season <- data.frame(
  Model = c("Configural", "Metric", "Scalar", "Partial Scalar"),
  df = c(
    fitMeasures(fit.config_season, "df"),
    fitMeasures(fit.metric_season, "df"),
    fitMeasures(fit.scalar_season, "df"),
    fitMeasures(fit.partial_season, "df")
  ),
  chisq = c(
    fitMeasures(fit.config_season, "chisq"),
    fitMeasures(fit.metric_season, "chisq"),
    fitMeasures(fit.scalar_season, "chisq"),
    fitMeasures(fit.partial_season, "chisq")
  ),
  CFI = c(
    fitMeasures(fit.config_season, "cfi"),
    fitMeasures(fit.metric_season, "cfi"),
    fitMeasures(fit.scalar_season, "cfi"),
    fitMeasures(fit.partial_season, "cfi")
  ),
  RMSEA = c(
    fitMeasures(fit.config_season, "rmsea"),
    fitMeasures(fit.metric_season, "rmsea"),
    fitMeasures(fit.scalar_season, "rmsea"),
    fitMeasures(fit.partial_season, "rmsea")
  ),
  SRMR = c(
    fitMeasures(fit.config_season, "srmr"),
    fitMeasures(fit.metric_season, "srmr"),
    fitMeasures(fit.scalar_season, "srmr"),
    fitMeasures(fit.partial_season, "srmr")
  )
)

# Add delta statistics
inv_table_season$delta_chisq <- c(NA, diff(inv_table_season$chisq)[c(1,2)],
                                  inv_table_season$chisq[4] - inv_table_season$chisq[2])
inv_table_season$delta_df <- c(NA, diff(inv_table_season$df)[c(1,2)],
                               inv_table_season$df[4] - inv_table_season$df[2])

kable(inv_table_season, digits = 3,
      caption = "Table 2. Measurement Invariance Across Seasons") %>%
  kable_styling() %>%
  footnote(general = "The partial scalar model allows TOI depth to vary across seasons.",
           footnote_as_chunk = TRUE)


# -----------------------------------------------------------------------------
# Table S1: Partial Invariance Testing (Seasons)
# -----------------------------------------------------------------------------

cat("\n=== TABLE S1: PARTIAL INVARIANCE TESTING (SEASONS) ===\n")

# Use semTools to identify problematic intercepts
pi.season <- partialInvariance(
  fit = list(
    fit.configural = fit.config_season,
    fit.loadings = fit.metric_season,
    fit.intercepts = fit.scalar_season
  ),
  type = "intercepts",
  refgroup = 1,
  p.adjust = "none",
  method = "satorra.bentler.2001",
  return.fit = TRUE
)

cat("\nProblematic Intercepts (Ranked by Modification Index):\n")
print(pi.season$results)


# -----------------------------------------------------------------------------
# Table 3: Measurement Invariance Across Teams
# -----------------------------------------------------------------------------

cat("\n=== TABLE 3: MEASUREMENT INVARIANCE ACROSS TEAMS ===\n")

# Configural
fit.config_team <- sem(model_depth, data = data, group = "teamAbbrev", std.lv = TRUE)

# Metric
fit.metric_team <- sem(model_depth, data = data, group = "teamAbbrev",
                       group.equal = c("loadings"), std.lv = TRUE)

# Scalar
fit.scalar_team <- sem(model_depth, data = data, group = "teamAbbrev",
                       group.equal = c("loadings", "intercepts"), std.lv = TRUE)

# Partial scalar
fit.partial_team <- sem(
  model_depth,
  data = data,
  std.lv = TRUE,
  group = "teamAbbrev",
  group.equal = c("loadings", "intercepts"),
  group.partial = c("toi_depth_z~1")
)

# Create table (same structure as Table 2)
inv_table_team <- data.frame(
  Model = c("Configural", "Metric", "Scalar", "Partial Scalar"),
  df = c(
    fitMeasures(fit.config_team, "df"),
    fitMeasures(fit.metric_team, "df"),
    fitMeasures(fit.scalar_team, "df"),
    fitMeasures(fit.partial_team, "df")
  ),
  chisq = c(
    fitMeasures(fit.config_team, "chisq"),
    fitMeasures(fit.metric_team, "chisq"),
    fitMeasures(fit.scalar_team, "chisq"),
    fitMeasures(fit.partial_team, "chisq")
  ),
  CFI = c(
    fitMeasures(fit.config_team, "cfi"),
    fitMeasures(fit.metric_team, "cfi"),
    fitMeasures(fit.scalar_team, "cfi"),
    fitMeasures(fit.partial_team, "cfi")
  ),
  RMSEA = c(
    fitMeasures(fit.config_team, "rmsea"),
    fitMeasures(fit.metric_team, "rmsea"),
    fitMeasures(fit.scalar_team, "rmsea"),
    fitMeasures(fit.partial_team, "rmsea")
  ),
  SRMR = c(
    fitMeasures(fit.config_team, "srmr"),
    fitMeasures(fit.metric_team, "srmr"),
    fitMeasures(fit.scalar_team, "srmr"),
    fitMeasures(fit.partial_team, "srmr")
  )
)

inv_table_team$delta_chisq <- c(NA, diff(inv_table_team$chisq)[c(1,2)],
                                inv_table_team$chisq[4] - inv_table_team$chisq[2])
inv_table_team$delta_df <- c(NA, diff(inv_table_team$df)[c(1,2)],
                             inv_table_team$df[4] - inv_table_team$df[2])

kable(inv_table_team, digits = 3,
      caption = "Table 3. Measurement Invariance Across NHL Teams") %>%
  kable_styling() %>%
  footnote(general = "The partial scalar model allows TOI depth to vary across teams.",
           footnote_as_chunk = TRUE)


# -----------------------------------------------------------------------------
# Table S2: Partial Invariance Testing (Teams)
# -----------------------------------------------------------------------------

cat("\n=== TABLE S2: PARTIAL INVARIANCE TESTING (TEAMS) ===\n")

pi.team <- partialInvariance(
  fit = list(
    fit.configural = fit.config_team,
    fit.loadings = fit.metric_team,
    fit.intercepts = fit.scalar_team
  ),
  type = "intercepts",
  refgroup = 1,
  p.adjust = "none",
  method = "satorra.bentler.2001",
  return.fit = TRUE
)

cat("\nProblematic Intercepts (Ranked by Modification Index):\n")
print(pi.team$results)


# =============================================================================
# DEPTH IN THE NHL
# =============================================================================

# -----------------------------------------------------------------------------
# Figure 2: Distribution of Depth
# -----------------------------------------------------------------------------
# "Across all 15 seasons, the TDI follows an approximately normal distribution"

cat("\n=== FIGURE 2: DISTRIBUTION OF DEPTH ===\n")

fig2 <- ggplot(data_clean, aes(x = tdi_factor)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 60,
                 fill = col_blue_pale,
                 color = scales::alpha(col_blue_dark, 0.25)) +
  geom_density(linewidth = 1.1, color = col_oil_orange) +
  labs(
    title = "Distribution of Depth (TDI) Factor Scores",
    subtitle = "Team–game factor scores across all seasons",
    x = "Depth (latent factor score)",
    y = "Density",
    caption = "Hockey Decoded — Depth as dispersion (higher = more even roster contributions)"
  ) +
  theme_hockey_decoded(base_size = 12)

print(fig2)
ggsave("figure_2_depth_distribution.png", fig2,
       width = 7.5, height = 4.75, dpi = 300, bg = "white")

# Report descriptive statistics
cat(sprintf("\nDescriptive Statistics:\n"))
cat(sprintf("  Mean: %.2f\n", mean(data_clean$tdi_factor, na.rm = TRUE)))
cat(sprintf("  SD: %.2f\n", sd(data_clean$tdi_factor, na.rm = TRUE)))
cat(sprintf("  Min: %.2f\n", min(data_clean$tdi_factor, na.rm = TRUE)))
cat(sprintf("  Max: %.2f\n", max(data_clean$tdi_factor, na.rm = TRUE)))


# -----------------------------------------------------------------------------
# ICC Calculation
# -----------------------------------------------------------------------------
# "The intraclass correlation at the team level (ICC = 0.019)"

cat("\n=== INTRACLASS CORRELATION (ICC) ===\n")

# Fit null model with random intercept for team
icc_model <- lmer(tdi_factor ~ 1 + (1 | teamAbbrev), data = data_clean)
icc_summary <- as.data.frame(VarCorr(icc_model))

between_var <- icc_summary$vcov[1]  # Team variance
within_var <- icc_summary$vcov[2]   # Residual variance
total_var <- between_var + within_var
icc_value <- between_var / total_var

cat(sprintf("Between-team variance: %.4f (%.1f%%)\n", between_var, between_var/total_var * 100))
cat(sprintf("Within-team variance: %.4f (%.1f%%)\n", within_var, within_var/total_var * 100))
cat(sprintf("ICC = %.3f\n", icc_value))


# -----------------------------------------------------------------------------
# Figure 3: Lag-1 Autocorrelation by Team
# -----------------------------------------------------------------------------

cat("\n=== FIGURE 3: LAG-1 AUTOCORRELATION BY TEAM ===\n")

# Calculate lag-1 autocorrelation for each team-season
# Use variable name ac1_by_ts to match reference code
ac1_by_ts <- data_clean %>%
  group_by(teamAbbrev, season) %>%
  summarise(
    n_games = n(),
    ac1 = tryCatch({  # Column name must be ac1, not lag1_acf
      if (n() >= 10) {
        acf(tdi_factor, lag.max = 1, plot = FALSE)$acf[2]
      } else {
        NA_real_
      }
    }, error = function(e) NA_real_),
    .groups = "drop"
  ) %>%
  filter(!is.na(ac1), n_games >= 10)

# Summary statistics for reporting
median_acf <- median(ac1_by_ts$ac1, na.rm = TRUE)
q25_acf <- quantile(ac1_by_ts$ac1, 0.25, na.rm = TRUE)
q75_acf <- quantile(ac1_by_ts$ac1, 0.75, na.rm = TRUE)

cat(sprintf("Median lag-1 ACF: r = %.3f\n", median_acf))
cat(sprintf("IQR: [%.3f, %.3f]\n", q25_acf, q75_acf))

# Order teams by median ACF (ASCENDING - this is critical!)
team_order <- ac1_by_ts %>%
  filter(!teamAbbrev %in% c("ATL", "UTA")) %>%
  group_by(teamAbbrev) %>%
  summarize(med = median(ac1, na.rm = TRUE), .groups = "drop") %>%
  arrange(med) %>%  # ASCENDING order
  pull(teamAbbrev)

# Create Figure 3
fig3 <- ac1_by_ts %>%
  filter(!teamAbbrev %in% c("ATL", "UTA")) %>%
  mutate(teamAbbrev = factor(teamAbbrev, levels = team_order)) %>%
  ggplot(aes(x = teamAbbrev, y = ac1)) +
  geom_violin(fill = col_blue_pale,
              color = scales::alpha(col_blue_dark, 0.35),
              alpha = 0.85,
              trim = FALSE) +
  geom_boxplot(width = 0.18,
               outlier.shape = NA,
               fill = col_white,
               color = scales::alpha(col_blue_dark, 0.7)) +
  geom_hline(yintercept = 0,
             linewidth = 0.7,
             linetype = "dashed",
             color = scales::alpha(col_oil_orange, 0.95)) +
  labs(
    title = "Short-Term Stability of Depth by Team",
    subtitle = "Lag-1 autocorrelation within team-seasons",
    x = "Team",
    y = "Lag-1 autocorrelation of Depth"
  ) +
  theme_hockey_decoded(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    panel.grid.major.x = element_blank()
  )

print(fig3)
ggsave("figure_3_ac1_by_team.png", fig3,
       width = 10, height = 5.5, dpi = 300, bg = "white")

# -----------------------------------------------------------------------------
# Rolling Window Analysis & Figure S1 + Table S3
# -----------------------------------------------------------------------------
# "I estimated a series of multilevel models in which next-game depth was 
#  regressed onto the rolling mean of depth computed over the previous n games"

cat("\n=== ROLLING WINDOW ANALYSIS ===\n")

# Prepare data
df_rolling <- data_clean %>%
  arrange(teamAbbrev, game_id_num) %>%
  mutate(
    season = as.factor(season),
    depth = tdi_factor
  )

# Helper function to fit model for window size n
fit_window <- function(n) {
  tmp <- df_rolling %>%
    group_by(teamAbbrev, season) %>%
    mutate(
      depth_roll_n = slide_dbl(depth, mean, .before = n - 1, .complete = TRUE),
      depth_next = lead(depth)
    ) %>%
    ungroup() %>%
    filter(!is.na(depth_roll_n), !is.na(depth_next))
  
  # Mixed effects model
  m <- lmer(depth_next ~ depth_roll_n + (1 | teamAbbrev) + (1 | season), data = tmp)
  
  # Extract fixed effect
  fx <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE)
  slope <- fx %>%
    filter(term == "depth_roll_n") %>%
    transmute(
      window = n,
      beta = estimate,
      se = std.error,
      ci_low = conf.low,
      ci_high = conf.high
      # Remove p.value since it might not exist
    )
  
  # Simple correlations per team-season
  cors <- tmp %>%
    group_by(teamAbbrev, season) %>%
    summarise(
      r = suppressWarnings(cor(depth_roll_n, depth_next, use = "pairwise.complete.obs")),
      .groups = "drop"
    ) %>%
    summarise(
      mean_r = mean(r, na.rm = TRUE),
      median_r = median(r, na.rm = TRUE),
      q25 = quantile(r, .25, na.rm = TRUE),
      q75 = quantile(r, .75, na.rm = TRUE)
    ) %>%
    mutate(window = n)
  
  left_join(slope, cors, by = "window")
}

# Run across window sizes 1-30
cat("Fitting rolling window models (this may take a minute)...\n")
stability <- map_dfr(1:30, fit_window)

cat("\nRolling window analysis complete.\n")
cat(sprintf("β at n=10: %.2f\n", stability$beta[stability$window == 10]))
cat(sprintf("β at n=20: %.2f\n", stability$beta[stability$window == 20]))
cat(sprintf("β at n=30: %.2f\n", stability$beta[stability$window == 30]))


# Figure S1: Predictive Stability by Window Size
cat("\n=== FIGURE S1: PREDICTIVE STABILITY BY ROLLING WINDOW SIZE ===\n")

fig_s1 <- ggplot(stability, aes(x = window, y = beta)) +
  geom_hline(yintercept = 0, linewidth = 0.7, linetype = "dashed",
             color = scales::alpha(col_oil_orange, 0.95)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high),
              fill = scales::alpha(col_blue_pale, 0.15)) +
  geom_line(linewidth = 1.1, color = col_blue_pale) +
  geom_point(size = 2.6, color = col_blue_pale) +
  geom_vline(xintercept = 10, linetype = 3, linewidth = 0.7,
             color = scales::alpha(col_blue_dark, 0.5)) +
  annotate("text", x = 10.2, y = max(stability$beta, na.rm = TRUE),
           label = "Chosen window = 10", hjust = 0, vjust = 1,
           size = 3.5, color = col_blue_dark) +
  labs(
    title = "Predictive stability of Depth by rolling-window size",
    subtitle = "Fixed-effect slope from mixed model: Depth(t+1) ~ mean Depth(t-n+1…t) + (1|team) + (1|season)",
    x = "Rolling window length (games)",
    y = "β (effect of rolling Depth on next-game Depth)"
  ) +
  theme_hockey_decoded(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

print(fig_s1)
ggsave("figure_s1_predictive_stability_by_window.png", fig_s1,
       width = 7.5, height = 5, dpi = 300, bg = "white")


# Table S3: Incremental Gain by Window Size
cat("\n=== TABLE S3: INCREMENTAL GAIN IN PREDICTIVE STABILITY ===\n")

table_s3 <- stability %>%
  arrange(window) %>%
  mutate(
    beta_increment = beta - lag(beta),
    marginal_gain = beta_increment / (window - lag(window))
  ) %>%
  filter(window %in% c(1, 2, 3, 5, 7, 10, 15, 20, 25, 30)) %>%
  select(
    `Window Size (n)` = window,
    `β (Rolling → Next)` = beta,
    `95% CI Lower` = ci_low,
    `95% CI Upper` = ci_high,
    `Δβ from Previous` = beta_increment,
    `Marginal Gain/Game` = marginal_gain
  ) %>%
  mutate(across(where(is.numeric), ~round(.x, 3)))

kable(table_s3,
      caption = "Table S3. Predictive stability of depth by rolling window size and incremental gains",
      align = "c") %>%
  kable_styling() %>%
  footnote(general = "β represents the fixed-effect slope from: Depth(t+1) ~ rolling Depth + (1|team) + (1|season). Δβ shows change from previous window.",
           footnote_as_chunk = TRUE)


# -----------------------------------------------------------------------------
# Figure 4: 10-Game Rolling Depth Predicts Next Game
# -----------------------------------------------------------------------------
# "The rolling 10-game average of depth predicts subsequent Depth
#  (β = 0.38, 95% CI [0.35, 0.41], p < .001)"

cat("\n=== FIGURE 4: ROLLING 10-GAME DEPTH PREDICTION ===\n")

# Calculate rolling 10-game mean and next-game depth
data_roll <- data_clean %>%
  group_by(teamAbbrev, season) %>%
  mutate(
    depth_roll10 = dplyr::lag(slide_dbl(tdi_factor, mean, .before = 10, .complete = TRUE)),
    depth_next   = dplyr::lead(tdi_factor)
  ) %>%
  ungroup()

df_model <- data_roll %>% filter(!is.na(depth_roll10), !is.na(depth_next))

cat(sprintf("Sample size for rolling prediction: %d observations\n", nrow(df_model)))

# Fit mixed effects model
m_next <- lmer(depth_next ~ depth_roll10 + (1 | teamAbbrev) + (1 | season),
               data = df_model, REML = TRUE)

# Extract coefficients
fx <- broom.mixed::tidy(m_next, effects = "fixed", conf.int = TRUE)
beta <- fx$estimate[fx$term == "depth_roll10"]
lwr  <- fx$conf.low[fx$term == "depth_roll10"]
upr  <- fx$conf.high[fx$term == "depth_roll10"]
pvl  <- fx$p.value[fx$term == "depth_roll10"]

cat(sprintf("\nRolling 10-game β = %.3f, 95%% CI [%.3f, %.3f], p < .001\n",
            beta, lwr, upr))

# Create Figure 4 with scatter points
fig4 <- ggplot(df_model, aes(x = depth_roll10, y = depth_next)) +
  geom_point(alpha = 0.18, size = 1.1, color = scales::alpha(col_blue_pale, 0.9)) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.1,
              color = col_oil_orange, fill = scales::alpha(col_orange_pale, 0.25)) +
  labs(
    title = "Rolling Ten-Game Average Depth Predicts Next-Game Depth",
    x = "Depth (10-game rolling mean, t−1)",
    y = "Depth (t)"
  ) +
  theme_hockey_decoded(base_size = 12)

print(fig4)
ggsave("figure_4_rolling_depth_prediction.png", fig4,
       width = 7.5, height = 5, dpi = 300, bg = "white")


# =============================================================================
# DOES DEPTH HELP YOU WIN? (MATCHUP-LEVEL ANALYSIS)
# =============================================================================

# -----------------------------------------------------------------------------
# Load Schedule Data and Add Home/Away Information
# -----------------------------------------------------------------------------

cat("\n=== LOADING SCHEDULE DATA ===\n")

# Load schedule metadata
sched <- read.csv("all_games_meta_20102025.csv", header = TRUE)

sched <- sched %>%
  rename("game_id" = "id") %>%
  mutate(season = as.character(season))

# Join schedule to add home/away information
data_long <- data_clean %>%
  left_join(
    sched %>% select(game_id, homeTeam.abbrev, awayTeam.abbrev),
    by = "game_id"
  ) %>%
  mutate(
    home_away = case_when(
      teamAbbrev == homeTeam.abbrev ~ "home",
      teamAbbrev == awayTeam.abbrev ~ "away",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(home_away))  # Drop any mismatches

cat(sprintf("Joined schedule data: %s team-games with home/away info\n", nrow(data_long)))

# -----------------------------------------------------------------------------
# Create Matchup-Level Dataset
# -----------------------------------------------------------------------------

cat("\n=== PREPARING MATCHUP-LEVEL DATA ===\n")

# Pivot to matchup level (one row per game, home vs away columns)
data_wide <- data_long %>%
  select(game_id, season, home_away, teamAbbrev, outcome,
         tdi_factor, depth_rolling10, xgoal, corsi_for) %>%
  pivot_wider(
    names_from = home_away,
    values_from = c(teamAbbrev, outcome, tdi_factor, depth_rolling10, xgoal, corsi_for),
    names_sep = "_"
  )

# Create differentials
data_wide <- data_wide %>%
  mutate(
    # Within-game depth differential
    depth_diff = tdi_factor_home - tdi_factor_away,
    
    # 10-game rolling depth differential (inclusive of current game)
    depth_roll10_diff = depth_rolling10_home - depth_rolling10_away,
    
    # Volume differentials
    xg_diff = xgoal_home - xgoal_away,
    cf_diff = corsi_for_home - corsi_for_away,
    
    # Home outcome
    win_home = outcome_home  # 1 if home won, 0 if not
  )

cat(sprintf("Created %s matchups\n", nrow(data_wide)))


# -----------------------------------------------------------------------------
# Within-Game Depth Differential Model
# -----------------------------------------------------------------------------

cat("\n=== WITHIN-GAME DEPTH DIFFERENTIAL MODEL ===\n")

m_concurrent <- glmer(
  win_home ~ scale(depth_diff) + (1 | season),
  data = data_wide,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

summary(m_concurrent)
confint(m_concurrent)


# -----------------------------------------------------------------------------
# 10-Game Rolling Depth Differential Model
# -----------------------------------------------------------------------------

cat("\n=== 10-GAME ROLLING DEPTH DIFFERENTIAL MODEL ===\n")

m_prospective <- glmer(
  win_home ~ scale(depth_roll10_diff) + (1 | season),
  data = data_wide,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

summary(m_prospective)
confint(m_prospective)


# -----------------------------------------------------------------------------
# With xG and CF Controls
# -----------------------------------------------------------------------------

cat("\n=== 10-GAME ROLLING DEPTH WITH CONTROLS ===\n")

m_add_depth <- glmer(
  win_home ~ scale(xg_diff) + scale(cf_diff) + scale(depth_roll10_diff) + (1 | season),
  data = data_wide,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

summary(m_add_depth)
confint(m_add_depth)


# -----------------------------------------------------------------------------
# Figure 5: Predicted Win Probability
# -----------------------------------------------------------------------------

cat("\n=== FIGURE 5: PREDICTED WIN PROBABILITY ===\n")

library(ggeffects)

# Create prediction grid
pred_grid <- expand.grid(
  depth_roll10_diff = seq(
    min(data_wide$depth_roll10_diff, na.rm = TRUE),
    max(data_wide$depth_roll10_diff, na.rm = TRUE),
    length.out = 100
  ),
  season = unique(data_wide$season)[1]  # Reference season
)

# Predictions (fixed effects only)
pred_grid$pred_prob <- predict(m_prospective, newdata = pred_grid,
                               type = "response", re.form = NA)

# Get marginal predictions from incremental model
pred_depth <- ggpredict(
  m_add_depth,
  terms = "depth_roll10_diff [all]"
) %>%
  as_tibble() %>%
  rename(depth_roll10_diff = x, win_prob = predicted)

# Figure styling (matches theme_hockey_decoded)
fig5 <- ggplot(pred_depth, aes(x = depth_roll10_diff, y = win_prob)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high),
              fill = scales::alpha("#3B4B64", 0.15)) +
  geom_line(color = "#E76F2B", linewidth = 1.2) +
  labs(
    title = "Predicted Win Probability as a Function of Recent Depth",
    subtitle = "10-game rolling Depth differential (home minus away)",
    x = "Depth Difference (10-game rolling mean, standardized)",
    y = "Predicted Probability of Home Win"
  ) +
  theme_hockey_decoded(base_size = 12) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1))  # Honest 0-100% scale

print(fig5)
ggsave("figure_5_predicted_win_prob_by_depth.png", fig5,
       width = 7.5, height = 4.5, dpi = 300, bg = "white")

# -----------------------------------------------------------------------------
# Prepare Matchup-Level Data with Lagged Rolling Averages
# -----------------------------------------------------------------------------

cat("\n=== PREPARING MATCHUP-LEVEL DATA ===\n")

# 1) Build minimal lookup from schedule
sched_lookup <- sched %>%
  mutate(season = as.character(season)) %>%
  transmute(
    game_id,
    season,
    home_abbrev = toupper(homeTeam.abbrev),
    away_abbrev = toupper(awayTeam.abbrev),
    home_score = suppressWarnings(as.integer(homeTeam.score)),
    away_score = suppressWarnings(as.integer(awayTeam.score))
  ) %>%
  distinct(game_id, .keep_all = TRUE)

# Create rolling averages
data_fixed <- data_long %>%
  arrange(teamAbbrev, game_id_num) %>%
  group_by(teamAbbrev) %>%
  mutate(
    depth_roll10 = depth_rolling10,  # Use existing column
    xg_roll10 = slide_dbl(xgoal, mean, .before = 9, .complete = TRUE),
    cf_roll10 = slide_dbl(corsi_for, mean, .before = 9, .complete = TRUE)
  ) %>%
  ungroup()

# Create lagged versions (previous game)
data_lagged <- data_fixed %>%
  arrange(teamAbbrev, game_id_num) %>%
  group_by(teamAbbrev) %>%
  mutate(
    depth_roll10_lag1 = dplyr::lag(depth_roll10, 1),
    xg_roll10_lag1 = dplyr::lag(xg_roll10, 1),
    cf_roll10_lag1 = dplyr::lag(cf_roll10, 1)
  ) %>%
  ungroup()

# Reshape to matchup level (home vs away)
matchups <- data_lagged %>%
  select(
    game_id, season, teamAbbrev, home_away,
    outcome,
    depth_roll10_lag1, xg_roll10_lag1, cf_roll10_lag1
  ) %>%
  filter(!is.na(home_away)) %>%
  group_by(game_id, home_away) %>%
  slice_head(n = 1) %>%  # one team per side per game
  ungroup() %>%
  pivot_wider(
    names_from = home_away,
    values_from = c(teamAbbrev, outcome,
                    depth_roll10_lag1, xg_roll10_lag1, cf_roll10_lag1),
    values_fn = first,
    names_glue = "{tolower(home_away)}_{.value}"
  )

# Re-derive outcomes from schedule scores (recommended)
matchups <- matchups %>%
  left_join(select(sched_lookup, game_id, home_score, away_score), by = "game_id") %>%
  mutate(
    home_outcome = as.integer(home_score > away_score),
    away_outcome = 1L - home_outcome,
    
    # Differentials
    depth_roll10_diff_lag1 = home_depth_roll10_lag1 - away_depth_roll10_lag1,
    xg_diff_lag1 = home_xg_roll10_lag1 - away_xg_roll10_lag1,
    cf_diff_lag1 = home_cf_roll10_lag1 - away_cf_roll10_lag1
  ) %>%
  filter(!is.na(home_outcome))

cat(sprintf("Created %s matchups with lagged rolling averages\n", nrow(matchups)))


# -----------------------------------------------------------------------------
# Leave-One-Season-Out Cross-Validation
# -----------------------------------------------------------------------------

cat("\n=== LEAVE-ONE-SEASON-OUT CROSS-VALIDATION ===\n")

# Prepare model data
model_data <- matchups %>%
  mutate(
    depth_diff_lag1 = scale(depth_roll10_diff_lag1)[,1],
    xg_diff = scale(xg_diff_lag1)[,1],
    cf_diff = scale(cf_diff_lag1)[,1]
  ) %>%
  select(season, game_id, home_outcome, depth_diff_lag1, xg_diff, cf_diff) %>%
  filter(!is.na(home_outcome), !is.na(depth_diff_lag1))

seasons <- sort(unique(model_data$season))

cat("Running LOSO-CV (this may take a minute)...\n")

cv_results_metrics <- map_dfr(seasons, function(test_season) {
  train_data <- filter(model_data, season != test_season)
  test_data <- filter(model_data, season == test_season)
  
  # Make outcome a factor
  train_data <- mutate(train_data, home_outcome = factor(home_outcome, levels = c(0, 1)))
  test_data <- mutate(test_data, home_outcome = factor(home_outcome, levels = c(0, 1)))
  
  # Baseline: home win rate in TEST data (not training)
  baseline_prob <- mean(as.numeric(as.character(test_data$home_outcome)))
  
  # Fit depth model
  fit_depth <- glm(home_outcome ~ depth_diff_lag1 + xg_diff + cf_diff,
                   data = train_data, family = binomial)
  
  # Predictions
  test_preds <- test_data %>%
    mutate(
      .pred_1 = predict(fit_depth, newdata = ., type = "response"),
      .pred_class = factor(ifelse(.pred_1 > 0.5, 1, 0), levels = c(0, 1)),
      .pred_baseline = factor(1, levels = c(0, 1))
    )
  
  # Calculate metrics
  acc_model <- accuracy_vec(test_preds$home_outcome, test_preds$.pred_class)
  acc_baseline <- accuracy_vec(test_preds$home_outcome, test_preds$.pred_baseline)
  
  logloss_model <- mn_log_loss_vec(
    truth = test_preds$home_outcome,
    estimate = test_preds$.pred_1,
    event_level = "second"
  )
  
  # Calculate baseline log loss using TEST set's home win rate
  logloss_baseline <- -(
    baseline_prob * log(baseline_prob) +
      (1 - baseline_prob) * log(1 - baseline_prob)
  )
  
  tibble(
    season = test_season,
    accuracy_model = acc_model,
    accuracy_baseline = baseline_prob,  # Also use test set rate here
    logloss = logloss_model,
    logloss_baseline = logloss_baseline,
    n_games = nrow(test_preds)
  )
})

# Overall performance
cat("\nOverall Performance:\n")
cat(sprintf("Depth Model: Accuracy = %.1f%%, Log Loss = %.3f\n",
            mean(cv_results_metrics$accuracy_model) * 100,
            mean(cv_results_metrics$logloss)))
cat(sprintf("Baseline: Accuracy = %.1f%%, Log Loss = %.3f\n",
            mean(cv_results_metrics$accuracy_baseline) * 100,
            mean(cv_results_metrics$logloss_baseline)))

# Display by season
print(cv_results_metrics)

# -----------------------------------------------------------------------------
# Figure 6: Accuracy and Log Loss by Season
# -----------------------------------------------------------------------------

cat("\n=== FIGURE 6: ACCURACY AND LOG LOSS BY SEASON ===\n")

library(patchwork)

# MoneyPuck data (accuracy + log loss)
moneypuck <- tibble::tribble(
  ~season,     ~acc_mp, ~logloss_mp,
  "20202021",   0.601,   0.6596,
  "20212022",   0.641,   0.648,
  "20222023",   0.606,   0.656,
  "20232024",   0.611,   0.661,
  "20242025",   0.604,   0.658
)

# Ensure factor order matches CV results
cv_results_metrics <- cv_results_metrics %>%
  mutate(season = factor(season, levels = sort(unique(season))))

moneypuck <- moneypuck %>%
  mutate(season = factor(season, levels = levels(cv_results_metrics$season)))

# Shared color names (keep legends consistent across plots)
cols <- c(
  "Depth model (LOSO-CV)"    = "#3B4B64",  # navy
  "MoneyPuck pre-game model" = "#6EB4B8",  # teal
  "Baseline (home win rate)" = "#9A9A9A"   # gray
)

# =========================
# P1: ACCURACY (top panel)
# =========================
p_acc <-
  ggplot() +
  # Depth model
  geom_line(
    data = cv_results_metrics,
    aes(season, accuracy_model, color = "Depth model (LOSO-CV)", group = 1),
    linewidth = 1.3
  ) +
  geom_point(
    data = cv_results_metrics,
    aes(season, accuracy_model, color = "Depth model (LOSO-CV)"),
    size = 2.6
  ) +
  # MoneyPuck
  geom_line(
    data = moneypuck,
    aes(season, acc_mp, color = "MoneyPuck pre-game model", group = 1),
    linewidth = 1.2, linetype = "dotted"
  ) +
  geom_point(
    data = moneypuck,
    aes(season, acc_mp, color = "MoneyPuck pre-game model"),
    size = 3, shape = 17
  ) +
  # Baseline
  geom_line(
    data = cv_results_metrics,
    aes(season, accuracy_baseline, color = "Baseline (home win rate)", group = 1),
    linewidth = 1.1, linetype = "longdash"
  ) +
  geom_point(
    data = cv_results_metrics,
    aes(season, accuracy_baseline, color = "Baseline (home win rate)"),
    size = 2.3, shape = 15
  ) +
  # Random 50% reference
  geom_hline(yintercept = 0.5, linetype = "dashed",
             color = "#E76F2B", linewidth = 0.8, alpha = 0.7, show.legend = FALSE) +
  annotate("text", x = 2, y = 0.503,
           label = "Random baseline (50%)",
           hjust = 0, vjust = -0.4,
           color = "#E76F2B", size = 3.1, fontface = "italic") +
  scale_color_manual(values = cols, name = NULL,
                     breaks = c("Depth model (LOSO-CV)", "MoneyPuck pre-game model", "Baseline (home win rate)")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0.50, 0.66),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "Predictive Accuracy and Log Loss by Season",
    subtitle = "Depth model vs MoneyPuck vs seasonal baseline (LOSO-CV for depth model)",
    x = NULL, y = "Accuracy"
  ) +
  theme_minimal(base_family = "Charter") +
  theme(
    plot.title = element_text(face = "bold", size = 15, color = "#041E42"),
    plot.subtitle = element_text(size = 12, color = "#3B4B64"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "bottom",
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# =========================
# P2: LOG LOSS (bottom panel)
# =========================
p_logloss <-
  ggplot() +
  # Depth model
  geom_line(
    data = cv_results_metrics,
    aes(season, logloss, color = "Depth model (LOSO-CV)", group = 1),
    linewidth = 1.3
  ) +
  geom_point(
    data = cv_results_metrics,
    aes(season, logloss, color = "Depth model (LOSO-CV)"),
    size = 2.6
  ) +
  # MoneyPuck
  geom_line(
    data = moneypuck,
    aes(season, logloss_mp, color = "MoneyPuck pre-game model", group = 1),
    linewidth = 1.2, linetype = "dotted"
  ) +
  geom_point(
    data = moneypuck,
    aes(season, logloss_mp, color = "MoneyPuck pre-game model"),
    size = 3, shape = 17
  ) +
  # Baseline
  geom_line(
    data = cv_results_metrics,
    aes(season, logloss_baseline, color = "Baseline (home win rate)", group = 1),
    linewidth = 1.1, linetype = "longdash"
  ) +
  geom_point(
    data = cv_results_metrics,
    aes(season, logloss_baseline, color = "Baseline (home win rate)"),
    size = 2.3, shape = 15
  ) +
  # Random 0.693 reference
  geom_hline(yintercept = 0.693, linetype = "dashed",
             color = "#E76F2B", linewidth = 0.8, alpha = 0.7, show.legend = FALSE) +
  annotate("text", x = 2, y = 0.6935,
           label = "Random baseline (0.693)",
           hjust = 0, vjust = -0.4,
           color = "#E76F2B", size = 3.1, fontface = "italic") +
  scale_color_manual(values = cols, name = NULL,
                     breaks = c("Depth model (LOSO-CV)", "MoneyPuck pre-game model", "Baseline (home win rate)")) +
  scale_y_continuous(limits = c(0.645, 0.70),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(
    x = "Season", y = "Log Loss"
  ) +
  theme_minimal(base_family = "Charter") +
  theme(
    plot.title = element_text(face = "bold", size = 15, color = "#041E42"),
    plot.subtitle = element_text(size = 12, color = "#3B4B64"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "bottom",
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# =========================
# Stack with shared legend
# =========================
fig6 <- p_acc / p_logloss +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(fig6)
ggsave("figure_6_accuracy_logloss_by_season.png", fig6,
       width = 8, height = 7, dpi = 300, bg = "white")
# =============================================================================
# WHY DOES DEPTH HELP YOU WIN? (MEDIATION ANALYSIS)
# =============================================================================

cat("\n=== MEDIATION ANALYSIS ===\n")

# -----------------------------------------------------------------------------
# Prepare Mediation Data
# -----------------------------------------------------------------------------
cat("\n=== MEDIATION ANALYSIS (CORRECTED) ===\n")

# Original mediation uses team-level data with already-standardized variables
model_mediation <- '
    # Observed predictor: 10-game rolling depth (lagged)
    # Parallel mediators: xgoal_z, sogs_z, corsi_for_z
    # Outcome: outcome

    # Mediator regressions
    xgoal_z      ~ a1*depth_roll10_lag1
    sogs_z       ~ b1*depth_roll10_lag1
    corsi_for_z  ~ c1*depth_roll10_lag1

    # Outcome regression (direct + mediated paths)
    outcome ~ c_p*depth_roll10_lag1 + 
              a2*xgoal_z + b2*sogs_z + c2*corsi_for_z

    # Allow mediators to correlate (recommended in parallel mediation)
    xgoal_z ~~ sogs_z + corsi_for_z
    sogs_z  ~~ corsi_for_z

    # Indirect effects
    xgoal_ind  := a1*a2
    sogs_ind   := b1*b2
    corsi_ind  := c1*c2
    ind_total  := xgoal_ind + sogs_ind + corsi_ind

    # Total effect of depth on outcome
    total := c_p + ind_total
  '

# Fit on TEAM-LEVEL data (data_lagged), not matchup data!
fit_obs <- sem(
  model_mediation,
  data          = data_lagged,  # TEAM-LEVEL data!
  estimator     = "MLR",      # robust SEs
  missing       = "fiml",
  fixed.x       = TRUE,
  meanstructure = TRUE
)

fitMeasures(fit_obs, c("rmsea", "cfi", "srmr"))
summary(fit_obs, standardized = TRUE, rsquare = TRUE, ci = TRUE)
params_obs <- parameterEstimates(fit_obs, standardized = TRUE, ci = TRUE)

cat("\nMediation Parameter Estimates:\n")
print(params_obs %>% filter(op %in% c("~", ":=")))

lavaanPlot(
  model = fit_obs,
  node_options = list(shape = "box", fontname = "Helvetica"),
  edge_options = list(color = "grey"),
  coefs = TRUE,
  sig = .05
)

# -----------------------------------------------------------------------------
# SUPPLEMENTAL MEDIATION ANALYSIS: Excluding xG
# -----------------------------------------------------------------------------

cat("\n=== SUPPLEMENTAL: MEDIATION WITHOUT xG ===\n")
cat("To demonstrate suppression effect, we re-run mediation excluding xG\n")

# Define supplemental mediation model WITHOUT xgoal_z
model_no_xg <- '
    # Observed predictor: 10-game rolling depth (lagged)
    # Parallel mediators: sogs_z, corsi_for_z (xgoal_z EXCLUDED)
    # Outcome: outcome

    # Mediator regressions
    sogs_z       ~ b1*depth_roll10_lag1
    corsi_for_z  ~ c1*depth_roll10_lag1

    # Outcome regression (direct + mediated paths)
    outcome ~ c_p*depth_roll10_lag1 + 
              b2*sogs_z + c2*corsi_for_z

    # Allow mediators to correlate
    sogs_z  ~~ corsi_for_z

    # Indirect effects
    sogs_ind   := b1*b2
    corsi_ind  := c1*c2
    ind_total  := sogs_ind + corsi_ind

    # Total effect
    total := c_p + ind_total
  '

# Fit the model without xG (using same data as main mediation)
fit_no_xg <- sem(
  model_no_xg,
  data          = data_lagged,
  estimator     = "MLR",
  missing       = "fiml",
  fixed.x       = TRUE,
  meanstructure = TRUE
)

cat("\n--- Mediation Model WITHOUT xG ---\n")
print(summary(fit_no_xg, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE))

# Extract key parameters
cat("\n--- Key Mediation Paths (WITHOUT xG) ---\n")
supp_params <- parameterEstimates(fit_no_xg, standardized = TRUE) %>%
  filter(label %in% c("b1", "c1", "b2", "c2", "c_p",
                      "sogs_ind", "corsi_ind", "ind_total", "total"))

print(supp_params)

# Create comparison table showing the difference
cat("\n--- COMPARISON: With vs. Without xG ---\n")
cat("When xG is INCLUDED in model:\n")
cat("  SOGs indirect effect: β = –0.011 (negative, suppression)\n")
cat("  CF indirect effect:   β = –0.023 (negative, suppression)\n")
cat("  xG indirect effect:   β = +0.040 (positive, quality matters)\n\n")

cat("When xG is EXCLUDED from model:\n")
sogs_ind_no_xg <- supp_params %>% filter(label == "sogs_ind") %>% pull(est)
corsi_ind_no_xg <- supp_params %>% filter(label == "corsi_ind") %>% pull(est)
cat(sprintf("  SOGs indirect effect: β = %.3f (now positive)\n", sogs_ind_no_xg))
cat(sprintf("  CF indirect effect:   β = %.3f (now positive)\n\n", corsi_ind_no_xg))

cat("Interpretation: Volume metrics (shots, possession) appear to help winning\n")
cat("only when shot quality is not controlled. Once quality is accounted for,\n")
cat("they become negative predictors (suppression effect).\n")


# =============================================================================
# SCRIPT COMPLETE
# =============================================================================

cat("\n=============================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("=============================================================================\n")
cat("\nAll analyses from the TDI paper have been reproduced.\n")
cat("Figures and tables are ready for export.\n")
cat("\nFor questions or issues, contact: dwiwad@gmail.com\n")
cat("=============================================================================\n")

