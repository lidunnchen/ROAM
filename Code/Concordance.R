### CONCORDANCE TESTING BETWEEN MODEL AGREEMENT WITH LD FOR 30 MIN VIDEO CLIPS OF THE FIVE ROUTINE BSAE BEHAVIOURS

# =========================================================
# PACKAGES
# =========================================================
library(readxl)
library(dplyr)
library(tidyr)
library(irr)       # Cohen's κ
library(irrCAC)    # Gwet AC1
library(ggplot2)
library(janitor)

# =========================================================
# LOAD FILE
# =========================================================
file_path <- "C:/Users/lchen/OneDrive - Toronto Zoo/Desktop/Polar Bears - UrsAI/ConcordanceMaster_30minVids.xlsx"

df <- read_excel(file_path)

# Ensure factor type
df <- df %>%
  mutate(
    bear1_behaviour = as.factor(bear1_behaviour),
    bear1_MANUAL_obs = as.factor(bear1_MANUAL_obs),
    bear2_behaviour = as.factor(bear2_behaviour),
    bear2_MANUAL_obs = as.factor(bear2_MANUAL_obs)
  )

# =========================================================
# AGREEMENT FUNCTION (κ + AC1)
# =========================================================
evaluate_agreement <- function(x, y, label = "") {
  cat("\n==============================\n")
  cat("Agreement for:", label, "\n")
  cat("==============================\n")
  
  paired <- cbind(x, y)
  
  # Cohen's kappa
  k_out <- kappa2(paired)
  print(k_out)
  
  # Gwet's AC1
  ac1_out <- gwet.ac1.raw(paired)
  print(ac1_out)
}

# =========================================================
# 1. SEPARATE AGREEMENT FOR BEAR 1 & BEAR 2
# =========================================================
evaluate_agreement(df$bear1_behaviour, df$bear1_MANUAL_obs,
                   "Bear 1: Model vs Manual")

evaluate_agreement(df$bear2_behaviour, df$bear2_MANUAL_obs,
                   "Bear 2: Model vs Manual")

# =========================================================
# 2. LONG FORMAT: COMBINE BEARS INTO ONE DATASET
# =========================================================
df_long <- df %>%
  pivot_longer(
    cols = c(bear1_behaviour, bear1_MANUAL_obs,
             bear2_behaviour, bear2_MANUAL_obs),
    names_to = c("bear", ".value"),
    names_pattern = "(bear\\d)_(.*)"
  ) %>%
  rename(model = behaviour, manual = MANUAL_obs)

df_long_clean <- df_long %>% filter(!is.na(model), !is.na(manual))

overall_summary <- df_long_clean %>%
  mutate(agree = model == manual) %>%
  summarise(
    total = n(),
    agreement_rate = mean(agree),
    disagreements = sum(!agree)
  )

print(overall_summary)


# Convert to factor
df_long$model  <- as.factor(df_long$model)
df_long$manual <- as.factor(df_long$manual)

# =========================================================
# 3. RUN κ + AC1 FOR BOTH BEARS COMBINED
# =========================================================
evaluate_agreement(df_long$model, df_long$manual,
                   "Combined Bear 1 + Bear 2")

# =========================================================
# 4. PER-BEHAVIOUR AGREEMENT TABLES
# =========================================================
per_behaviour <- df_long %>%
  mutate(agree = model == manual) %>%
  group_by(model) %>%
  summarise(
    N = n(),
    agreement_rate = mean(agree),
    disagreements = sum(!agree)
  )

print(per_behaviour)

###### EXTRACT CIs for cohens K and gwek AC1
# =========================================================
# BOOTSTRAP CONFIDENCE INTERVAL FOR COHEN'S KAPPA
# =========================================================

bootstrap_kappa_ci <- function(x, y, B = 1000) {
  library(irr)
  
  n <- length(x)
  
  # Bootstrap resampling of κ values
  boots <- replicate(B, {
    idx <- sample(1:n, replace = TRUE)
    kappa2(cbind(x[idx], y[idx]))$value
  })
  
  # 95% percentile CI
  return(quantile(boots, c(0.025, 0.975)))
}

bootstrap_kappa_ci(df$bear1_behaviour, df$bear1_MANUAL_obs) # gives for bear 1

combined_ci <- bootstrap_kappa_ci(df_long_clean$model, df_long_clean$manual)
combined_ci #gives Cohen's k for combined predictoins for all bears (concordance test - agreement betwee StereotypyAI predictions and human)

# bootstrap_kappa_ci for combined (2 bear) evaluation of concordance
# CI 2.5%     97.5% 
  # 0.6525852 0.7615985 

#### get Gwek's AC1
library(irrCAC)

bootstrap_ac1_ci <- function(x, y, B = 1000) {
  paired <- data.frame(x = x, y = y)
  N <- nrow(paired)
  
  ac1_vals <- replicate(B, {
    samp <- paired[sample(1:N, N, replace = TRUE), ]
    out <- try(gwet.ac1.raw(cbind(samp$x, samp$y))$est$coeff.val, silent = TRUE)
    if (inherits(out, "try-error") || is.na(out)) return(NA)
    return(out)
  })
  
  # Remove NAs caused by invalid bootstrap samples
  ac1_vals <- ac1_vals[!is.na(ac1_vals)]
  
  est <- gwet.ac1.raw(cbind(x, y))$est$coeff.val
  lower <- quantile(ac1_vals, 0.025, na.rm = TRUE)
  upper <- quantile(ac1_vals, 0.975, na.rm = TRUE)
  
  return(list(ac1 = est, lower = lower, upper = upper))
}


ac1_b1 <- bootstrap_ac1_ci(df$bear1_behaviour, df$bear1_MANUAL_obs)
ac1_b2 <- bootstrap_ac1_ci(df$bear2_behaviour, df$bear2_MANUAL_obs)
ac1_combined <- bootstrap_ac1_ci(df_long_clean$model, df_long_clean$manual)
ac1_combined


ac1_b1
ac1_b2
ac1_combined

# =========================================================
# 5. CONFUSION MATRIX PLOT (COMBINED)
# =========================================================
conf_mat <- table(df_long$manual, df_long$model)
conf_df <- as.data.frame(conf_mat)
colnames(conf_df) <- c("Manual", "Model", "Count")

ggplot(conf_df, aes(x = Manual, y = Model, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count)) +
  scale_fill_gradient(low = "white", high = "darkred") +
  theme_minimal() +
  ggtitle("Confusion Matrix: Manual vs Model (Both Bears)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# =========================================================
# 6. PER-BEHAVIOUR AGREEMENT BAR CHART
# =========================================================
library(ggplot2)
library(scales)

# Set order including No Detections
behaviour_order <- c(
  "FORAGING",
  "HEAD SWINGING",
  "LOCOMOTION",
  "RESTING",
  "SWIMMING",
  "No Detections"
)

# Replace NA values in model column
per_behaviour$model <- as.character(per_behaviour$model)
per_behaviour$model[is.na(per_behaviour$model)] <- "No Detections"
per_behaviour$model <- factor(per_behaviour$model, levels = behaviour_order)

# Colour palette
behaviour_colors <- c(
  "FORAGING"       = "orange",   # burnt orange
  "HEAD SWINGING"  = "#D36BFF",   # pinky purple
  "LOCOMOTION"     = "green",   # lime green
  "RESTING"        = "yellow",   # goldenrod
  "SWIMMING"       = "#1E90FF",   # water blue
  "No Detections"  = "#C8C8C8"    # grey
)

ggplot(per_behaviour, aes(x = model, y = agreement_rate, fill = model)) +
  geom_col() +
  geom_text(
    aes(label = percent(agreement_rate, accuracy = 0.1)),
    vjust = -0.5, size = 5    
  ) +
  
  # Horizontal 80% agreement threshold
  geom_hline(
    yintercept = 0.85,
    linetype = "dotted",
    color = "black",
    linewidth = 1
  ) +
  
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  scale_fill_manual(values = behaviour_colors) +
  
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major    = element_blank(),
    panel.grid.minor    = element_blank(),
    
    # Axis tick labels: plain black
    axis.text.x         = element_text(size = 12, color = "black", angle = 45, hjust = 1),
    axis.text.y         = element_text(size = 12, color = "black"),
    
    # Axis titles: bold black
    axis.title          = element_text(size = 14, face = "bold", color = "black"),
    
    # Axis lines
    axis.line           = element_line(color = "black", linewidth = 1),
    
    # Remove legend
    legend.position     = "none",
    
    # Ensure diagonal labels do not clip
    plot.margin = margin(10, 20, 20, 10)
  ) +
  
  labs(
    title = "Agreement Rate per Behaviour (Model vs Manual)",
    x = "Behaviour Category",
    y = "Agreement"
  )

#####6B take two including binomial CIs
library(ggplot2)
library(scales)

# Ensure behaviour is ordered correctly in THIS dataset
behaviour_agreement_ci$behaviour <- factor(
  behaviour_agreement_ci$behaviour,
  levels = behaviour_order
)

ggplot(behaviour_agreement_ci,
       aes(x = behaviour,
           y = specific_agreement,
           ymin = lower,
           ymax = upper,
           color = behaviour)) +
  
  # CI forest-style point + whiskers
  geom_pointrange(size = 1.0, fatten = 3) +
  
  # 85% threshold line
  geom_hline(yintercept = 0.85,
             linetype = "dotted",
             linewidth = 1,
             color = "black") +
  
  scale_color_manual(values = behaviour_colors) +
  
  # Y-axis: min 40%
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0.40, 1.0)
  ) +
  
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    # Axis ticks
    axis.text.x = element_text(
      size = 12, color = "black",
      angle = 45, hjust = 1
    ),
    axis.text.y = element_text(size = 12, color = "black"),
    
    # Axis titles bold black
    axis.title = element_text(size = 14, face = "bold", color = "black"),
    
    # Axis borders
    axis.line = element_line(color = "black", linewidth = 1),
    
    legend.position = "none"
  ) + 
  coord_flip()+
  
  labs(
    title = "Behaviour-Specific Agreement with 95% CI",
    x = "Behaviour",
    y = "Agreement"
  )


# =========================================================
# 7. OVERALL AGREEMENT SUMMARY TABLE
# =========================================================

df_long_clean <- df_long %>% filter(!is.na(model), !is.na(manual))

overall_summary <- df_long_clean %>%
  mutate(agree = model == manual) %>%
  summarise(
    total = n(),
    agreement_rate = mean(agree),
    disagreements = sum(!agree)
  )

print(overall_summary)

overall_summary <- df_long %>%
  mutate(agree = model == manual) %>%
  summarise(
    total = n(),
    agreement_rate = mean(agree),
    disagreements = sum(!agree)
  )

print(overall_summary)


#
#Value	Strength of Agreement
##<0.20	Poor
###0.21–0.40	Fair
####0.41–0.60	Moderate
#####0.61–0.80	Substantial
######0.81–1.00	Almost perfect

compute_ac1_ci <- function(df, behaviour_name) {
  
  # Subset rows where either rater used this behaviour
  sub <- df[df$model == behaviour_name | df$manual == behaviour_name, ]
  
  # Not enough data
  if (nrow(sub) < 2) {
    return(data.frame(
      behaviour = behaviour_name,
      ac1 = NA,
      lower = NA,
      upper = NA,
      n = nrow(sub)
    ))
  }
  
  paired <- cbind(sub$model, sub$manual)
  ac1 <- gwet.ac1.raw(paired)
  
  # AC1 returned nothing usable
  if (is.null(ac1$est) || nrow(ac1$est) == 0) {
    return(data.frame(
      behaviour = behaviour_name,
      ac1 = NA,
      lower = NA,
      upper = NA,
      n = nrow(sub)
    ))
  }
  
  est <- ac1$est[1, , drop = FALSE]  # ensure matrix stays 2D
  
  # Check that required columns exist; if not → return NA row
  required_cols <- c("coeff.val", "lower.ci", "upper.ci")
  missing_cols <- !(required_cols %in% colnames(est))
  
  if (any(missing_cols)) {
    return(data.frame(
      behaviour = behaviour_name,
      ac1 = NA,
      lower = NA,
      upper = NA,
      n = nrow(sub)
    ))
  }
  
  # If all good, extract the values
  return(data.frame(
    behaviour = behaviour_name,
    ac1  = est$coeff.val,
    lower = est$lower.ci,
    upper = est$upper.ci,
    n = nrow(sub)
  ))
}




ac1_behaviour <- do.call(
  rbind,
  lapply(behaviour_order, function(b) compute_ac1_ci(df_long_clean, b))
)

ac1_behaviour


library(ggplot2)

ggplot(ac1_behaviour, aes(x = behaviour, y = ac1, ymin = lower, ymax = upper)) +
  geom_pointrange(size = 0.9, color = "black") +
  geom_hline(yintercept = 0.80, linetype = "dotted", color = "red", linewidth = 1) +
  coord_flip() +
  theme_minimal(base_size = 15) +
  theme(
    axis.text = element_text(color = "black"),
    axis.title = element_text(face = "bold", color = "black"),
    panel.grid.major = element_blank()
  ) +
  labs(
    title = "Gwet's AC1 with 95% CI per Behaviour",
    x = "Behaviour Category",
    y = "AC1 Concordance (± 95% CI)"
  )

compute_category_agreement <- function(df, behaviour_name) {
  
  sub <- df[df$model == behaviour_name | df$manual == behaviour_name, ]
  n <- nrow(sub)
  
  if (n == 0)
    return(data.frame(
      behaviour = behaviour_name,
      specific_agreement = NA,
      precision = NA,
      recall = NA,
      f1 = NA,
      n = 0
    ))
  
  TP <- sum(sub$model == behaviour_name & sub$manual == behaviour_name)
  FP <- sum(sub$model == behaviour_name & sub$manual != behaviour_name)
  FN <- sum(sub$model != behaviour_name & sub$manual == behaviour_name)
  
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  f1        <- 2 * (precision * recall) / (precision + recall)
  
  # Specific agreement (Cicchetti, 1990)
  specific_agreement <- (2 * TP) / (2 * TP + FP + FN)
  
  data.frame(
    behaviour = behaviour_name,
    specific_agreement = specific_agreement,
    precision = precision,
    recall = recall,
    f1 = f1,
    n = n
  )
}

behaviour_agreement <- do.call(
  rbind,
  lapply(behaviour_order, function(b)
    compute_category_agreement(df_long_clean, b))
)

behaviour_agreement


library(dplyr)

behaviour_agreement_ci <- behaviour_agreement %>%
  rowwise() %>%
  mutate(
    lower = ifelse(!is.na(specific_agreement),
                   binom.test(round(specific_agreement * n), n)$conf.int[1],
                   NA),
    upper = ifelse(!is.na(specific_agreement),
                   binom.test(round(specific_agreement * n), n)$conf.int[2],
                   NA)
  )

library(ggplot2)
library(scales)

ggplot(behaviour_agreement_ci,
       aes(x = behaviour,
           y = specific_agreement,
           ymin = lower,
           ymax = upper)) +
  
  geom_pointrange(size = 1, color = "black") +
  
  # Horizontal 80% threshold line
  geom_hline(yintercept = 0.80, linetype = "dotted",
             color = "red", linewidth = 1) +
  
  coord_flip() +
  
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, 1)) +
  
  theme_minimal(base_size = 15) +
  theme(
    axis.text      = element_text(color = "black"),
    axis.title     = element_text(color = "black", face = "bold"),
    panel.grid.major = element_blank()
  ) +
  
  labs(
    title = "Agreement per Behaviour (Specific Agreement ± 95% CI)",
    x = "Behaviour Category",
    y = "Agreement (%)"
  )

