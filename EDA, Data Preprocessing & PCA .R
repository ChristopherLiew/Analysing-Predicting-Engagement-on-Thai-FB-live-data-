# Import libraries 
library(readr)
library(tidyverse)
library(psych)
library(ggplot2)
library(GGally)
library(lubridate)
library(funModeling)
library(nortest)
library(dummies)
library(car)
library(corrplot)
library(olsrr)

# Import Dataset
fb_live_df = read_csv("Live.csv") # Live.csv
fb_live_df = as_tibble(fb_live_df)

# Overview of Live Dataset Features
fb_live_df_red = fb_live_df[, c(1:12)]

### EDA ###
# 1. Categorical - Status Type
freq(fb_live_df_red$status_type)
describe(fb_live_df_red$status_type)

# 2. Datetime - Status Published
head(fb_live_df_red$status_published, 20)
# Convert to datetime
fb_live_df_red$status_published = mdy_hm(fb_live_df_red$status_published)
describe(fb_live_df_red$status_published)

# Split Dataset into PRE-EMOTICON & POST-EMOTICON as distribution of FB engagement data will be from disparate distributions
# Emoticon implementation date - 24 February 2016
fb_live_df_pre = fb_live_df_red[fb_live_df_red$status_published < ymd("2016/02/24"), ]
fb_live_df_post = fb_live_df_red[fb_live_df_red$status_published >= ymd("2016/02/24"), ]

# 3. Numerical 
# PRE-EMOTICON
# Pairplot 
ggpairs(fb_live_df_pre[,c(4:7)])
# num_reactions
# num_comments
# num_shares
# num_likes

# POST-EMOTICON
# Pairplot - Post-Emoticon
ggpairs(fb_live_df_post[,c(4:12)])
# num_reactions
# num_comments
# num_shares
# num_likes
# num_loves
# num_wows
# num_hahas
# num_sad
# num_angrys

### Preprocessing & Feature Engineering ###
# TBD: Create Prep Pipeline

# Create copy of dataset to work on
fb_live_df_main = fb_live_df_red

# Creating useful features
# 1. Day of Week
fb_live_df_main$dow = wday(fb_live_df_red$status_published, week_start = 1)

# 2. Month of Year
fb_live_df_main$moy = month(fb_live_df_red$status_published)

# 3. Interaction Terms: Pos & Neg Engagement Actions
fb_live_df_main$pos_emo_int = (fb_live_df_main$num_comments)*(fb_live_df_main$num_likes + fb_live_df_main$num_loves)
fb_live_df_main$neg_emo_int = (fb_live_df_main$num_comments)*(fb_live_df_main$num_sads + fb_live_df_main$num_angrys) # PRE-EMOTICON will have no neg_emo_int values

# 4. Drop num_reactions & status_id
fb_live_df_main = fb_live_df_main[, -c(1, 4)]

# Normalisation 
# Split dataset into PRE-EMOTICON & POST-EMOTICON + Apply Min Max Scaler (Since non-normal and ~3.3% outliers)
# Min-Max Scaling
min_max_scaler = function(x) {
  return((x - min(x))/ (max(x) - min(x)))
} 

# PRE-EMOTICON + Min Max Norm
fb_live_df_main_pre = fb_live_df_main[fb_live_df_main$status_published < ymd("2016/02/24"),]
fb_live_df_main_pre[, c(3:10,13:14)] = sapply(fb_live_df_main_pre[, c(3:10,13:14)], min_max_scaler)
# Drop Emoticon Reaction Columns
fb_live_df_main_pre = fb_live_df_main_pre %>% subset(select = -c(num_loves, num_wows, num_hahas, num_sads, num_angrys, neg_emo_int))

# POST-EMOTICON
fb_live_df_main_post = fb_live_df_main[fb_live_df_main$status_published >= ymd("2016/02/24"),]
fb_live_df_main_post[, c(3:10,13:14)] = sapply(fb_live_df_main_post[, c(3:10,13:14)], min_max_scaler)

# One-Hot Encoding of Dummy Variables
# PRE-EMOTICON
status_type_dummy = dummy(fb_live_df_main_pre$status_type)
dow_dummy = dummy(fb_live_df_main_pre$dow)
moy_dummy = dummy(fb_live_df_main_pre$moy)
fb_live_df_main_pre = cbind(fb_live_df_main_pre, status_type_dummy, dow_dummy, moy_dummy)
# Drop doy, moy
fb_live_df_main_pre = fb_live_df_main_pre %>% subset(select = -c(status_type, dow, moy))
# Rename columns
colnames(fb_live_df_main_pre)[10:28] = c("mon", "tues", "wed", "thurs", "fri", "sat", "sun", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
# Final PRE-EMOTICON dataset
glimpse(fb_live_df_main_pre)

# POST-EMOTICON
status_type_dummy = dummy(fb_live_df_main_post$status_type)
dow_dummy = dummy(fb_live_df_main_post$dow)
moy_dummy = dummy(fb_live_df_main_post$moy)
fb_live_df_main_post = cbind(fb_live_df_main_post, status_type_dummy, dow_dummy, moy_dummy)
# Drop original categorical features
fb_live_df_main_post = subset(fb_live_df_main_post, select = -c(dow, moy, status_type))
# Rename columns
colnames(fb_live_df_main_post)[16:34] = c("mon", "tues", "wed", "thurs", "fri", "sat", "sun", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
glimpse(fb_live_df_main_post)

# Box-Cos & Power Transforms to Standardise Numerical Features???

# Feature Selection (VIF + Pearson's)
# PRE-EMOTICON
# 1. Variance Inflation Factor (VIF)
baseline_lm_pre = lm(num_shares ~ . -num_shares -status_published -status_typelink -mon -jan, data=fb_live_df_main_pre)
summary(baseline_lm_pre)
vif(baseline_lm_pre) # Cutoff ~ VIF > 6

# 2. Pearson's Corr Heatmap
cormat_pre = cor(fb_live_df_main_pre[, c(2:28)])
corrplot(cormat_pre, order = "original", tl.col='black', tl.cex=.75) 

# POST-EMOTICON
# 1. Variance Inflation Factor (VIF)
baseline_lm_post = lm(num_shares ~ . -num_shares -status_published -status_typelink -mon -jan, data=fb_live_df_main_post)
summary(baseline_lm_post)
vif(baseline_lm_post) # Cutoff ~ VIF > 6

# 2. Pearson's Corr Heatmap
cormat_pre = cor(fb_live_df_main_post[, c(2:34)])
corrplot(cormat_pre, order = "original", tl.col='black', tl.cex=.75) 

# Analysing Outliers
# 1. PRE-EMOTICON
ols_plot_cooksd_bar(baseline_lm_pre)
pre_emoticon_outlier_ratio = (sum(cooks.distance(baseline_lm_pre) > 0.002)/ nrow(fb_live_df_main_pre)) # 3.31%

# 2. POST-EMOTICON
ols_plot_cooksd_bar(baseline_lm_post)
post_emoticon_outlier_ratio = (sum(cooks.distance(baseline_lm_post) > 0.001)/ nrow(fb_live_df_main_post)) # 3.36%

### Principal Component Analysis ###
# Only applies to organic numerical features
# 1. PRE-EMOTICON
pc_fb_live_pre = prcomp(fb_live_df_main_pre[, c(4,5)], center = TRUE, scale. = TRUE)
summary(pc_fb_live_pre)

# 2. POST-EMOTICON
pc_fb_live_post = prcomp(fb_live_df_main_post[, c(4:11)], center = TRUE, scale. = TRUE)
summary(pc_fb_live_post)

### EXPORT DATA ###
write.csv(fb_live_df_main_pre, file="live_pre_emoticon.csv", row.names = FALSE)
write.csv(fb_live_df_main_post, file="live_post_emoticon.csv", row.names = FALSE)