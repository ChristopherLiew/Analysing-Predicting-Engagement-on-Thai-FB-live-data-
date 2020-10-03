# Load Libraries
library(dbscan)
library(tidyverse)
library(anomalize)
library(dummies)
set.seed(42)

# Load Live Data
live_pre = read_csv(file.choose())
live_post = read_csv(file.choose())

# Drop Status Published Column
live_pre = subset(live_pre, select = -c(status_published))
live_post = subset(live_post, select = -c(status_published))

# Load Unnormalised Live Data
live_pre_unnorm = read_csv(file.choose())
live_post_unnorm = read_csv(file.choose())

# Min-max scale num_shares to ensure all features are normalised
live_pre$num_shares = min_max_scaler(live_pre$num_shares)
live_post$num_shares = min_max_scaler(live_post$num_shares)

## 1.DBScan ##
# PRE-EMOTICON
# K-Distance Graph - Determining eps 
kNNdistplot(live_pre, k=5)
abline(h = 0.1, lty = 2)
sorted_kNN_dist_pre = sort(kNNdist(live_pre, k=5), decreasing = FALSE) # Knee at ~ 0.11
# DBSCAN model
dbs_pre = dbscan(live_pre, eps = 0.1, minPts = 5)
print(dbs_pre)
# Result: 95 cluster(s) and 482 noise points.

## Get Noise Points for PRE-EMOTICON data (Unnormalised for interpretability)##
noise_pts_pre = live_pre_unnorm[dbs_pre$cluster == 0,]
## Analyse Noise Points ##
noise_pts_pre = noise_pts_pre[order(noise_pts_pre$num_shares, decreasing = TRUE),]
# Noise points contain many of the most shared posts 
# By DOW
noise_pts_pre$dow = as.factor(noise_pts_pre$dow)
ggplot(data = noise_pts_pre, aes(x=dow, y=num_shares, fill=dow)) + geom_bar(stat="identity") + theme_minimal()
# By MOY
noise_pts_pre$moy = as.factor(noise_pts_pre$moy)
ggplot(data = noise_pts_pre, aes(x=moy, y=num_shares, fill=moy)) + geom_bar(stat="identity") + theme_minimal()
# By Post Type
noise_pts_pre$status_type = as.factor(noise_pts_pre$status_type)
ggplot(data = noise_pts_pre, aes(x=status_type, y=num_shares, fill=status_type)) + geom_bar(stat="identity") + theme_minimal()
# By Positive Interaction Variable
ggplot(data = noise_pts_pre, aes(x=pos_emo_int, y=num_shares, color=pos_emo_int)) + geom_point(stat="identity") + geom_smooth(method='lm', formula= y~x) + theme_minimal()
# Correlation Heatmap
corrplot(cor(noise_pts_pre[, c(3,4,5,8)]))

# Need more features + facebook data + text data + trend data to see why it went more viral than others 
#-> nature of exceptional posts 
#-> depends on content rather than exogenous seasonal factors
# How do we prove this? Feature importance? Using a linear/ RF model?

# POST-EMOTICON
# K-Distance Graph - Determining eps 
kNNdistplot(live_post, k=5)
abline(h = 0.10, lty = 2)
sort(kNNdist(live_post, k=5), decreasing = FALSE)
# Knee at ~ 0.1
# DBSCAN model
dbs_post = dbscan(live_post, eps = 0.1, minPts = 5)
print(dbs_post)
# 174 cluster(s) and 650 noise points.
## Get Noise Points for PRE-EMOTICON data (Unnormalised for interpretability)##
noise_pts_post = live_post_unnorm[dbs_post$cluster == 0,]
## Analyse Noise Points ##
noise_pts_post = noise_pts_post[order(noise_pts_post$num_shares, decreasing = TRUE),]
# Noise points contain many of the most shared posts 
# By DOW
noise_pts_post$dow = as.factor(noise_pts_post$dow)
ggplot(data = noise_pts_post, aes(x=dow, y=num_shares, fill=dow)) + geom_bar(stat="identity") + theme_minimal()
# By MOY
noise_pts_post$moy = as.factor(noise_pts_post$moy)
ggplot(data = noise_pts_post, aes(x=moy, y=num_shares, fill=moy)) + geom_bar(stat="identity") + theme_minimal()
# By Post Type
noise_pts_post$status_type = as.factor(noise_pts_post$status_type)
ggplot(data = noise_pts_post, aes(x=status_type, y=num_shares, fill=status_type)) + geom_bar(stat="identity") + theme_minimal()
# By Positive Interaction Variable
ggplot(data = noise_pts_post, aes(x=pos_emo_int, y=num_shares, color=pos_emo_int)) + geom_point(stat="identity") + geom_smooth(method='lm', formula= y~x) + theme_minimal()
# By Negative Interaction Variable
ggplot(data = noise_pts_post, aes(x=neg_emo_int, y=num_shares, color=neg_emo_int)) + geom_point(stat="identity") + geom_smooth(method='lm', formula= y~x) + theme_minimal()
# By num_wows
ggplot(data = noise_pts_post, aes(x=num_wows, y=num_shares, color=num_wows)) + geom_point(stat="identity") + geom_smooth(method='lm', formula= y~x) + theme_minimal()
# By num_hahas
ggplot(data = noise_pts_post, aes(x=num_hahas, y=num_shares, color=num_hahas)) + geom_point(stat="identity") + geom_smooth(method='lm', formula= y~x) + theme_minimal()
# Correlation Heatmap
corrplot(cor(noise_pts_post[, c(3:10, 13:14)]))

## 2. Anomalise - Time Series ##
# Prepare time series PRE-EMOTICON dataset
live_pre_unnorm_ts = subset(cbind(live_pre_unnorm, dummy(live_pre_unnorm$status_type)), select = -c(status_type, dow, moy))
# Decompose and Find Anomalies
tbl_df(arrange(live_pre_unnorm_ts, status_published)) %>%
  time_decompose(num_shares) %>%
  anomalize(remainder, alpha=0.015, max_anoms = 0.05) %>%
  time_recompose() %>%
  plot_anomalies(time_recomposed = TRUE) +
  ggtitle("Top 5% of Anomalies: Pre-Emoticons")

# Prepare time series POST-EMOTICON dataset
live_post_unnorm_ts = subset(cbind(live_post_unnorm, dummy(live_post_unnorm$status_type)), select = -c(status_type, dow, moy))
# Decompose and Find Anomalies
tbl_df(arrange(live_post_unnorm_ts, status_published)) %>%
  time_decompose(num_shares) %>%
  anomalize(remainder, alpha=0.015, max_anoms = 0.05) %>% 
  # Alpha is the exponential smoothing parameter, when we decrease it it decreases the band by which a point would need to exceed to be considered an outlier
  # s(t) = a*x(t) + (1-a)*x(t-1)
  # Increases weights on past instances
  time_recompose() %>%
  plot_anomalies(time_recomposed = TRUE) +
  ggtitle("Top 5% of Anomalies: Post-Emoticons")


