# Load Libraries
library(tidyverse)
library(xgboost)
library(paradox)
library(mlr3)
library(mlr3tuning)
library(mlr3learners)
set.seed(42)

# Load Live Data
live_pre = read_csv(file.choose())
live_post = read_csv(file.choose())

# Drop Status Published Column
live_pre = subset(live_pre, select = -c(status_published))
live_post = subset(live_post, select = -c(status_published))

### Hyperparameter Tuning with Randomised Search ###
## PRE-EMOTICON ##
# 1. Create Task
X_pre = subset(live_pre, select=-c(num_shares))
Y_pre = live_pre %>% select(num_shares)
task_fb_live_pre = TaskRegr$new(id = "fb_live_pre", backend = cbind(X_pre, Y_pre), target = "num_shares")

# 2. Create XGBoost Learner 
learner = lrn("regr.xgboost")

# 3. Cross Val Resampling with 10 Folds
resampling = rsmp("cv", folds=10L)

# Set RMSE as Measure 
measure = msr("regr.rmse")

# Define Parameters Search Space
search_space = ParamSet$new(list(
  ParamFct$new("booster", c("gbtree", "gblinear", "dart")),
  ParamInt$new("nrounds", lower=50, upper=100),
  ParamDbl$new("eta", lower=0.01, upper=0.3),
  ParamInt$new("max_depth", lower=2, upper=10),
  ParamDbl$new("gamma", lower=0.0, upper=5.0),
  ParamDbl$new("subsample", lower=0.5, upper=1.0),
  ParamDbl$new("colsample_bytree", lower=0.5, upper=1.0)
))

# Set Terminator
terminator = trm("evals", n_evals = 20)

# Instantiate Tuner Instance
tuner_instance = TuningInstanceSingleCrit$new(
  task = task_fb_live_pre,
  learner = learner,
  resampling = resampling,
  measure = measure,
  search_space = search_space,
  terminator = terminator
)

# Load Tuner with Random Search
rnd_fb_live_pre_tuner = tnr("random_search")

# Initiate Random Search
rnd_fb_live_pre_tuner$optimize(tuner_instance)

# Optimal Hyperparams
print(tuner_instance$result)
# booster: dart, nrounds: 69, eta: 0.2157147, max_depth: 5, gamma: 0.4264975, subsample: 0.7896956, colsample_bytree: 0.6876473
# Regression RMSE: 0.03768123

## Cross Val Results with Optimal Hyperparams ##
# CV scores for Pre-Emoticon Dataset
xg_data_pre = xgb.DMatrix(as.matrix(X_pre), label=as.numeric(Y_pre$num_shares))
params = list(max_depth=5, eta=0.2157147, subsample=0.7896956, booster="dart", objective="reg:squarederror", colsample_bytree=0.6876473, gamma= 0.4264975, silent=0)
metrics = list('rmse')
xg_best_model_pre = xgb.cv(params = params, data = xg_data_pre, nrounds = 69, nfold = 10, prediction = TRUE, metrics = metrics, early_stopping_rounds = 50) 
xg_best_model_pre$evaluation_log %>% filter(test_rmse_mean == min(test_rmse_mean))

# Best Scores with Optimal Hyperparams
# Best score iteration 69:
# Train RMSE mean = 0.0374834
# Test RMSE mean = 0.0386709

## POST-EMOTICON ##
# 1. Create Task
X_post = subset(live_post, select=-c(num_shares))
Y_post = live_post %>% select(num_shares)
task_fb_live_post = TaskRegr$new(id = "fb_live_post", backend = cbind(X_post, Y_post), target = "num_shares")

# 2. Use existing Learner, Eval, ParamSet

# 3. Create new tuner instance for Post
tuner_instance_post = TuningInstanceSingleCrit$new(
  task = task_fb_live_pre,
  learner = learner,
  resampling = resampling,
  measure = measure,
  search_space = search_space,
  terminator = terminator
)

# Load Tuner with Random Search
rnd_fb_live_post_tuner = tnr("random_search")

# Initiate Random Search
rnd_fb_live_post_tuner$optimize(tuner_instance_post)

# Optimal Hyperparams
print(tuner_instance_post$result)
# booster: gblinear, nrounds: 73, eta: 0.2182722, max_depth: 9, gamma: 1.600217, subsample: 0.7039262, colsample_bytree: 0.5663904
# Regression RMSE: 0.03757522

## Cross Val Results with Optimal Hyperparams ##
# CV scores for Post-Emoticon Dataset
xg_data_post = xgb.DMatrix(as.matrix(X_post), label=as.numeric(Y_post$num_shares))
params = list(max_depth=9, eta=0.2182722, subsample=0.7039262, booster="gblinear", objective="reg:squarederror", colsample_bytree=0.5663904, gamma=1.600217, silent=0)
metrics = list('rmse')
xg_best_model_post = xgb.cv(params = params, data = xg_data_post, nrounds = 73, nfold = 10, prediction = TRUE, metrics = metrics, early_stopping_rounds = 50) 
xg_best_model$evaluation_log %>% filter(test_rmse_mean == min(test_rmse_mean))

# Best Scores with Optimal Hyperparams
# Best score iteration 56:
# Train RMSE mean = 0.0218388
# Test RMSE mean = 0.0220236
