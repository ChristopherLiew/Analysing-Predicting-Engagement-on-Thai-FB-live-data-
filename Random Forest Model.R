# Load Libraries
library(tidyverse)
library(ranger)
library(caret)
library(mlr3)
library(mlr3tuning)
library(mlr3learners)
library(paradox)
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

# 2. Create RF Learner 
learner = lrn("regr.ranger")

# 3. Cross Val Resampling with 10 Folds
resampling = rsmp("cv", folds=10L)

# Set RMSE as Measure 
measure = msr("regr.rmse")

# Define Parameters Search Space (Add in mtry)
search_space = ParamSet$new(list(
  ParamInt$new("num.trees", lower=100, upper=1000),
  ParamInt$new("max.depth", lower=2, upper=10),
  ParamInt$new("min.node.size", lower=2, upper=10),
  ParamLgl$new("oob.error", default = TRUE)
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
# Optimal hyperparams:
# num.trees: 455, max.depth: 8; min.node.size: 3; oob.error FALSE; regr.rmse: 0.0346964

# Build Optimal PRE-EMOTICON Random Forest with Cross Validation
train.control = trainControl(method="cv", number=10)

mtry = c(26/3)
min.node.size = c(3)
splitrule = c('variance')
tuneGrid = data.frame(mtry, min.node.size, splitrule)

rf_pre_cv = train(num_shares ~., data=live_pre, method="ranger", tuneGrid=tuneGrid)
# Best Results
rf_pre_cv$results
# RMSE: 0.03769611; R-Squared: 0.6839219
rf_pre_cv$bestTune

# Feature importance (Variance)
rf_pre = ranger(num_shares ~., data=live_pre, num.trees=455, max.depth=8, min.node.size=3, mtry=8.7, importance = 'impurity')
feat_imp = rf_pre$variable.importance
feat_imp = enframe(feat_imp) # Converts named atomic vecs to a 1~2 col data frame
# FI horizontal bar chart
ggplot(data=feat_imp, aes(x=name, y=value, fill=value)) + geom_bar(stat="identity") + coord_flip() + theme_minimal()
# Highest MDI (Variance) for features: pos_emo_int & num_comments & num_likes



