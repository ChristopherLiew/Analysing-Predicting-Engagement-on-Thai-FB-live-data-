# TBD: Add in Feature selection? or Use Stepwise Regression (AIC) first then CV as second filter to whittle model down?
# Load libraries
library(tidyverse)
library(car)
library(caret)
set.seed(42) # The answer to all things in life

# Load Live Data
live_pre = read_csv(file.choose())
live_post = read_csv(file.choose())

# Drop Status Published Column
live_pre = subset(live_pre, select = -c(status_published))
live_post = subset(live_post, select = -c(status_published))

### Baseline linear regression with CV ### 
# Set up CV
train.control = trainControl(method="cv", number=10)

# PRE-EMOTICON
# Train our baseline linear model
# Drop status_typelink, mon, jan to prevent perfect multicollinearity
base_model_pre = train(num_shares ~ . -num_shares -status_typelink -mon -jan, data=live_pre, 
                       method="lm", 
                       trControl=train.control)
# Summary
summary(base_model_pre)
# Results
print(base_model_pre)

# POST-EMOTICON
# Train our baseline linear model
# Drop status_typelink, mon, jan to prevent perfect multicollinearity
base_model_post = train(num_shares ~ . -num_shares -status_typelink -mon -jan, data=live_post, 
                       method="lm", 
                       trControl=train.control)
# Summary
summary(base_model_post)
# Results
print(base_model_post)
