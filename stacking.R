# Stacking without sampling / 6 : 1 ratio

# set working directory
setwd("C:/Users/82106/Desktop/SDS Brightics 공모전")

# import library
library("dplyr")
library("tidyr")
library("data.table")
library("ggplot2")
library("stringr")
library("corrplot")
library("caret")
library("randomForest")
library("xgboost")
library("rgl")
library("FNN")
library("glmnet")

# import data set
quality <- fread("qualityData.csv")


# split train & test
quality_train <- quality %>% filter(is.na(Y) == F)
quality_test <- quality %>% filter(is.na(Y) == T)

# control ratio
quantile(quality_train$Y, probs = seq(0,1,0.1)) %>% round(2)

high_y <- quality_train %>% filter(Y > 1.80)
low_y <- quality_train %>% filter(Y <= 1.80)

folds_ratio <- createFolds(low_y$Y, 10)

low_y_new <- rbind(low_y[folds_ratio$Fold01,], low_y[folds_ratio$Fold02,], low_y[folds_ratio$Fold03,],
                   low_y[folds_ratio$Fold04,], low_y[folds_ratio$Fold05,], low_y[folds_ratio$Fold06,],
                   low_y[folds_ratio$Fold07,])

quality_train <- rbind(high_y, low_y_new) # 6:1 y label

hist(quality_train$Y, breaks = 1000)
quantile(quality_train$Y, probs = seq(0,1,0.1)) %>% round(2)
rm(folds_ratio)

# create folds for stacking
folds <- createFolds(quality_train$Y, 5)

# level_0 model xgboost

# converting data into xgb format
dtrain <- xgb.DMatrix(data = as.matrix(quality_train[,-87]), label = quality_train$Y)
dtest <- xgb.DMatrix(data = as.matrix(quality_test[,-87], label = holdout$Y))

# customizing wmae metric : xgboost package
wmae <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err1 <- labels / sum(labels) 
  err2 <- abs(labels - preds)
  err <- sum(err1 * err2)
  return(list(metric = "wmae", value = err))
}

# Tuning
max_depth_grid <- seq(4,10,2)
min_child_grid <- seq(1,9,2)
colsample_bytree <- seq(0.4, 1, 0.2)
subsample <- c(0.5, 1, 0.25)

xgb_tune_1 <- as.data.frame(matrix(nrow = length(max_depth_grid)*length(min_child_grid)*length(colsample_bytree)*length(subsample),
                                   ncol = 5, NA))
colnames(xgb_tune_1) <- c("max_depth", "min_child_weight", "colsample_bytree", "subsample","WMAE")

x <- 1
for(i in max_depth_grid){
  for(j in min_child_grid){
    for (k in colsample_bytree){
      for (l in subsample){
        params_1 = list(objective = "reg:linear",
                        eta = 0.1,
                        max_depth = i,
                        min_child_weight = j,
                        colsample_bytree = k,
                        subsample = l)
        xgb_cv_1 <- xgb.cv(params = params_1,
                           data = dtrain,
                           nrounds = 10000,
                           verbose = 1,
                           prediction = TRUE,
                           early_stopping_rounds = 100,
                           nthread = 2,
                           maximize = FALSE,
                           folds = folds,
                           feval = wmae)
        best_wmae <- xgb_cv_1$evaluation_log[,min(test_wmae_mean)]
        xgb_tune_1[x,1] <- i
        xgb_tune_1[x,2] <- j
        xgb_tune_1[x,3] <- k
        xgb_tune_1[x,4] <- l
        xgb_tune_1[x,5] <- best_wmae
        x <- x+1
      }
    }
  }
}

xgb_tune_1[which.min(xgb_tune_1$WMAE),]
# max_depth min_child_weight colsample_bytree subsample      WMAE
# 10        3                0.8              1         0.9207469

# search optimal nrounds with optimized params
params <- list(objective = "reg:linear",
               eta = 0.01, 
               max_depth = 10, 
               min_child_weight = 3, 
               subsample = 1, 
               colsample_bytree = 0.8)

xgb_cv_optiaml <- xgb.cv(method = "xgbTree",
                         params = params,
                         data = dtrain,
                         nrounds = 10000,
                         maximize = F,
                         prediction = T,
                         feval = wmae,
                         early_stopping_rounds = 150,  # Stopping. Best iteration:
                         folds = folds)                # [431]	train-wmae:0.212738+0.018505	test-wmae:0.927890+0.304680

# fitting xgb model
xg_model <- xgb.train(method = "xgbTree",
                      params = params,
                      data = dtrain,
                      nrounds = 431,
                      maximize = F,
                      prediction = T,
                      verbose = TRUE,
                      feval = wmae)

# importance plot for xgb
mat <- xgb.importance (feature_names = colnames(quality_train),
                       model = xg_model)
xgb.plot.importance (importance_matrix = mat[1:30])

# level_0 model random forest

# customizing wmae metric : caret package
WMAE <- function(data, lev = NULL, model = NULL) {
  err1 <- data[, "obs"] / sum(data[, "obs"]) 
  err2 <- abs(data[, "obs"] - data[, "pred"])
  err <- sum(err1 * err2)
  names(err) <- "WMAE"
  err
}

control <- trainControl(method = "cv", 
                        number = 5, 
                        search = "grid")

tunegrid <- expand.grid(.mtry = (1:15)) 

rf_random <- train(Y ~ .,
                   data = quality_train,
                   method = "rf",
                   metric = "MAE",
                   tuneGrid = tunegrid, 
                   trControl = control
)

print(rf_random)
rf_random$results[which.min(rf_random$results$MAE),]

# mtry      RMSE  Rsquared       MAE    RMSESD RsquaredSD      MAESD
# 15   0.9845325 0.2254249 0.3978129 0.1763977 0.07602182 0.02662968

# 5-fold cv using customized folds

# fold index should be training index
fold_tr_1 <- c(folds$Fold2, folds$Fold3, folds$Fold4, folds$Fold5)
fold_tr_2 <- c(folds$Fold1, folds$Fold3, folds$Fold4, folds$Fold5)
fold_tr_3 <- c(folds$Fold1, folds$Fold2, folds$Fold4, folds$Fold5)
fold_tr_4 <- c(folds$Fold1, folds$Fold2, folds$Fold3, folds$Fold5)
fold_tr_5 <- c(folds$Fold1, folds$Fold2, folds$Fold3, folds$Fold4)

custom_folds_tr <- list()
custom_folds_tr$Fold1 <- fold_tr_1
custom_folds_tr$Fold2 <- fold_tr_2
custom_folds_tr$Fold3 <- fold_tr_3
custom_folds_tr$Fold4 <- fold_tr_4
custom_folds_tr$Fold5 <- fold_tr_5

control <- trainControl(method = "cv", 
                        index = custom_folds_tr, 
                        savePredictions = "final")

tunegrid <- expand.grid(.mtry = 15) 

rf_cv <- train(Y~. , data = quality_train, method = "rf", metric = "MAE", tuneGrid = tunegrid, 
               ntree = 1000, trControl = control)

# order pred values
rf_oof_pred <- rf_cv$pred
rf_oof_pred$Resample <- as.factor(rf_oof_pred$Resample)
rf_oof_pred <- rf_oof_pred[order(rf_oof_pred$rowIndex),]

# fitting rf model
rf_model <- randomForest(Y~. , data = quality_train, mtry = 15, ntree = 1000)

# view variable importance plot
varImpPlot(rf_model)
mat2 <- varImpPlot(rf_model) %>% as.data.frame()
mat2$Feature <- rownames(mat2)
rownames(mat2) <- NULL
mat2 <- mat2[order(desc(mat2$IncNodePurity)),]

# level_0 model knn regression

# select important feature from xgboost & random forest
xg_imp_feature <- mat$Feature %>% head(10) # xgb top 10 features
rf_imp_feature <- mat2$Feature %>% head(10) # rf top 10 features
commom_imp_feature <- unique(c(xg_imp_feature, rf_imp_feature)) # xgb & rf top features

# quality_train for knn
quality_train_knn <- quality_train %>% select(commom_imp_feature, Y)

# optimize parameter by Grid search
control <- trainControl(method = "cv", 
                        number = 5, 
                        search = "grid")

knn_cv <- train(Y ~ .,
                data = quality_train_knn,
                method = "knn",
                trControl = control,
                preProcess = c("center", "scale"),
                tuneGrid = expand.grid(k = seq(1, 31, by = 2))
)

ggplot(knn_cv$results, aes(x=k, y=MAE)) + geom_point()
knn_cv$results[which.min(knn_cv$results$MAE),]

# k      RMSE  Rsquared       MAE   RMSESD RsquaredSD      MAESD
# 9 0.9945102 0.2267231 0.3940864 0.164426  0.1151644 0.03633279

# 5-fold cv using customized folds

control <- trainControl(method = "cv", 
                        index = custom_folds_tr, 
                        savePredictions = "final")

knn_cv2 <- train(Y~. , data = quality_train_knn, method = "knn", preProcess = c("center", "scale"),
                 tuneGrid = expand.grid(k = 9), trControl = control)

# order pred values
knn_oof_pred <- knn_cv2$pred
knn_oof_pred$Resample <- as.factor(knn_oof_pred$Resample)
knn_oof_pred <- knn_oof_pred[order(knn_oof_pred$rowIndex),]

# fitting knn model
knn_model <- train(Y~. , data = quality_train_knn, method = "knn", tuneGrid = expand.grid(k = 9),
                   preProcess = c("center", "scale"))

# stack level0 models(xgboost, randomforest, KNN) with xgboost
# meta input data : xgb_cv_optiaml, rf_oof_pred, knn_oof_pred

stacking <- as.data.frame(matrix(nrow = 2560, ncol = 4, NA))
names(stacking) <- c("XGboost", "RandomForest", "KNN", "Y")
stacking$XGboost <- xgb_cv_optiaml$pred
stacking$RandomForest <- rf_oof_pred$pred
stacking$KNN <- knn_oof_pred$pred
stacking$Y <- quality_train$Y
stacking$Time_Sequence <- quality_train$Time_Sequence # imp plot에서 상위 등장했던 변수 2개 추가
stacking$x35 <- quality_train$x35
stacking <- stacking[,c(1:3,5,6,4)]
head(stacking)

# converting data into xgb format
dstacking <- xgb.DMatrix(data = as.matrix(stacking[,-6]), label = stacking$Y)

# Tuning
max_depth_grid <- seq(4,10,2)
min_child_grid <- seq(1,9,2)

xgb_tune_2 <- as.data.frame(matrix(nrow = length(max_depth_grid)*length(min_child_grid), ncol = 3, NA))
colnames(xgb_tune_2) <- c("max_depth", "min_child_weight", "WMAE")

x <- 1
for(i in max_depth_grid){
  for(j in min_child_grid){
    params_1 = list(objective = "reg:linear",
                    eta = 0.1,
                    max_depth = i,
                    min_child_weight = j,
                    colsample_bytree = 1,
                    subsample = 1)
    xgb_cv_1 <- xgb.cv(params = params_1,
                       data = dstacking,
                       nrounds = 10000,
                       verbose = 1,
                       prediction = TRUE,
                       early_stopping_rounds = 100,
                       nthread = 2,
                       maximize = FALSE,
                       feval = wmae,
                       nfold = 5)
    best_wmae <- xgb_cv_1$evaluation_log[,min(test_wmae_mean)]
    xgb_tune_2[x,1] <- i
    xgb_tune_2[x,2] <- j
    xgb_tune_2[x,3] <- best_wmae
    x <- x+1
  }
}

xgb_tune_2[which.min(xgb_tune_2$WMAE),]

# max_depth min_child_weight      WMAE
#         6                1 0.7942398

colsample_bytree <- seq(0.4, 1, 0.2)
subsample <- c(0.5, 1, 0.25)

xgb_tune_3 <- as.data.frame(matrix(nrow = length(colsample_bytree)*length(subsample), ncol = 3, NA))
colnames(xgb_tune_3) <- c("colsample_bytree", "subsample", "WMAE")

x <- 1
for(i in colsample_bytree){
  for(j in subsample){
    params_1 = list(objective = "reg:linear",
                    eta = 0.1,
                    max_depth = 6,
                    min_child_weight = 1,
                    colsample_bytree = i,
                    subsample = j)
    xgb_cv_1 <- xgb.cv(params = params_1,
                       data = dstacking,
                       nrounds = 10000,
                       verbose = 1,
                       prediction = TRUE,
                       early_stopping_rounds = 100,
                       nthread = 2,
                       maximize = FALSE,
                       feval = wmae,
                       nfold = 5)
    best_wmae <- xgb_cv_1$evaluation_log[,min(test_wmae_mean)]
    xgb_tune_3[x,1] <- i
    xgb_tune_3[x,2] <- j
    xgb_tune_3[x,3] <- best_wmae
    x <- x+1
  }
}

xgb_tune_3[which.min(xgb_tune_3$WMAE),]
# colsample_bytree subsample      WMAE
#              0.6         1 0.8281821

# colsample_bytree 와 subsample은 stacking data의 특성상 1로 설정하는 것이 나아보임

# eta search
eta <- seq(0.05, 0.5, 0.025)

xgb_tune_4 <- as.data.frame(matrix(nrow = length(eta), ncol = 2, NA))
colnames(xgb_tune_4) <- c("eta", "WMAE")

x <- 1
for(i in eta){
  params_1 = list(objective = "reg:linear",
                  eta = i,
                  max_depth = 6,
                  min_child_weight = 1,
                  colsample_bytree = 1,
                  subsample = 1)
  xgb_cv_1 <- xgb.cv(params = params_1,
                     data = dstacking,
                     nrounds = 10000,
                     verbose = 1,
                     prediction = TRUE,
                     early_stopping_rounds = 100,
                     nthread = 2,
                     maximize = FALSE,
                     feval = wmae,
                     nfold = 5)
  best_wmae <- xgb_cv_1$evaluation_log[,min(test_wmae_mean)]
  xgb_tune_4[x,1] <- i
  xgb_tune_4[x,2] <- best_wmae
  x <- x+1
}

xgb_tune_4[which.min(xgb_tune_4$WMAE),]

# eta 확인
params <-  list(objective = "reg:linear",
                eta = 0.1,
                max_depth = 6,
                min_child_weight = 1,
                colsample_bytree = 1,
                subsample = 1)

xgb_cv_stacking <- xgb.cv(params = params,
                          data = dstacking,
                          nrounds = 10000,
                          nfold = 5,
                          prediction = TRUE,
                          verbose = 1,
                          early_stopping_rounds = 500,
                          nthread = 2,
                          feval = wmae,
                          maximize = FALSE)

# final stacking model
xg_stacking_model <- xgb.train(method = "xgbTree",
                               params = params,
                               data = dstacking,
                               nrounds = 100,
                               maximize = F,
                               prediction = T,
                               verbose = TRUE,
                               feval = wmae)

mat3 <- xgb.importance (feature_names = colnames(quality_train),
                        model = xg_stacking_model)
xgb.plot.importance (importance_matrix = mat3)

# glm meta learner
elastic_net_result <- as.data.frame(matrix(nrow = 11, ncol = 3, NA))
names(elastic_net_result) <- c("alpha", "lamda", "RMSE")
x <- 1

for(i in seq(0,1,0.1)){
 cv.elasticnet <- cv.glmnet(as.matrix(stacking[,-6]), stacking$Y, nfolds = 5, alpha = i, type.measure = "mse")
 rmse <- sqrt(min(cv.elasticnet$cvm))
 lamda <- cv.elasticnet$lambda[which(cv.elasticnet$cvm==min(cv.elasticnet$cvm))]
 elastic_net_result[x,1] <- i
 elastic_net_result[x,2] <- lamda
 elastic_net_result[x,3] <- rmse
 x <- x + 1
}

# rf meta learner
WMAE <- function(data, lev = NULL, model = NULL) {
  err1 <- data[, "obs"] / sum(data[, "obs"]) 
  err2 <- abs(data[, "obs"] - data[, "pred"])
  err <- sum(err1 * err2)
  names(err) <- "WMAE"
  err
}

control <- trainControl(method = "cv", 
                        number = 5, 
                        search = "grid",
                        summaryFunction = WMAE)

tunegrid <- expand.grid(.mtry = (5)) 

rf_random <- train(Y ~ .,
                   data = stacking,
                   method = "rf",
                   tuneGrid = tunegrid, 
                   trControl = control
)

rf_random$results
# mtry      WMAE    WMAESD
# 5 0.8937063 0.1947712

# lm meta learner
lm <- lm(Y~. , data = stacking[,c(-4,-5)])
summary(lm)

pred_lm <- predict(lm, stacking[,-6])

err1 <- stacking$Y / sum(stacking$Y) 
err2 <- abs(stacking$Y - pred_lm)
err <- sum(err1 * err2)

# test set 예측

# predict holdout set with level0 model
xgb_pred <- predict(xg_model, newdata = dtest)
rf_pred <- predict(rf_model, newdata = quality_test)
knn_pred <- predict(knn_model, newdata = quality_test)

stacking_final <- as.data.frame(matrix(nrow = 300, ncol = 4, NA))
names(stacking_final) <- c("XGboost", "RandomForest", "KNN", "Y")
stacking_final$XGboost <- xgb_pred
stacking_final$RandomForest <- rf_pred
stacking_final$KNN <- knn_pred
stacking_final$Time_Sequence <- quality_test$Time_Sequence # imp plot에서 상위 등장했던 변수 2개 추가
stacking_final$x35 <- quality_test$x35
stacking_final$Y <- NULL
head(stacking_final)

# predict holdout stacking set with level1 model
dstacking_final <- xgb.DMatrix(data = as.matrix(stacking_final))
predict(xg_stacking_model, newdata = dstacking_final)

final_pred <- predict(xg_stacking_model, newdata = dstacking_final)
final_pred <- data.frame(Time_Sequence = quality_test$Time_Sequence, Y = final_pred)