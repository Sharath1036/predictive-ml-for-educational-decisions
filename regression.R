df <- read.csv("E:/Machine Learning/Datasets/College_Regression.csv")
View(df)
str(df)
summary(df)

# Creating dummy variables for categorical variables
library(fastDummies)
df <- dummy_cols(df, select_column="hostel")
View(df)

# Removing the columns "hostel" and "hostelNone"
which(colnames(df) == "hostel")
which(colnames(df) == "hostel_NONE")
df <- df[,-18]
df <- df[,-12]

# Missing value imputation
df$cutoff[is.na(df$cutoff)] <- mean(df$cutoff, na.rm=TRUE)
df$intake[is.na(df$intake)] <- mean(df$intake, na.rm=TRUE)
df$fees[is.na(df$fees)] <- mean(df$fees, na.rm=TRUE)
df$salary[is.na(df$salary)] <- mean(df$salary, na.rm=TRUE)
df$rating[is.na(df$rating)] <- mean(df$rating, na.rm=TRUE)
df$infrastructure[is.na(df$infrastructure)] <- mean(df$infrastructure, na.rm=TRUE)
df$faculty[is.na(df$faculty)] <- mean(df$faculty, na.rm=TRUE)
df$crowd[is.na(df$crowd)] <- mean(df$crowd, na.rm=TRUE)

summary(df)
df$intake <- round(df$intake, digits=1)

pairs(~cutoff+courses+intake+fees+salary, data=df) # FIG 1
# courses, intake, fees and rank have outliers
# salary has some functional (logarithmic) relationship with cutoff

# Converting the log relation to linear relation
# df$salary <- log(1+df$salary)
# plot(df$cutoff, df$salary) # FIG 2

## Capping and flooring treatment for outliers
# Capping for courses
quantile_value <- quantile(df$courses, 0.75)
UV <- 1.5 * quantile_value
df$courses[df$courses > UV] <- UV
summary(df$courses)

# Capping for salary
quantile_value <- quantile(df$salary, 0.75)
UV <- 1.5 * quantile_value
df$salary[df$salary > UV] <- UV
summary(df$salary)

# Flooring for cutoff
quantile_value <- quantile(df$cutoff, 0.25)
LV <- 0.75 * quantile_value
df$cutoff[df$cutoff < LV] <- LV
summary(df$cutoff)

# Winsorization for intake
quantile_value <- quantile(df$intake, 0.75)
UV <- 1.5 * quantile_value
df$intake[df$intake > UV] <- UV
summary(df$intake)

quantile_value <- quantile(df$intake, 0.25)
LV <- 0.75 * quantile_value
df$intake[df$intake < LV] <- LV
summary(df$intake)

# Winsorization for fees
quantile_value <- quantile(df$fees, 0.75)
UV <- 1.5 * quantile_value
df$fees[df$fees > UV] <- UV
summary(df$fees)

quantile_value <- quantile(df$fees, 0.25)
LV <- 0.75 * quantile_value
df$fees[df$fees < LV] <- LV
summary(df$fees)

# Finding the significant variables w/o test-train split, say
## LINEAR REGRESSION
library(LiblineaR)
linear <- lm(cutoff~., data=df)
summary(linear)
plot(linear, which = 2) # FIG 2

# Here 'crowd' and 'new_ban' are more significantly impacting cutoff. 
# 'government' and 'autonomous' also can be considered for analysis.
# 'fees' had some outliers but still it seems to have some relation with cutoff. 

## TEST-TRAIN SPLIT
library(caTools)
set.seed(0)
split <- sample.split(df, SplitRatio=0.7)
train <- subset(df, split==TRUE)
test <- subset(df, split==FALSE)

## LINEAR REGRESSION
linear <- lm(cutoff~., data=train)
summary(linear)
test$linear <- predict(linear, test)
View(test)
MSELinear <- mean((test$cutoff-test$linear)^2)
MSELinear # 187.63

## SUBSET SELECTION ##
require("leaps")
View(df)
lm_subset <- regsubsets(cutoff~ .-college_code, data = df, nvmax = 15) # Creates 2^p subsets for p variables. Here there are 15 variables and the regsubset can hold upto 8 variables, so nvmax is used for p>8
summary(lm_subset)
# The best model is 'new_ban' followed by 'autonomous', 'crowd' and 'fees'. We'll use these subsets for further analysis

summary(lm_subset)$adjr2
which.max(summary(lm_subset)$adjr2)
# We get the best model with 9 variables as our highest adj R^2 value

coef(lm_subset, 11)
# The coefficients of the variables selected model are: division, courses, fees, salary, crowd, government, autonomous, new_ban, hostel_BOTH

## RIDGE AND LASSO SHRINKAGE
require("glmnet")
x <- model.matrix(cutoff~.-college_code, data = df)[, -1]
y <- df$cutoff
grid <- 10^seq(10, -2, length = 30)
grid 

# RIDGE
lm_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(lm_ridge)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit) # FIG 3
opt_lambda <- cv_fit$lambda.min
opt_lambda # 3.03

tss <- sum(y - mean(y)^2)
y_a <- predict(lm_ridge, s = opt_lambda, newx = x)
rss <- sum((y - y_a)^2)
rsq <- 1 - rss/tss
rsq # 1.029592

# Lasso
lm_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(lm_lasso)
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit) # FIG 5
opt_lambda <- cv_fit$lambda.min
opt_lambda # 0.17

tss <- sum(y - mean(y)^2)
y_a <- predict(lm_lasso, s = opt_lambda, newx = x)
rss <- sum((y - y_a)^2)
rsq <- 1 - rss/tss
rsq # 1.029325

## DECISION TREE
library(rpart)
library(rpart.plot)
regtree <- rpart(cutoff~.-college_code, data=train, control=rpart.control(maxdepth = 10))
rpart.plot(regtree, digits=-3) # FIG 6
printcp(regtree)
# Predicting values on test set
test$decision <- predict(regtree, test, type="vector")
View(test)
MSEdecision <- mean((test$decision - test$cutoff)^2)
MSEdecision # 201.13

## FULLTREE
fulltree <- rpart(cutoff~.-college_code, data=train, control=rpart.control(cp=0))
rpart.plot(fulltree, digits=-3) # FIG 7
printcp(fulltree)
test$fulltree <- predict(fulltree, test, type="vector")
View(test)
MSEfull <- mean((test$fulltree - test$cutoff)^2)
MSEfull # 201.32

## PRUNED TREE
mincp <- regtree$cptable[which.min(regtree$cptable[,"xerror"]), "CP"]
mincp # 0.02912
prunedtree<-prune(fulltree, cp= mincp )
rpart.plot(prunedtree, digits=-3) # FIG 8
test$pruned <- predict(prunedtree, test, type="vector")
View(test)
MSEpruned <- mean((test$pruned - test$cutoff)^2)
MSEpruned # 190.47

## BAGGING
library(randomForest)
set.seed(0)
bagging <- randomForest(cutoff~.-college_code, data=train, mtry=15)
test$bagging <- predict(bagging, test)
View(test)
MSEbagging <- mean((test$bagging - test$cutoff)^2)
MSEbagging # 162.108

## RANDOM FOREST
randomforest <- randomForest(cutoff~.-college_code, data=train, ntree=500)
test$randomforest <- predict(randomforest, test)
View(test)
MSErandomforest <- mean((test$randomforest - test$cutoff)^2)
MSErandomforest # 157.85

## GRADIENT BOOST
library(gbm)
set.seed(0)
boosting <- gbm(cutoff~.-college_code, data=train, distribution="gaussian", n.trees=5000, interaction.depth=4, shrinkage=0.2, verbose = F)
test$boost <- predict(boosting, test, n.trees=5000)
View(test)
MSEboost <- mean((test$boost-test$cutoff)^2)
MSEboost # 193.24

## SUPPORT VECTOR MACHINES
library (e1071)
svmfit <- svm(cutoff~.-college_code, data=train , kernel = "linear", cost =0.1, scale = TRUE )
summary (svmfit)
## Predicting on test set
test$svm <- predict (svmfit,test)
MSEsvm <- mean((test$svm-test$cutoff)^2)
MSEsvm # 186.08
plot(test$svm,test$cutoff)

## Saving the plots
png(filename="E:/Machine Learning/Plots/college plots/prunedtree.png")
rpart.plot(prunedtree, digits=-3)
dev.off()

## CONCLUSION: The Random Forest Model provides the least error rate of 217.78 on the test set and its predicted values are best suited for analysis