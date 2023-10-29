df <- read.csv("E:/Machine Learning/Datasets/College_Classification.csv")
View(df)
str(df)

library(fastDummies)
df <- dummy_cols(df, select_column="hostel")
View(df)
which(colnames(df) == "hostel")
which(colnames(df) == "hostel_NONE")
df <- df[,-19]
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

# Rounding off the intake variable values upto 1 decimal places
df$intake <- round(df$intake, digits=1)

# Finding accuracy of any model w/o test-train split, say
# Linear Discriminant Analysis
require("MASS")
lda.fit <- lda(admission~., data = df)
lda.fit
lda.pred <- predict(lda.fit, df)
lda.pred$posterior
lda.class <- lda.pred$class
table(df$admission,lda.class) 
sum(lda.pred$posterior[,1]>0.8)
354/387 # 91.47

# Test-train split
require("caTools")
set.seed(0)
split <- sample.split(df, SplitRatio = 0.7)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# Logistic regression
train.fit <- glm(admission~. -college_code, data=train, family=binomial)
test.probs <- predict(train.fit, test, type='response')
test.pred <- rep('NO', 130)
test.pred[(test.probs > .5)] = 'YES'
table(test.pred, test$admission)
116/130 # 89.23%

# Linear Discriminant Analysis
require("MASS")
trainLDA <- lda(admission~.-college_code, data=train)
test.predict <- predict(trainLDA, test)
test.predict$posterior
test.class <- test.predict$class
table(test.class, test$admission)
115/130 # 88.46%

# K nearest neighbour
require("class")
trainX <- train[,-16]
testX <- test[,-16]
trainY <- train$admission
testY <- test$admission

trainX_s <- scale(trainX)
testX_s <- scale(testX)
set.seed(0)


# k=3
knn.pred <- knn(trainX_s, testX_s, trainY, k=15)
table(knn.pred,testY)
112/130 # 86.15%

# k=5
113/130 # 86.92%

# k=7
114/130 # 87.69%

# k=10
115/130 # 88.46% (highest)

# k=8
113/130

# k=16
111/130 # 85.38%

# DECISION TREE
library(rpart)
library(rpart.plot)
classtree <- rpart(admission~.-college_code, data=train, method="class", control=rpart.control(maxdepth = 6))
rpart.plot(classtree, digits=-3) # FIG 10

# Predicting values on test set
test$decision <- predict(classtree, test, type="class")
View(test)
table(test$admission, test$decision)
111/130 # 85.38%

## BAGGING
library(randomForest)
train$admission <- as.factor(train$admission)
bagging <- randomForest(admission~.-college_code, method="class", data=train, mtry=16)
test$bagging <- predict(bagging, test, type="response")
View(train)
View(test)
table(test$admission, test$bagging) 
112/130 # 86.15%

## RANDOM FOREST
randomforest <- randomForest(admission~., data=train, ntree=500)
test$randomforest <- predict(randomforest, test, type="class")
View(test)
table(test$admission, test$randomforest)
115/130 # 88.46%

# ## GRADIENT BOOST
# library(gbm)
# boosting <- gbm(admission~.-college_code, data=train, distribution="bernoulli", n.trees=5000, interaction.depth=4, shrinkage=0.2, verbose = F)
# test$gradient <- predict(boosting, test, type="response")
# View(test)
# table(test$admission, test$gradient)

## ADAPTIVE BOOST
library(adabag)
adaboost <- boosting(admission~.-college_code, data=train, boos=TRUE)
predada <- predict(adaboost, test)
table(predada$class, test$admission) 
119/130 # 91.53%
t1 <- adaboost$trees[[5]]
plot(t1)
text(t1, pretty=100) # FIG 11

## XG BOOST
library(xgboost)
trainY <- train$admission=="1"
trainX <- model.matrix(admission~.-1, data=train)
View(trainX)
trainX <- trainX[,-13]

testY <- test$admission=="1"
testX <- model.matrix(admission~.-1, data=test)
testX <- testX[,-13]

Xmatrix_train <- xgb.DMatrix(data=trainX, label=trainY)
Xmatrix_test <- xgb.DMatrix(data=testX, label=testY)
Xgboosting <- xgboost(data=Xmatrix_train, nround=50, objective ="multi:softmax", eta=0.3, num_class=2, max_depth=100)
xgpred <- predict(Xgboosting, Xmatrix_test)
table(testY, xgpred) 
127/136 # 93.38%

## SUPPORT VECTOR MACHINES
library(e1071)
svmfit = svm (admission~.-college_code, data=train , kernel = "linear", cost =1 ,scale = TRUE)
 
## To check the support vectors
svmfit$index

## Finding best value of C / Tuning the hyperparameter
set.seed (0)
tune.out = tune(svm, admission~.,data=train ,kernel="linear", ranges =list(cost=c(0.001 , 0.01, 0.1, 1,10,100)))
bestmod = tune.out$best.model
summary (bestmod)
ypredL=predict (bestmod ,test)
table(predict=ypredL , truth=test$admission) 
116/130 # 89.23%
# The best model comes at cost=0.1

## Polynomial Kernel
svmfitP = svm(admission~.-college_code, data=train , kernel ="polynomial", cost=1, degree=2)
# Hyperparameter Tuning
tune.outP=tune(svm ,admission~.,data=train, cross = 4, kernel="polynomial", ranges =list(cost=c(0.001,0.1, 1,5,10),degree=c(0.5,1,2,3,5,10) ))
bestmodP =tune.outP$best.model
summary (bestmodP)
ypredP = predict (bestmodP,test)
table(predict = ypredP, truth = test$admission)
115/130 # 88.46%

# The best model comes at cost=1 and degree=1. This indicates that there is a linear relationship between admission and other variables
# From the classification analysis we can conclude that the best accuracy comes out from the linear kernel and the best cost occurs at cost=0.1

## Saving the plots
png(filename="E:/Machine Learning/Plots/college plots/adaboost.png")
plot(t1)
text(t1, pretty=100)
dev.off()

## CONCLUSION: The Adaptive Boost model provides the best accuracy of 91.53% on the test set and its predicted values are best suited for analysis