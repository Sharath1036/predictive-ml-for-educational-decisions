df <- read.csv("E:/Machine Learning/Datasets/College_Regression.csv")

library(fastDummies)
df <- dummy_cols(df, select_column="hostel")
View(df)

# Removing the columns "hostel" and "hostelNone"
which(colnames(df) == "hostel")
which(colnames(df) == "hostel_NONE")
df <- df[,-18]
df <- df[,-12]

# df$intake <- round(df$intake, digits=1)

# Missing value imputation
df$cutoff[is.na(df$cutoff)] <- mean(df$cutoff, na.rm=TRUE)
df$intake[is.na(df$intake)] <- mean(df$intake, na.rm=TRUE)
df$fees[is.na(df$fees)] <- mean(df$fees, na.rm=TRUE)
df$salary[is.na(df$salary)] <- mean(df$salary, na.rm=TRUE)
df$rating[is.na(df$rating)] <- mean(df$rating, na.rm=TRUE)
df$infrastructure[is.na(df$infrastructure)] <- mean(df$infrastructure, na.rm=TRUE)
df$faculty[is.na(df$faculty)] <- mean(df$faculty, na.rm=TRUE)
df$crowd[is.na(df$crowd)] <- mean(df$crowd, na.rm=TRUE)
View(df)

library(neuralnet)

nn <- neuralnet(
  cutoff ~ crowd + fees + autonomous + new_ban,
  data = df,
  hidden = c(5, 3),
  act.fct = "logistic",
  linear.output = FALSE
)

summary(nn)
plot(nn) # FIG 12

require("caTools")
set.seed(0)
split <- sample.split(df, SplitRatio = 0.7)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

test$nn <- predict(nn, test)
View(test)
MSEnn <- mean((test$cutoff-test$nn)^2)
MSEnn # 397.06

## Saving the plots
pdf(file="E:/Machine Learning/Plots/college/nn.pdf")
plot(nn)
dev.off()