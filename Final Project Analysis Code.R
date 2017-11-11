#Practical Machine Learning Course Project
##Zachary Henderson
## Date: 11 November 2017

#Loading Data
##Set up URL for download
URLTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
URLTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

##Download datasets
training <- read.csv(url(URLTrain))
testing <- read.csv(url(URLTest))

library(caret)
library(lattice)
library(AppliedPredictiveModeling)
library(doParallel)
library(rpart)

#Clean Training Data and Reduce Variables

for(i in 1:(ncol(training)-1)){
  if(class(training[, i]) == 'factor'){    
    training[, i] <- as.numeric(as.character(training[, i]))    
  }
}
#The dataset has 160 variables; however, there many unneeded variables that can be removed.
#The first seven columns are identification variables that will be removed.
training <- training[,-(1:7)]

#Next I have chosen to remove variables that have with near zero variance. This process - removing near 
#zero variance variables - removes 59 variables from each dataset leaving 94 variables.

NZV <- nearZeroVar(training)
training <- training[ , -NZV]

#Extraneous variables remain in the data, particalurly those with values of NA.
#Remove the NA variables with a mean greater than 95%.

AllNA <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[,AllNA == FALSE]
dim(training)
#This process leaves 53 variables for analysis.

##Create Training and Testing Datasets
set.seed(12345)
inTrain <- createDataPartition(training$classe, p=0.7, list = FALSE)
TrainSet<- training[inTrain, ]
TestSet <- training[-inTrain, ]
dim(TrainSet)
dim(TestSet)

#Pre-Model Analysis
#Before fitting a model to the data, it is helpful to determine what an
#expected classification should be. This will help determine how we optimize
#models. 

bookTheme(set = TRUE)
histogram(x = TrainSet$classe,
          main = "Histogram of Classe of Exercise in Training Dataset",
          xlab = "Classe of Exercise",
          ylab = "Frequency in Training Data")
#The histogram above indicates that each classes is within an order of 
#magnitude of each other and demonstrates that the variable we are trying to model - 
#classe of exercise - is relatively balanced in the dataset. 

##Work on fixing
#Correlation Analysis
#A correlation analysis is also useful before proceeding to modeling and analysis.
#correlationMatrix <- cor(TrainSet[, -54])
#corrplot(correlationMatrix, order ="FPC", method = "color", type = "lower",
       #  tl.cex =0.8, tl.col =rgb(0, 0, 0))

#Prediction Model Building
#Three methods will by applied to the training data set and the one with the highest accuarcy
#will be applied to the test dataset for the quiz predictions. The methods utilized will be 
#random forest, decision tree, and generalized boosted models.

#Random Forest
registerDoParallel()
##Set up a control function for Random Forest
controlRF <- trainControl(method="cv", 2, savePredictions = "final")
##Model the Random Forest algorithm
modelRF <- train(classe~., data = TrainSet, method ="rf",
                 trControl=controlRF)
##Test model against test data
predict_rf <- predict(modelRF, TestSet)
##Determine level of accuracy
confusionMatrix(predict_rf, TestSet$classe)$overall[1]
##Accuracy: 0.9916737


#Decision Trees
modelDF <- train(classe~., data = TrainSet, method = "rpart")
predict_df <- predict(modelDF, TestSet)
confusionMatrix(predict_df, TestSet$classe)$overall[1]
##Accuracy: 0.4963466

#Generalized Boosted Model
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats =1)
modelGBM <-train(classe~., data = TrainSet, method = "gbm", trControl = controlGBM, verbose = FALSE)
predict_gbm <- predict(modelGBM, TestSet)
confusionMatrix(predict_gbm, TestSet$class)$overall[1]
##Accuracy: 0.9571793

#Of the three models, the random forest model performed with the highest in-sample accuracy.

#Conduct the predictions of the test dataset using the random forest model.

predict_rf_final <- predict(modelRF, testing)
predict_rf_final
#Predictions: BABAAEDBAABCBAEEABBB
