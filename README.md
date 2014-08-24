CourseraProjectSubmission
=========================

CourseraProjectSubmission-Machine Learning
library(gdata)                  
library(caret)
require(randomForest)

trainin = read.csv("pmltraining.csv") 
testing = read.csv("pmltesting.csv") 

mymodel <- randomForest(trainin$classe ~ raw_timestamp_part_1+	num_window+	roll_belt+	pitch_forearm+	magnet_dumbbell_z+	yaw_belt+	pitch_belt+	magnet_dumbbell_y+	roll_forearm+	magnet_dumbbell_x+	accel_dumbbell_y+	accel_belt_z+	roll_dumbbell+	accel_forearm_x+	magnet_belt_z+	magnet_belt_y+	accel_dumbbell_x+	magnet_forearm_x+	gyros_dumbbell_y+	magnet_arm_y, data = trainin, ntree = 100)

predict(mymodel, testing)

gbmImp <- varImp(mymodel , scale = FALSE)
# Variable Importance of variables was found and top 20 variables of highest importance were selected

gbmImp
mymodel
#The Model created mymodel has the below Confusion matrix:
#     A    B    C    D    E  class.error
#A 5580    0    0    0    0 0.0000000000
#B    1 3796    0    0    0 0.0002633658
#C    0    4 3418    0    0 0.0011689071
#D    0    0    1 3214    1 0.0006218905
#E    0    0    0    0 3607 0.0000000000
 
 # To check for out of sample error, K fold classification was used with K = 5
 
 library(plyr)
library(randomForest)
 
data <- trainin
 
# in this cross validation  we use the trainin set to
# predict the Classe from the other variables in the dataset
# with the random forest model
 
k = 5 
 
# sample from 1 to k, nrow times (the number of observations in the data)
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k
 
# prediction and testset data frames that we add to with each iteration over
# the folds
 
prediction <- data.frame()
testsetCopy <- data.frame()
 summary(testsetCopy)
#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)
 
for (i in 1:k){
# remove rows with id i from dataframe to create training set
# select rows with id i to create test set
trainingset <- subset(data, id %in% list[-i])
testset <- subset(data, id %in% c(i))
# run a random forest model
mymodel <- randomForest(trainingset$classe ~ raw_timestamp_part_1+	num_window+	roll_belt+	pitch_forearm+	magnet_dumbbell_z+	yaw_belt+	pitch_belt+	magnet_dumbbell_y+	roll_forearm+	magnet_dumbbell_x+	accel_dumbbell_y+	accel_belt_z+	roll_dumbbell+	accel_forearm_x+	magnet_belt_z+	magnet_belt_y+	accel_dumbbell_x+	magnet_forearm_x+	gyros_dumbbell_y+	magnet_arm_y, data = trainingset, ntree = 100)
# remove response column 160, Classe
temp <- as.data.frame(predict(mymodel, testset[,-160]))
# append this iteration's predictions to the end of the prediction data frame
prediction <- rbind(prediction, temp)
# append this iteration's test set to the test set copy data frame
# keep only the Classe Column
testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,160]))
progress.bar$step()
}
 
# add predictions and actual Classe  values
result <- cbind(prediction, testsetCopy[, 1])
names(result) <- c("Predicted", "Actual")
result$Difference <- abs(if (result$Actual == result$Predicted){0} else {1} )
 
# As an example use Mean Absolute Error as Evalution
summary(result$Difference)

# Output Model had very high accuracy across different training and test samples

