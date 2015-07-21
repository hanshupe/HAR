library(caret)
library(ggplot2)
library(Hmisc)
library(RCurl)




# The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 
# 
# 1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
# 2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 
# 
set.seed(12345)
temp <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", ssl.verifypeer = FALSE)
raw_training_data <- read.csv(textConnection(myCsv))

temp <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", ssl.verifypeer = FALSE)
raw_validation_data <- read.csv(textConnection(myCsv))

summary(raw_training_data)

# Partition training data into a training and testing dataset.

# Returns a list of variables with their labeling of near-zero-variable
# I inspected the list manually and determined that all variables were
# worth eliminating from the dataset for building the training model.
nsv <- nearZeroVar(raw_training_data,saveMetrics=TRUE)

# Returns column positions for near-zero-value variables
nsv_positions <- nearZeroVar(raw_training_data,saveMetrics=FALSE)

# Using the position, we filter out the variables that were near-zero-value.
filtered_training <- raw_training_data[-c(nsv_positions)]

# Filter variables related to time. We are not looking at time windows since we want to predict classe using the sensor readings, which have nothing to do with time.
excluding_vars <- names(filtered_training) %in% c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")
filtered_training  <- filtered_training[!excluding_vars]

# Filter variables that are mostly NA, which were covariates produced by the research team. These are not relevant to our investigation since we are looking 
# into predicting the classe given by an instantaenous movement.
exclude_cols <- grep("^var|^avg|^max|^min|^std|^amplitude",names(filtered_training))
filtered_training <- filtered_training[-c(exclude_cols)]

# Finding correlated variables so that we may exclude one of the highly correlated pairs
filtered_training_no_class <- filtered_training[-c(dim(filtered_training))]
correlated_cols_to_exclude <- findCorrelation(cor(filtered_training_no_class), cutoff= 0.75)
filtered_training <- filtered_training[-c(correlated_cols_to_exclude)]

# List of remaining variables after preprocessing
print(names(filtered_training))

# featurePlot(x=filtered_training,y=filtered_training,plot="pairs")

filtered_partition = createDataPartition(filtered_training$classe, p=0.75, list=F)

training <- filtered_training[filtered_partition,]
probe <- filtered_training[-filtered_partition,]

classeFit <- train(training$classe ~., data=training, method="rf",prox=TRUE)
# Printout confusion matrix and OOB estimated error
# randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 17
# 
# OOB estimate of  error rate: 0.01%
# Confusion matrix:
#   A    B    C    D    E  class.error
# A 4185    0    0    0    0 0.0000000000
# B    1 2847    0    0    0 0.0003511236
# C    0    1 2566    0    0 0.0003895598
# D    0    0    0 2412    0 0.0000000000
# E    0    0    0    0 2706 0.0000000000
classeFit$finalModel

# Execute Prediction on probe dataset
pred <- predict(classeFit,probe)

# Get confussion matrix for prediction on probe
probe$predRight <- pred == probe$classe
table(pred,probe$classe)
# 
# pred    A    B    C    D    E
# A 1395    0    0    0    0
# B    0  949    0    0    0
# C    0    0  855    0    0
# D    0    0    0  804    0
# E    0    0    0    0  901

test_set <- raw_validation_data[-c(nsv_positions)]
test_set <- test_set[!excluding_vars]
test_set <- test_set[-c(exclude_cols)]
test_set <- test_set[-c(correlated_cols_to_exclude)]

predOnTest <- predict(classeFit,test_set)
# B A B A A E D B A A B C B A E E A B B B
# B A B A A E D B A A B C B A E E A B B B
predOnTest

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("answers//problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predOnTest)