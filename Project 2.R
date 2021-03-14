#clean R environment
rm(list=ls())
#loadinng libraries
x = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees", "rpart",'MASS','xgboost','stats')
#load Packages
lapply(x, require, character.only = TRUE)
str(train)
library(dummies)
library(tidyverse)
library(caret)
library(usdm)
library(mlbench)

#setting working directory
setwd("/Users/alfazalm/Documents/carfare")
#loading csv fiile
train=read.csv("train_cab.csv",header = T)
test=read.csv("test.csv",header = T)
test2 = read.csv("test.csv",header = T)


#creating columns with date,day,month,year and hour of pickup from the column "pickup_date_time".
train$pickup_date = as.Date(as.character(train$pickup_datetime))
train$pickup_weekday = as.factor(format(train$pickup_date,"%u"))
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))
train$pickup_hour = as.factor(format(strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S"),"%H"))
#same thing does for test data
test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))
test$pickup_hour = as.factor(format(strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S"),"%H"))
str(train)
#changing data type of fare amount to numeric
train$fare_amount=as.numeric(train$fare_amount)
#fare_amount should not be zero and less than zero
train=train[-which(train$fare_amount == 0 ),]
train=train[-which(train$fare_amount < 0 ),]
#Records with No.of passengers less than 1 shuold be removed
train=train[-which(train$passenger_count < 1 ),]
#Records with no. of passengers greater than 6 should be removed
train=train[-which(train$passenger_count > 6 ),]
#Passenger count shoul be integer checking all values are integer
unique(train$passenger_count)
unique(test$passenger_count)
#There is a value 1.3 it should be removed
train = train[-which(train$passenger_count == 1.3),]
unique(train$passenger_count)
#Latitude should be in range of -90 to 90
#checking the range
range(train$pickup_latitude)
#There are values grearter than 90 So it need to be removed
train = train[-which(train$pickup_latitude > 90),]
range(train$dropoff_latitude)
#The values are in range of -90 to 90
range(test$pickup_latitude)
#The values are in range of -90 to 90
range(test$dropoff_latitude)
#The values are in range of -90 to 90

#longitude should be in range of -180 to 180
range(train$pickup_longitude)
#The values are in range of -180to 180
range(train$dropoff_latitude)
#The values are in range of -180to 180
range(test$pickup_longitude)
#The values are in range of -180to 180
range(test$dropoff_longitude)
#The values are in range of -180to 180
var = c("fare_amount", "pickup_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude",
        "passenger_count","pickup_date","pickup_weekday","pickup_mnth","pickup_yr","pickup_hour")
####Missing Value Analysis####
missing_value = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_value$percentage = 0
missing_value$percentage = (missing_value[1]/nrow(train)) * 100
names(missing_value)[1] =  "Missing_values"
names(missing_value)[2] = "Percentage"
# There is no missing value in "Pickup_datetime" but there are missing values in the column we created from the same column
#let's checkc it
mi = which (is.na(train$pickup_date))
#let's check the pickup_datetime of that particular observation with pickup_date null value
train$pickup_datetime[mi]
#it's found pickup_datetime is 43 for an observation that should be removed
train=train[-(mi),]
#Now check the missing values and missing percentage once again
missing_value = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_value$percentage = 0
missing_value$percentage = (missing_value[1]/nrow(train)) * 100
names(missing_value)[1] =  "Missing_values"
names(missing_value)[2] = "Percentage"
####Missing value imputation for passenger count
# Mode method and knn imputation will be checked for a sample value and the best method will be used to impute the missing value
#taking random index of 500 and value and both method will be checked
value_p = train$passenger_count[500]
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
mode_p = getmode(train$passenger_count)
train2 = train
str(train2)
#removing some character and factor columns for knnImputation
train2$passenger_count[500] = NA
train2 = subset(train2, select = -c(pickup_datetime,pickup_date,pickup_weekday,pickup_mnth,pickup_yr))
train2 = knnImputation(train2, k = 181)
knn_p = train2$passenger_count[500]
# Actual value =3
# mode = 1
#knn imputation = 2.39
# mode gives value of 1 and knn imputation gives value 2.39 and actual value is 3 so knn_imputation is used for imputing missing value
train$passenger_count = train2$passenger_count
#passenger count should be integers so we will round the value
train$passenger_count=round(train$passenger_count)
#####Missing value imputation for fare amount
value_f = train$fare_amount[500]
train2 = train
mode_f = getmode(train$fare_amount)
train2$passenger_count[500] = NA
train2 = subset(train2, select = -c(pickup_datetime,pickup_date,pickup_weekday,pickup_mnth,pickup_yr))
train2 = knnImputation(train2, k = 181)
knn_f = train2$fare_amount[500]
# Actual value =6
# mode = 6.5
#knn imputation = 6
# mode gives value of 6.5 and knn imputation gives value 6 and actual value is 6 so knn_imputation is used for imputing missing value
train$fare_amount = train2$fare_amount
#outlier analysis is done after converting latitude,longitude variables
#to distance because distance is used for modelling
###########Converting latitude and longitude variable to distance

my_dist <- function(long1, lat1, long2, lat2) {
  rad = pi/180
  a1 = lat1*rad
  a2 = long1*rad
  b1 = lat2*rad
  b2 = long2*rad
  dlon = b2 - a2
  dlat = b1 - a1
  a = (sin(dlat/2))^2 + cos(a1)*cos(b1)*(sin(dlon/2))^2
  c= 2*atan2(sqrt(a), sqrt(1 - a))
  R =6378137
  d = R*c
  return(d)
}
train$distance = my_dist(train$pickup_longitude,train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude)
test$distance = my_dist(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)
###outlier Analysis
boxplot(train$fare_amount,
        names = c("Fare Amount"),
        las = 2,
        col = c("blue"),
        border = "brown",
        horizontal = FALSE,notch = FALSE
)
boxplot(train$distance,
        names = c("Distance"),
        las = 2,
        col = c("blue"),
        border = "brown",
        horizontal = FALSE,notch = FALSE
)
#There are outliers in fare amount and distance
outliers_f=boxplot(train$fare_amount, plot=FALSE)$out
outliers_d=boxplot(train$distance, plot=FALSE)$out
#put NA values in outliers
train[,'fare_amount'][train[,'fare_amount'] %in% outliers_f] = NA
train[,'distance'][train[,'distance'] %in% outliers_d] = NA
#imputing in outlier values
train3 = subset(train, select = -c(pickup_datetime,pickup_date,pickup_weekday,pickup_mnth,pickup_yr))
train3 = knnImputation(train3, k = 181)
train$fare_amount=train3$fare_amount
train$distance=train3$distance
#checking whether there is missing values
sum(is.na(train))

##Adding new feature time depending on pickup hour column
train$pickup_hour=as.integer(train$pickup_hour)

test$pickup_hour=as.integer(test$pickup_hour)

train$time[train$pickup_hour>=4 & train$pickup_hour<=10]=1 #'Morning'
train$time[train$pickup_hour>10 & train$pickup_hour<=16]=2 #'Day'
train$time[train$pickup_hour>16 & train$pickup_hour<=22]=3 #'Night'
train$time[train$pickup_hour>22 | train$pickup_hour<5]=4 #'Midnight'

test$time[test$pickup_hour>=4 & test$pickup_hour<=10]=1 #'Morning'
test$time[test$pickup_hour>10 & test$pickup_hour<=16]=2 #'Day'
test$time[test$pickup_hour>16 & test$pickup_hour<=22]=3 #'Night'
test$time[test$pickup_hour>22 | test$pickup_hour<5]=4 #'Midnight'

##Removing variables that we used for feature engineering
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
train = subset(train,select = -c(pickup_datetime))
test = subset(test,select = -c(pickup_datetime))

###Feature selection

##Correlation plot for numeric variables
n = sapply(train,is.numeric)

numeric = train[,n]

cnames = colnames(numeric)

corrgram(train[,cnames], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
##There is no enough correlation to remove variables
###Anova test for categorical variables
aov_results = aov(fare_amount ~ passenger_count + pickup_hour + pickup_weekday + pickup_mnth + pickup_yr,data = train)

summary(aov_results)
#Pickup_weekday has a p value greater than 0.05, so null hypotheis true and the variable will be omitted
train = subset(train,select=-pickup_weekday)
test = subset(test,select=-pickup_weekday)
#####Feature Scaling
hist(train$fare_amount)
hist(train$distance)
#Distance is right skewed Normalisation required
train[,'distance'] = (train[,'distance'] - min(train[,'distance']))/
  (max(train[,'distance'] - min(train[,'distance'])))

#Drop unwanted variables
train = subset(train,select=-c(pickup_hour,pickup_date))
test = subset(test,select=-c(pickup_hour,pickup_date))
#####Modeling####
#Adding Dummy variables for categorical and factor variables for better performance of model
categ2 = c('time','pickup_yr','pickup_mnth')
train = dummy.data.frame(train, categ2)
test = dummy.data.frame(test, categ2)
##Splitting train data to 80:20 ratio for model performance checking
train_index=sample(1:nrow(train),0.8*nrow(train))
train_80= train[train_index,]
train_20= train[-train_index,]
#Linear Regression
vif(train[,-1])
vifcor(train[,-1],th=0.8)
# train_20 values without target variable
train_20_var= subset(train_20, select=-fare_amount)
##It shows there is no collinearity problem
model_lr = lm(fare_amount~., train_80)
predictions_lr = predict(model_lr,train_20_var)
regr.eval(train_20$fare_amount,predictions_lr)
#mae       mse      rmse      mape 
#1.5840254 5.1925158 2.2787092 0.1918588

###Decision Tree
model_dt = rpart(fare_amount ~ .,data=train_80,method="anova")

predictions_dt = predict(model_dt,train_20_var)

regr.eval(train_20$fare_amount,predictions_dt)
#mae       mse      rmse      mape 
#1.7557661 5.9000076 2.4289931 0.2201762 

#Random Forest
model_rf = randomForest(fare_amount~., train_80, ntree = 500, importance = TRUE)
predictions_rf = predict(model_rf,train_20_var)
regr.eval(train_20$fare_amount,predictions_rf)
#mae       mse      rmse      mape 
#1.5664307 4.9213228 2.2184055 0.1915134

##Best method is random forest where accuracy is better
##We are tuning the random forest algorithm to find out the best mtry value using algorithm tools


seed = 7
metric = "Accuracy"

x1=train_80[,-1]
x2=train_80[,1]
# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry= tuneRF(x1, x2, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)
### error is less with mtry value 8
model_rf2 = randomForest(fare_amount~., train_80, ntree = 500,mtry = 8, importance = TRUE)
predictions_rf2 = predict(model_rf2,train_20_var)

regr.eval(train_20$fare_amount,predictions_rf2)
##it is found with mtry value 8 accuracy decreasing so the default random forest method is more accurate
  ###########Finalising model and saving model and applying on test dataset######
test_pickup_datetime = test2$pickup_datetime
predictions_rf3 = predict(model_rf,test)
predictions = data.frame(test_pickup_datetime,"predictions" = predictions_rf3)
# save the predicted fare_amount in disk as .csv format 
write.csv(predictions,"predictions_R1.csv",row.names = FALSE)

summary(model_lr)
summary(model_dt)