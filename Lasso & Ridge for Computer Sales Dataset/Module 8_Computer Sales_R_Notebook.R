# Lasso & Ridge Regression

### Problem Statement:- 

  - Build a model to predict the computer sales

### Data Understanding

```{r}
library(readr)
Computer_Data <- read_csv("/Users/thanush/Desktop/Digi 360/Module 8/Datasets-7/Computer_Data.csv")
head(Computer_Data)
```
```{r}
#Attaching the dataset
attach(Computer_Data)
```

### Data Cleaning
```{r}
#Replace categorical values with dummy values
library(plyr)
Computer_Data$cd <- revalue(Computer_Data$cd, c("yes"=1, "no"=0))
Computer_Data$multi <- revalue(Computer_Data$multi, c("yes"=1, "no"=0))
Computer_Data$premium <- revalue(Computer_Data$premium, c("yes"=1, "no"=0))
head(Computer_Data)
```

```{r}
#Remove the first column since it is serial number.
Computer_Data$X1 <- NULL
head(Computer_Data)
```
```{r}
# Converting the datatype as numeric for categorical fatures
Computer_Data$cd <- as.numeric(Computer_Data$cd)
Computer_Data$multi <- as.numeric(Computer_Data$multi)
Computer_Data$premium <- as.numeric(Computer_Data$premium)
head(Computer_Data)
```

### Data Visualization

```{r}
plot(Computer_Data)
```
```{r}
# Finding the Corrleation Coefficient
cor(Computer_Data)
```

```{r}
#Predicting the profit of the startups
pf1 <- mean(Computer_Data$price)
pf1
```

```{r}
# Error in Prediction
# AV - PV
err1 <- Computer_Data$price - mean(Computer_Data$price)
```

```{r}
# RMSE
MSE1 <- mean(err1^2)
MSE1
```


### Building Linear Regression Model

```{r}
m1 <- lm(price~speed+hd+ram+screen+cd+multi+premium+ads+trend)
summary(m1)
```

```{r}
#Predictions
predpf2 <- predict(m1, data=Computer_Data)
MSE2 <- mean(m1$residuals)^2
```

```{r}
# Residual plot
plot(Computer_Data$price, predpf2)

```

```{r}
barplot(sort(m1$coefficients), ylim=c(-0.5, 5))
```

### Regularization Methods
```{r}
# Converting the data into compatible format in which model accepts 
cd_x <- model.matrix(price~.-1,data=Computer_Data)
cd_y <- Computer_Data$price
```

```{r}
library(glmnet)
```

```{r}
# Lambda is the hyperparameter to tune the ridge regression

# glmnet automatically selects the range of λ values

# setting lamda as 10^10 till 10^-2
lambda <- 10^seq(10, -2, length = 50)
```

```{r}
# Note: glmnet() function standardizes the variables to get them on to same scale by default. 

ridge_reg <- glmnet(cd_x,cd_y,alpha=0,lambda=lambda)
summary(ridge_reg)
```

```{r}
# Below graph shows how the coefficients vary with change in lambda
# With increase in lambda the coefficients value converges to 0 
plot(ridge_reg,xvar="lambda",label=T)
```

```{r}
# ridge regression coefficients, stored in a matrix 
dim(coef(ridge_reg))
plot(ridge_reg)
```

```{r}
#Display 1st lambda value
ridge_reg$lambda[1] 
# Display coefficients associated with 50th lambda value
coef(ridge_reg)[,1] 
# Calculate L2 norm
sqrt(sum(coef(ridge_reg)[-1,1]^2)) 
```

```{r}
ridge_reg$lambda[50]
coef(ridge_reg)[,50] 
sqrt(sum(coef(ridge_reg)[-1,41]^2)) # Calculate L2 norm
```


Here we can observe that smaller L2 norm for smaller values of lambda.

### Splitting the dataset

```{r}
n=nrow(Computer_Data)
n1=n*0.7
n2=n-n1
index=sample(1:n,n1)
train=Computer_Data[index,]
head(train)
```

```{r}
test=Computer_Data[-index,]
head(test)
```

```{r}
x_train <- model.matrix(price~.-1,data=train)
y_train <- train$price
```

```{r}
head(x_train)
```

```{r}
x_test <- model.matrix(price~.-1,data=test)
y_test <- test$price
```


### Ridge Regression Model
```{r}
ridge_mod = glmnet(x_train, y_train, alpha=0, lambda = lambda)
plot(ridge_mod)
```

```{r}
#Predictions
ridge_pred = predict(ridge_mod, s = -2, newx = x_test)
mean((ridge_pred - y_test)^2)
```

```{r}
# Fit ridge regression model on training data
cv.out = cv.glmnet(x_train, y_train, alpha = 0) 
```

```{r}
# Select lamda that minimizes training MSE
bestlam = cv.out$lambda.min  
bestlam
```

```{r}
# Draw plot of training MSE as a function of lambda
plot(cv.out) 
```

```{r}
# predicting on test data with best lambda
ridge_pred1 = predict(ridge_mod, s = bestlam, newx = x_test)
mean((ridge_pred1 - y_test)^2) # Calculate test MSE
```

### Lasso Regression Model

```{r}
# Fit lasso model on training data
lasso_mod = glmnet(x_train,y_train, alpha = 1, lambda = lambda)
```

```{r}
plot(lasso_mod)    # Draw plot of coefficients
```

```{r}
cv.out = cv.glmnet(x_train, y_train, alpha = 1) # Fit lasso model on training data
```

```{r}
plot(cv.out) # Draw plot of training MSE as a function of lambda
```

```{r}
bestlam_lasso = cv.out$lambda.min # Select lamda that minimizes training MSE
bestlam_lasso
```

```{r}
# Use best lambda to predict test data
lasso_pred = predict(lasso_mod, s = bestlam_lasso, newx = x_test)
```

```{r}
mean((lasso_pred - y_test)^2) # Calculate test MSE
```

```{r}
# Fit lasso model on full dataset
out = glmnet(cd_x, cd_y, alpha = 1, lambda = lambda)
```

```{r}
# Display coefficients using lambda chosen by CV
lasso_coef = predict(out, type = "coefficients", s = bestlam)[1:10,] 
lasso_coef
```

### Conclusion

We can conclude that MSE is less for Lasso regression compared to Linear and Ridge for the given problem statement.

