# Lasso & Ridge Regression

### Problem Statement:- 

  - Build a model to predict the profit of 50 start ups
  
### Data Understanding
```{r}
library(readr)
X50_Startups <- read_csv("/Users/thanush/Desktop/Digi 360/Module 8/Datasets-7/50_Startups.csv")
head(X50_Startups)
```

```{r}
# Renaming the coulmn names
colnames(X50_Startups) <- c("rd","ad","mk","st","pf")
head(X50_Startups)
```
```{r}
# Attching the dataset
attach(X50_Startups)
```

### Data Preparation

```{r}
#State is not numeric so let’s assign dummy variables.
X50_Startups <- cbind(X50_Startups,ifelse(X50_Startups$st=="New York",1,0), ifelse(X50_Startups$st=="California",1,0),ifelse(X50_Startups$st=="Florida",1,0))

head(X50_Startups)
```
```{r}
#Rename the dummy variable columns
colnames(X50_Startups)[6] <- "ny" 
colnames(X50_Startups)[7] <- "cf"
colnames(X50_Startups)[8] <- "fl"

```

```{r}
#Dropping the state column since dummy values are already assigned
X50_Startups$st <- NULL
head(X50_Startups)
```

### Data Visualization

```{r}
#Plotting the scatter plot for all the columns
attach(X50_Startups)
plot(X50_Startups)
```

```{r}
#Finding the correlation coefficient for the entire dataset
cor(X50_Startups)
```
```{r}
#Predicting the profit of the startups
pf1 <- mean(X50_Startups$pf)
pf1
```

```{r}
# Error in Prediction
# AV - PV
err1 <- X50_Startups$pf - mean(X50_Startups$pf)
```

```{r}
# RMSE
MSE1 <- mean(err1^2)
MSE1
```

### Building Linear Regression Model

```{r}
m1 <- lm(pf ~ rd + ad + mk + ny + cf + fl)
summary(m1)
```

```{r}
#Predictions
predpf2 <- predict(m1, data=X50_Startups)
MSE2 <- mean(m1$residuals)^2
```

```{r}
# Residual plot
plot(X50_Startups$pf, predpf2)

barplot(sort(m1$coefficients), ylim=c(-0.5, 5))
```

### Regularization Methods

```{r}
# Converting the data into compatible format in which model accepts 
strtup_x <- model.matrix(pf~.-1,data=X50_Startups)
strtup_y <- X50_Startups$pf
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

ridge_reg <- glmnet(strtup_x,strtup_y,alpha=0,lambda=lambda)
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
n=nrow(X50_Startups)
n1=n*0.7
n2=n-n1
index=sample(1:n,n1)
train=X50_Startups[index,]
head(train)
```

```{r}
test=X50_Startups[-index,]
head(test)
```

```{r}
x_train <- model.matrix(pf~.-1,data=train)
y_train <- train$pf
```

```{r}
head(x_train)
```

```{r}
x_test <- model.matrix(pf~.-1,data=test)
y_test <- test$pf
```


```{r}
# Predict and MSE for Linear Regression
pred_ols = predict(m1, test)
mean((pred_ols - y_test)^2)
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
out = glmnet(strtup_x, strtup_y, alpha = 1, lambda = lambda) 
```

```{r}
# Display coefficients using lambda chosen by CV
lasso_coef = predict(out, type = "coefficients", s = bestlam)[1:7,] 
lasso_coef
```


### Conclusion

    - MSE for Ridge Regression is 84529679
    - MSE for Lasso Regression is 65992310
    - MSE for Linear regression is 52344939
    
We can conclude that MSE is less for linear regression compared to Lasso and Ridge for the given problem statement
  
  
