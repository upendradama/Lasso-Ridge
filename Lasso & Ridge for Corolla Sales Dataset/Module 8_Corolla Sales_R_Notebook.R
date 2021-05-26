# Lasso & Ridge Regression

### Problem Statement:-

  - Build a model to predict sales of Toyota Corolla

```{r}
# Loading the datset.
library(readr)
ToyotaCorolla <- read_csv("/Users/thanush/Desktop/Digi 360/Module 8/Datasets-7/ToyotaCorolla.csv")
head(ToyotaCorolla)
```
```{r}
# Extracting selected columns given in probelm statement
ToyotaCorolla <- ToyotaCorolla[,c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
head(ToyotaCorolla)
```

```{r}
#Renaming the column names
colnames(ToyotaCorolla) <- c("price","age","km","hp","cc","door","gear","qrt","wgt")
head(ToyotaCorolla)
```
```{r}
#Lets' see quick summary

summary(ToyotaCorolla)
```

```{r}
# Attching the dataframe
attach(ToyotaCorolla)
```

```{r}
#Predicting the profit of the startups
pf1 <- mean(ToyotaCorolla$price)
pf1
```

```{r}
# Error in Prediction
# AV - PV
err1 <- ToyotaCorolla$price - mean(ToyotaCorolla$price)
```

```{r}
# RMSE
MSE1 <- mean(err1^2)
MSE1
```


### Building Linear Regression Model

```{r}
m1 <- lm(price ~ age + km + hp + cc + door + gear + qrt + wgt)
summary(m1)
```

```{r}
#Predictions
predpf2 <- predict(m1, data=ToyotaCorolla)
MSE2 <- mean(m1$residuals)^2
```

```{r}
# Residual plot
plot(ToyotaCorolla$price, predpf2)
```


```{r}
barplot(sort(m1$coefficients), ylim=c(-0.5, 5))
```

### Regularization Methods

```{r}
# Converting the data into compatible format in which model accepts 
tc_x <- model.matrix(price~.-1,data=ToyotaCorolla)
tc_y <- ToyotaCorolla$price
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

ridge_reg <- glmnet(tc_x,tc_y,alpha=0,lambda=lambda)
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
n=nrow(ToyotaCorolla)
n1=n*0.7
n2=n-n1
index=sample(1:n,n1)
train=ToyotaCorolla[index,]
head(train)
```

```{r}
test=ToyotaCorolla[-index,]
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
out = glmnet(tc_x, tc_y, alpha = 1, lambda = lambda) 
```

```{r}
# Display coefficients using lambda chosen by CV
lasso_coef = predict(out, type = "coefficients", s = bestlam)[1:8,] 
lasso_coef
```

### Conclusion

    - MSE for Ridge Regression is 2219718
    - MSE for Lasso Regression is 1985819
    - MSE for Linear regression is 1855379
    
We can conclude that MSE is less for linear regression compared to Lasso and Ridge for the given problem statement.