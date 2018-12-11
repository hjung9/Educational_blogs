post3
================
Hwayoung Jung
December 2, 2018

1. Purpose
----------

In this post, I will show how to build a model with variable selection methods using stepwise selection. I will generate models to predict important factors associated with house price. Variables are shown below:

-   id: an identical number labeled for each house

-   date: Date of the house was sold

-   price: The house price (Dependent variable)

-   bedrooms: The number of bedroons in the house

-   bathrooms: The number of bathrooms divided by the number of bedrooms

-   sqft\_living: area of the house in square ft.

-   sqft\_lot: area of the house lot in square ft.

-   floors: The number of total floors in the house.

-   waterfront: Whether the house is built near waterfront for view

-   view: Whether or not it has been viewed by other people

-   condition: Overall condition of the house

-   grade: overall grade of the house using "King County grading system"

-   sqft\_above: Area of the house from basement in square ft.

-   sqft\_basement: Area of the basement in square ft.

-   yr\_built: Years of the house was built

-   yr\_renovated: Year of the house renovation

-   zipcode: zipcode of the address of the house

-   lat: Latitude of the house

-   long: longitude of the house

-   sqft\_living15: Area of livingroom in 2015

-   sqft\_lot15: Area of lot in 2015

The dataset is obtained from kaggle. (<https://www.kaggle.com/harlfoxem/housesalesprediction>).

2. Overview of analysis techniques
----------------------------------

2.1 Model selection criteria
============================

![alt text here](https://github.com/hjung9/STAT576_1/blob/master/post3_1.png)

This is a typical multiple regression form, whre X1~Xp are variables and Beta1~Betap are coefficients of each variable, beta0 is an intercept, e is an error term. It might be tempting to include as many variables as possible to consider many factors that would explain the dependent variable. Indeed, R^2 will increase as you include more variables in a model. But will all the variables actually contribute in predicting the dependent variable? The problem of R^2 is that it blindly increases regardless actual contribution of variables. That means, even if you include any irrelevant variable, R^2 still increases. There are several model selection criteria to handle this issue: AIC, BIC, Cp and adjusted R^2. Among these, AIC is most frequently used.

![alt text here](https://github.com/hjung9/STAT576_1/blob/master/post_3_2AICBIC.png)

Both AIC and BIC have the similar structure- combining measurement of prediction accuracy and penalty for increasing complexity of the model, which increases as the number of variable increases. n is the number of variable and m represents the sample size. Unlike AIC that just subtract n from the prediction accuracy measuremnt, BIC considers both the number of variables and the sample size, usually resulting in higher penatly for including more variables. The smaller the criteria, the better.

Using the model selection criteria shown above, we can conduct stepwise model selection. There are mainly 3 ways: forward, backward and mixed selection. The method is called 'stepwise' because it adds/removes a single variable at a time.

2.2 Forward stepwise model
==========================

-   Forward stepwise model starts fitting a null model that does not include any variables first, then add one variable at a time until all variables are included.

2.3 Backward stepwise model
===========================

-   Backward stepwisse model starts from a full model that includes all variables, then remove one least significantly contributing variable at a time until all variables are included.

2.4 Mixed stepwise model
========================

-   Mixed selection is a combination of forward and backward selection. The constraints of forward and backward selection is that, once variables are included, the change cannot be reversed. For example, in backward selection, if a variable is removed, then the removed variable cannot be included in the model later. In forward selection, if a variable is added, it cannot be removed. But the bidirectional selection flexibly remove a variable that was previously added and add a variable that was previously removed if the previous change does not improve the fitness of the model which is checked by model criteria value (AIC, BIC ..etc).

3. Pros cons of the model
-------------------------

Advantages: - Computationally efficient and fast - Easy to interprete with smaller number of variables included in the final model

Disadvantages: - P-values do not provide meaningful information because of the bias - Not efficient in dealing with variables with high collinearity. It will just randomly remove one variable among highly correlated variables.

Even though I excluded some variables, there are still 13 variables so it needs to go further selection process to make more simpler, accurate model. In the following analysis, I will use forward, backward and bidirectinoal selection model with different model selection criteria.

4. Data analysis
----------------

4.1 Pre-processing
==================

``` r
dat=read.csv("kc_house_data.csv")
dim(dat)
```

    ## [1] 21613    21

The original dataset includes 21613 rows and 21 columns. To make the prediction more meaningful, I have selected some rows and columns to include for data analysis. For rows, I only chose rows of which the zipcode is 98006. This dataset only contains variables about features of house itself (such as \#of room, area of specific place in the house etc..). Since house price is also subject to change according to external traits such as population of a city and development of public transportation and so on. For example, house price cannot be compared between san francisco in CA and Knoxville in TN. Hence, to remove the effect of location, I randomly selected just one zipcode, 98006.

``` r
dat = subset(dat, zipcode=="98006")
head(dat)
```

    ##             id            date   price bedrooms bathrooms sqft_living
    ## 142 1777500060 20140708T000000  527700        5      2.50        2820
    ## 145 6071600370 20150227T000000  500000        4      2.25        2030
    ## 154 7855801670 20150401T000000 2250000        4      3.25        5180
    ## 168 1836980160 20150324T000000  807100        4      2.50        2680
    ## 175 1687900520 20140929T000000  673000        4      2.25        2590
    ## 289 9552700140 20140702T000000  675000        5      2.25        2900
    ##     sqft_lot floors waterfront view condition grade sqft_above
    ## 142     9375      1          0    0         4     8       1550
    ## 145     8517      1          0    0         4     8       1380
    ## 154    19850      2          0    3         3    12       3540
    ## 168     4499      2          0    0         3     9       2680
    ## 175     8190      2          0    0         4     8       2590
    ## 289    10300      1          0    0         3     8       1450
    ##     sqft_basement yr_built yr_renovated zipcode     lat     long
    ## 142          1270     1968            0   98006 47.5707 -122.128
    ## 145           650     1961            0   98006 47.5495 -122.174
    ## 154          1640     2006            0   98006 47.5620 -122.162
    ## 168             0     1999            0   98006 47.5650 -122.125
    ## 175             0     1980            0   98006 47.5619 -122.125
    ## 289          1450     1985            0   98006 47.5461 -122.151
    ##     sqft_living15 sqft_lot15
    ## 142          2820       9375
    ## 145          2230       8824
    ## 154          3160       9750
    ## 168          2920       4500
    ## 175          2260       8335
    ## 289          2310      10300

For columns, I excluded unnecessary, irrelevant variables with house price, such as id, date, zipcode, latitude and longitude. Also excluded the yr\_renovated because it contained too many 0s and didn't seem to be meaningful to include it.

``` r
#excluding unnecessary, irrelevant variables with house price
dat=dat[,-c(1,2,16,17,18,19)]
```

I will covert variable waterfront, view and condition because they are categorical variables. For remaining independent variables, I will standardize those because they are in different units.

``` r
#converting data type as factor for categorical variables
dat=transform(dat, waterfront=as.factor(waterfront))
dat=transform(dat, view=as.factor(view))
dat=transform(dat, condition=as.factor(condition))
```

The scale of price is quite large. The minimum is 75000 and the maximum is 7700000. I will log-transform the dependent variable, rather than scaling it. For log-transformation, generally We need to be cautious because it would not work for the values equal to 0. In this case, we add 1 as a convention. log(dat+1).

But in this data, the price is all above 0, so I just used log(dat)

``` r
sdat=as.data.frame(cbind(log(dat[,1]),scale(dat[,-c(1,7,8,9)],center=T,scale=T),dat[,c(7,8,9)]))#standardizing excluding categorical variable and dependent variable
#renaming the first column
colnames(sdat)[1] = "price"
```

Then I will split the data into half training and half test data.

``` r
#splitting into training and test data
x=sdat[,2:15]
y=sdat[,1]
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
```

4.2 Forward selection model
===========================

Now we are ready to run forward, backward and mixed stepwise selection models. Let's get started with forward selection model.

``` r
#1. forward selection
library(MASS)
fit = lm(price~., data=sdat[train,])
#explain stepAIC function.
#Model selection criteria
```

Using stepAIC function automatically adds variables step by step. To see the results, use object\_name\_of\_model$anova

``` r
fit_fw = stepAIC(fit, direction="forward") # stepwise selection
```

    ## Start:  AIC=-912.57
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition

``` r
#results
fit_fw$anova
```

    ## Stepwise Model Path 
    ## Analysis of Deviance Table
    ## 
    ## Initial Model:
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## Final Model:
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## 
    ##   Step Df Deviance Resid. Df Resid. Dev      AIC
    ## 1                        230   5.473394 -912.571

This shows the final model of forward selection, but here is a problem. The final model contains all variables in the initial model and no variables were selected at all.

4.3 Backward selection model
============================

Then, what about backward selection method? Let's try.

``` r
fit_bw = stepAIC(fit, direction="backward") # stepwise selection
```

    ## Start:  AIC=-912.57
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## 
    ## Step:  AIC=-912.57
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bedrooms       1   0.00310 5.4765 -914.43
    ## - sqft_lot15     1   0.02059 5.4940 -913.64
    ## - yr_built       1   0.03386 5.5073 -913.04
    ## <none>                       5.4734 -912.57
    ## - sqft_lot       1   0.04732 5.5207 -912.43
    ## - bathrooms      1   0.07056 5.5440 -911.38
    ## - condition      3   0.17161 5.6450 -910.88
    ## - sqft_living15  1   0.20172 5.6751 -905.56
    ## - sqft_living    1   0.20527 5.6787 -905.40
    ## - floors         1   0.23574 5.7091 -904.07
    ## - waterfront     1   0.24888 5.7223 -903.50
    ## - grade          1   0.43319 5.9066 -895.61
    ## - sqft_above     1   0.57595 6.0493 -889.66
    ## - view           4   1.06580 6.5392 -876.27
    ## 
    ## Step:  AIC=-914.43
    ## price ~ bathrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + yr_built + sqft_living15 + sqft_lot15 + waterfront + 
    ##     view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot15     1   0.02069 5.4972 -915.49
    ## - yr_built       1   0.03717 5.5137 -914.75
    ## <none>                       5.4765 -914.43
    ## - sqft_lot       1   0.04456 5.5211 -914.41
    ## - bathrooms      1   0.06747 5.5440 -913.38
    ## - condition      3   0.17757 5.6541 -912.48
    ## - sqft_living15  1   0.19864 5.6751 -907.56
    ## - sqft_living    1   0.20537 5.6819 -907.26
    ## - floors         1   0.23818 5.7147 -905.83
    ## - waterfront     1   0.25279 5.7293 -905.19
    ## - grade          1   0.45473 5.9312 -896.57
    ## - sqft_above     1   0.59292 6.0694 -890.83
    ## - view           4   1.08849 6.5650 -877.29
    ## 
    ## Step:  AIC=-915.49
    ## price ~ bathrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + yr_built + sqft_living15 + waterfront + view + 
    ##     condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot       1   0.02436 5.5215 -916.39
    ## - yr_built       1   0.02870 5.5259 -916.19
    ## <none>                       5.4972 -915.49
    ## - bathrooms      1   0.07634 5.5735 -914.06
    ## - condition      3   0.19174 5.6889 -912.95
    ## - sqft_living    1   0.19934 5.6965 -908.62
    ## - floors         1   0.22978 5.7270 -907.29
    ## - sqft_living15  1   0.23471 5.7319 -907.08
    ## - waterfront     1   0.26375 5.7609 -905.82
    ## - grade          1   0.49811 5.9953 -895.89
    ## - sqft_above     1   0.59607 6.0933 -891.86
    ## - view           4   1.06812 6.5653 -879.28
    ## 
    ## Step:  AIC=-916.39
    ## price ~ bathrooms + sqft_living + floors + grade + sqft_above + 
    ##     yr_built + sqft_living15 + waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## <none>                       5.5215 -916.39
    ## - yr_built       1   0.06226 5.5838 -915.60
    ## - bathrooms      1   0.08051 5.6021 -914.79
    ## - condition      3   0.21012 5.7317 -913.09
    ## - sqft_living    1   0.19110 5.7126 -909.92
    ## - sqft_living15  1   0.21043 5.7320 -909.08
    ## - floors         1   0.23463 5.7562 -908.03
    ## - waterfront     1   0.26848 5.7900 -906.57
    ## - grade          1   0.48835 6.0099 -897.29
    ## - sqft_above     1   0.59384 6.1154 -892.95
    ## - view           4   1.15410 6.6756 -877.13

``` r
fit_bw$anova
```

    ## Stepwise Model Path 
    ## Analysis of Deviance Table
    ## 
    ## Initial Model:
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## Final Model:
    ## price ~ bathrooms + sqft_living + floors + grade + sqft_above + 
    ##     yr_built + sqft_living15 + waterfront + view + condition
    ## 
    ## 
    ##              Step Df    Deviance Resid. Df Resid. Dev       AIC
    ## 1                                      230   5.473394 -912.5710
    ## 2 - sqft_basement  0 0.000000000       230   5.473394 -912.5710
    ## 3      - bedrooms  1 0.003103178       231   5.476497 -914.4298
    ## 4    - sqft_lot15  1 0.020692162       232   5.497189 -915.4908
    ## 5      - sqft_lot  1 0.024356332       233   5.521546 -916.3900

Final models included fewer variables than in initial model. Let's proceed on with the final model

``` r
fit_bw_f=lm(price~sqft_living + sqft_lot + grade + sqft_above + sqft_living15 +sqft_lot15 + waterfront + view + condition,data=sdat[train,])
summary(fit_bw_f)
```

    ## 
    ## Call:
    ## lm(formula = price ~ sqft_living + sqft_lot + grade + sqft_above + 
    ##     sqft_living15 + sqft_lot15 + waterfront + view + condition, 
    ##     data = sdat[train, ])
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.54619 -0.07583  0.00133  0.08428  0.79378 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   13.39641    0.15797  84.805  < 2e-16 ***
    ## sqft_living    0.13808    0.02311   5.974 8.54e-09 ***
    ## sqft_lot      -0.02175    0.01360  -1.599 0.111131    
    ## grade          0.10181    0.02085   4.883 1.93e-06 ***
    ## sqft_above     0.08260    0.02163   3.819 0.000172 ***
    ## sqft_living15  0.06943    0.02164   3.209 0.001519 ** 
    ## sqft_lot15     0.01225    0.01434   0.854 0.394059    
    ## waterfront1    0.37128    0.12472   2.977 0.003217 ** 
    ## view1          0.07328    0.06656   1.101 0.271999    
    ## view2          0.10597    0.03677   2.882 0.004319 ** 
    ## view3          0.11682    0.04494   2.600 0.009927 ** 
    ## view4          0.35815    0.05893   6.077 4.92e-09 ***
    ## condition3     0.09609    0.15941   0.603 0.547237    
    ## condition4     0.14095    0.15872   0.888 0.375439    
    ## condition5     0.20289    0.16079   1.262 0.208276    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1574 on 234 degrees of freedom
    ## Multiple R-squared:  0.8598, Adjusted R-squared:  0.8514 
    ## F-statistic: 102.5 on 14 and 234 DF,  p-value: < 2.2e-16

Notice the Final reduced model of backward selection still contains so many variables.

4.4 Mixed selection model
=========================

Let's see if the variable selection option, "both" with BIC criteria produce more parsimonious final models with fewer variables. To select BIC as a model selection criteria, set k=log(the number of rows in data) in the stepAIC function. (By default, the k=2, which is meant for AIC criteria)

``` r
fit_fb = stepAIC(fit, direction="both",k = log(nrow(sdat[train,])))
```

    ## Start:  AIC=-845.74
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## 
    ## Step:  AIC=-845.74
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - condition      3   0.17161 5.6450 -854.60
    ## - bedrooms       1   0.00310 5.4765 -851.12
    ## - sqft_lot15     1   0.02059 5.4940 -850.32
    ## - yr_built       1   0.03386 5.5073 -849.72
    ## - sqft_lot       1   0.04732 5.5207 -849.11
    ## - bathrooms      1   0.07056 5.5440 -848.07
    ## <none>                       5.4734 -845.74
    ## - sqft_living15  1   0.20172 5.6751 -842.25
    ## - sqft_living    1   0.20527 5.6787 -842.09
    ## - floors         1   0.23574 5.7091 -840.76
    ## - waterfront     1   0.24888 5.7223 -840.18
    ## - grade          1   0.43319 5.9066 -832.29
    ## - sqft_above     1   0.57595 6.0493 -826.34
    ## - view           4   1.06580 6.5392 -823.51
    ## 
    ## Step:  AIC=-854.6
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bedrooms       1   0.00906 5.6541 -859.72
    ## - yr_built       1   0.00982 5.6548 -859.69
    ## - sqft_lot15     1   0.03424 5.6792 -858.62
    ## - bathrooms      1   0.05398 5.6990 -857.75
    ## - sqft_lot       1   0.08326 5.7283 -856.48
    ## <none>                       5.6450 -854.60
    ## - sqft_living15  1   0.19122 5.8362 -851.83
    ## - waterfront     1   0.20370 5.8487 -851.30
    ## - sqft_living    1   0.27183 5.9168 -848.41
    ## - floors         1   0.28932 5.9343 -847.68
    ## + condition      3   0.17161 5.4734 -845.74
    ## - grade          1   0.41396 6.0590 -842.50
    ## - sqft_above     1   0.53837 6.1834 -837.44
    ## - view           4   1.20745 6.8525 -828.41
    ## 
    ## Step:  AIC=-859.72
    ## price ~ bathrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + yr_built + sqft_living15 + sqft_lot15 + waterfront + 
    ##     view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - yr_built       1   0.01215 5.6662 -864.71
    ## - sqft_lot15     1   0.03486 5.6889 -863.71
    ## - bathrooms      1   0.04668 5.7007 -863.19
    ## - sqft_lot       1   0.07674 5.7308 -861.88
    ## <none>                       5.6541 -859.72
    ## - sqft_living15  1   0.18339 5.8375 -857.29
    ## - waterfront     1   0.20781 5.8619 -856.25
    ## + bedrooms       1   0.00906 5.6450 -854.60
    ## - sqft_living    1   0.26347 5.9175 -853.90
    ## - floors         1   0.29564 5.9497 -852.55
    ## + condition      3   0.17757 5.4765 -851.12
    ## - grade          1   0.44172 6.0958 -846.51
    ## - sqft_above     1   0.55925 6.2133 -841.75
    ## - view           4   1.24840 6.9025 -832.12
    ## 
    ## Step:  AIC=-864.71
    ## price ~ bathrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + sqft_living15 + sqft_lot15 + waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot15     1   0.02738 5.6936 -869.02
    ## - bathrooms      1   0.05799 5.7242 -867.69
    ## - sqft_lot       1   0.08883 5.7550 -866.35
    ## <none>                       5.6662 -864.71
    ## - waterfront     1   0.20421 5.8704 -861.41
    ## + yr_built       1   0.01215 5.6541 -859.72
    ## + bedrooms       1   0.01139 5.6548 -859.69
    ## - sqft_living15  1   0.25650 5.9227 -859.20
    ## - sqft_living    1   0.26628 5.9325 -858.79
    ## - floors         1   0.30236 5.9686 -857.28
    ## + condition      3   0.15255 5.5137 -854.95
    ## - grade          1   0.50750 6.1737 -848.86
    ## - sqft_above     1   0.54711 6.2133 -847.27
    ## - view           4   1.25280 6.9190 -837.04
    ## 
    ## Step:  AIC=-869.02
    ## price ~ bathrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + sqft_living15 + waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bathrooms      1   0.06236 5.7560 -871.83
    ## - sqft_lot       1   0.06287 5.7565 -871.81
    ## <none>                       5.6936 -869.02
    ## - waterfront     1   0.21590 5.9095 -865.27
    ## + sqft_lot15     1   0.02738 5.6662 -864.71
    ## + bedrooms       1   0.01108 5.6825 -863.99
    ## + yr_built       1   0.00467 5.6889 -863.71
    ## - sqft_living    1   0.26045 5.9541 -863.40
    ## - sqft_living15  1   0.28212 5.9757 -862.50
    ## - floors         1   0.31316 6.0068 -861.21
    ## + condition      3   0.16771 5.5259 -859.92
    ## - grade          1   0.53546 6.2291 -852.16
    ## - sqft_above     1   0.55733 6.2509 -851.29
    ## - view           4   1.23356 6.9272 -842.26
    ## 
    ## Step:  AIC=-871.83
    ## price ~ sqft_living + sqft_lot + floors + grade + sqft_above + 
    ##     sqft_living15 + waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot       1   0.07788 5.8338 -874.00
    ## <none>                       5.7560 -871.83
    ## + bathrooms      1   0.06236 5.6936 -869.02
    ## - waterfront     1   0.20869 5.9646 -868.48
    ## + sqft_lot15     1   0.03175 5.7242 -867.69
    ## + yr_built       1   0.01205 5.7439 -866.83
    ## + bedrooms       1   0.00233 5.7536 -866.41
    ## - sqft_living15  1   0.27903 6.0350 -865.56
    ## - floors         1   0.30265 6.0586 -864.59
    ## + condition      3   0.13878 5.6172 -861.35
    ## - sqft_living    1   0.48225 6.2382 -857.31
    ## - sqft_above     1   0.56218 6.3181 -854.14
    ## - grade          1   0.62107 6.3770 -851.83
    ## - view           4   1.22225 6.9782 -845.95
    ## 
    ## Step:  AIC=-874
    ## price ~ sqft_living + floors + grade + sqft_above + sqft_living15 + 
    ##     waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## <none>                       5.8338 -874.00
    ## + sqft_lot       1   0.07788 5.7560 -871.83
    ## + bathrooms      1   0.07736 5.7565 -871.81
    ## - waterfront     1   0.20579 6.0396 -870.88
    ## + yr_built       1   0.04379 5.7900 -870.36
    ## - sqft_living15  1   0.22378 6.0576 -870.14
    ## + sqft_lot15     1   0.00214 5.8317 -868.57
    ## + bedrooms       1   0.00006 5.8338 -868.48
    ## - floors         1   0.26695 6.1008 -868.38
    ## + condition      3   0.14110 5.6927 -863.54
    ## - sqft_living    1   0.48669 6.3205 -859.56
    ## - sqft_above     1   0.53097 6.3648 -857.83
    ## - grade          1   0.64416 6.4780 -853.44
    ## - view           4   1.31705 7.1509 -845.38

``` r
fit_fb$anova
```

    ## Stepwise Model Path 
    ## Analysis of Deviance Table
    ## 
    ## Initial Model:
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## Final Model:
    ## price ~ sqft_living + floors + grade + sqft_above + sqft_living15 + 
    ##     waterfront + view
    ## 
    ## 
    ##              Step Df    Deviance Resid. Df Resid. Dev       AIC
    ## 1                                      230   5.473394 -845.7393
    ## 2 - sqft_basement  0 0.000000000       230   5.473394 -845.7393
    ## 3     - condition  3 0.171611004       233   5.645005 -854.6045
    ## 4      - bedrooms  1 0.009058165       234   5.654063 -859.7228
    ## 5      - yr_built  1 0.012149860       235   5.666213 -864.7057
    ## 6    - sqft_lot15  1 0.027383996       236   5.693597 -869.0227
    ## 7     - bathrooms  1 0.062357169       237   5.755954 -871.8279
    ## 8      - sqft_lot  1 0.077875526       238   5.833830 -873.9990

``` r
fit_fb_f=lm(price~sqft_living + grade + sqft_living15 + waterfront + condition,data=sdat[train,])
summary(fit_fb_f)
```

    ## 
    ## Call:
    ## lm(formula = price ~ sqft_living + grade + sqft_living15 + waterfront + 
    ##     condition, data = sdat[train, ])
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.61870 -0.08764 -0.00719  0.09418  0.78004 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   13.39141    0.17158  78.050  < 2e-16 ***
    ## sqft_living    0.18747    0.02050   9.144  < 2e-16 ***
    ## grade          0.12913    0.02127   6.072 4.88e-09 ***
    ## sqft_living15  0.09716    0.02139   4.543 8.79e-06 ***
    ## waterfront1    0.67066    0.12196   5.499 9.72e-08 ***
    ## condition3     0.13355    0.17272   0.773    0.440    
    ## condition4     0.17722    0.17228   1.029    0.305    
    ## condition5     0.26936    0.17447   1.544    0.124    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1713 on 241 degrees of freedom
    ## Multiple R-squared:  0.8291, Adjusted R-squared:  0.8242 
    ## F-statistic: 167.1 on 7 and 241 DF,  p-value: < 2.2e-16

``` r
coef(fit_fb)
```

    ##   (Intercept)   sqft_living        floors         grade    sqft_above 
    ##   13.52194527    0.11166826   -0.05351823    0.10415704    0.12689570 
    ## sqft_living15   waterfront1         view1         view2         view3 
    ##    0.06194474    0.35595422    0.05748886    0.12009638    0.12595412 
    ##         view4 
    ##    0.39084805

Notice this method effectively selected variables. Only 5 variables in this model. In the backward selection model, sqft\_living, sqft\_lot, grade, sqft\_above, sqft\_living15 , sqft\_lot15, waterfront, view and condition were selected. In the mixed model, sqft\_living, grade, sqft\_living15, waterfront and condition were selected, which all overlap with the variables selected in the backward selection model. By using coef(), we can see the coefficients of each variable. In the final model with BIC, waterfront shows the heighest value, followed by view of level 4th.

``` r
train_mse1=NA  #1. Forward training MSE
train_mse2=NA  #2. Backward training MSE
train_mse3=NA  #3. Mixed training MSE

test_mse1=NA  #1. Forward test MSE
test_mse2=NA  #2. Backward test MSE
test_mse3=NA  #3. Mixed test MSE 

errmat=matrix(0,20,6)
for(i in 1:20){
# Splitting into training and test dataset
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
# forward selection

fit_fw = lm(price~.,data=sdat[train,])

train_pred=fit_fw$fitted.values
test_pred=predict(fit_fw, newx=sdat[test,])

train_mse1=mean((y[train]-train_pred)^2) 
test_mse1=mean((y[test]-test_pred)^2)

fw_err=c(train_mse1,test_mse1)

# backward selection

fit_fb=lm(price~sqft_living + grade + sqft_living15 + waterfront + condition,data=sdat[train,])

train_pred=fit_bw$fitted.values
test_pred=predict(fit_bw, newx=sdat[test,])

train_mse2=mean((y[train]-train_pred)^2) 
test_mse2=mean((y[test]-test_pred)^2)

bw_err=c(train_mse2,test_mse2)

# mixed selection
fit_fb = lm(price~sqft_living + grade + sqft_living15 + waterfront + condition,data=sdat[train,])

train_pred=fit_fb$fitted.values
test_pred=predict(fit_fb, newx=sdat[test,])

train_mse3=mean((y[train]-train_pred)^2) 
test_mse3=mean((y[test]-test_pred)^2)

fb_err=c(train_mse3,test_mse3)

errmat[i,] = c(fw_err, bw_err,fb_err)
}

colnames(errmat) <- c("Fw.Tr","Fw.Ts","Bw.Tr","Bw.Ts","Mx.Tr","Mx.Ts")
boxplot(errmat,ylab="MSE")
```

![](post3_files/figure-markdown_github/unnamed-chunk-12-1.png)

For validation of each model, I have calculated mean square error of training and test data of 3 models, respectively. Boxplot shows the summarized results.

Fw.Tr: Training MSE of forward selection model

Fw.Ts: Test MSE of forward selection model

Bw.Tr: Training MSE of backward selection model

Bw.Ts: Test MSE of backward selection model

Mx.Tr: Training MSE of mixed selection model

Mx.Ts: Test MSE of mixed selection model

It seems backward selection model is the most reliable in terms of generalization, whereas forward and mixed models show larger difference between training and test MSE.

The actual values of MSE for each models looks like below

``` r
errmat
```

    ##            Fw.Tr     Fw.Ts     Bw.Tr     Bw.Ts      Mx.Tr     Mx.Ts
    ##  [1,] 0.02967368 0.3443471 0.3114081 0.3655165 0.03410569 0.3418563
    ##  [2,] 0.02974576 0.3728105 0.3364375 0.3318165 0.03404014 0.3680633
    ##  [3,] 0.03308768 0.3837969 0.3236745 0.3190350 0.03728910 0.3769815
    ##  [4,] 0.03263549 0.3660928 0.3540897 0.3260429 0.03628618 0.3633817
    ##  [5,] 0.03333977 0.3540031 0.3225638 0.3531127 0.03619061 0.3502646
    ##  [6,] 0.02753561 0.3214729 0.2767823 0.3225045 0.03266280 0.3170640
    ##  [7,] 0.03128865 0.3575045 0.3359716 0.3449003 0.03525652 0.3539839
    ##  [8,] 0.03629768 0.3297841 0.3598251 0.3026018 0.04121066 0.3254586
    ##  [9,] 0.03095026 0.3688125 0.3212225 0.2816968 0.03411918 0.3595157
    ## [10,] 0.03190052 0.3183137 0.3568497 0.3366057 0.03645966 0.3101121
    ## [11,] 0.03721735 0.3344569 0.3216922 0.3318627 0.04031486 0.3256515
    ## [12,] 0.03028733 0.3808498 0.3190811 0.3358859 0.03388634 0.3721738
    ## [13,] 0.03535803 0.3238579 0.3017941 0.3516764 0.03902741 0.3146494
    ## [14,] 0.02664532 0.3285242 0.3080059 0.3516622 0.03093626 0.3234939
    ## [15,] 0.03851889 0.3436094 0.3465336 0.3243037 0.04177181 0.3429386
    ## [16,] 0.03185000 0.3709195 0.3245448 0.3019340 0.03764122 0.3685619
    ## [17,] 0.04064296 0.3417286 0.3228875 0.3369420 0.04279213 0.3419708
    ## [18,] 0.03329076 0.3391293 0.3714333 0.3461631 0.03593710 0.3304169
    ## [19,] 0.03664664 0.3196202 0.3031218 0.2951361 0.03975232 0.3103241
    ## [20,] 0.02741122 0.3578784 0.3519254 0.3319535 0.03088510 0.3560927

5. Conclusion
-------------

To conclude, the important factors deciding house price turned out to be whether the house has the view of water front and nice overall view around the house. Therefore, rather than traits of the internal house structure such as area of livingroom, whether the house has a view of water near the house.

reference of image of AIC/BIC: <https://slideplayer.com/slide/10911060/>
