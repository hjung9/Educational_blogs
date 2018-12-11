post3
================
Hwayoung Jung
December 2, 2018

1. Purpose
----------

In this post, I will show how to build a model with variable selection methods using stepwise selection. I will generate models to predict important factors associated with house price. Variables are shown below: - id: an identical number labeled for each house - date: Date of the house was sold - price: The house price (Dependent variable) - bedrooms: The number of bedroons in the house - bathrooms: The number of bathrooms divided by the number of bedrooms - sqft\_living: area of the house in square ft. - sqft\_lot: area of the house lot in square ft. - floors: The number of total floors in the house. - waterfront: Whether the house is built near waterfront for view - view: Whether or not it has been viewed by other people - condition: Overall condition of the house - grade: overall grade of the house using "King County grading system" - sqft\_above: Area of the house from basement in square ft. - sqft\_basement: Area of the basement in square ft. - yr\_built: Years of the house was built - yr\_renovated: Year of the house renovation - zipcode: zipcode of the address of the house - lat: Latitude of the house - long: longitude of the house - sqft\_living15: Area of livingroom in 2015. - sqft\_lot15: Area of lot in 2015

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

    ## Start:  AIC=-821.02
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
    ## 1                        229   7.842403 -821.019

This shows the final model of forward selection, but here is a problem. The final model contains all variables in the initial model and no variables were selected at all.

4.3 Backward selection model
============================

Then, what about backward selection method? Let's try.

``` r
fit_bw = stepAIC(fit, direction="backward") # stepwise selection
```

    ## Start:  AIC=-821.02
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## 
    ## Step:  AIC=-821.02
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bathrooms      1   0.00247 7.8449 -822.94
    ## - floors         1   0.00286 7.8453 -822.93
    ## - yr_built       1   0.00434 7.8467 -822.88
    ## - bedrooms       1   0.03438 7.8768 -821.93
    ## - sqft_lot15     1   0.04318 7.8856 -821.65
    ## <none>                       7.8424 -821.02
    ## - sqft_lot       1   0.06548 7.9079 -820.95
    ## - waterfront     1   0.09829 7.9407 -819.92
    ## - sqft_above     1   0.20295 8.0454 -816.66
    ## - condition      4   0.46075 8.3031 -814.80
    ## - grade          1   0.44038 8.2828 -809.42
    ## - sqft_living15  1   0.47788 8.3203 -808.29
    ## - sqft_living    1   0.58872 8.4311 -805.00
    ## - view           4   1.03666 8.8791 -798.11
    ## 
    ## Step:  AIC=-822.94
    ## price ~ bedrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + yr_built + sqft_living15 + sqft_lot15 + waterfront + 
    ##     view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - floors         1   0.00236 7.8472 -824.87
    ## - yr_built       1   0.00666 7.8515 -824.73
    ## - bedrooms       1   0.03941 7.8843 -823.69
    ## - sqft_lot15     1   0.04142 7.8863 -823.63
    ## <none>                       7.8449 -822.94
    ## - sqft_lot       1   0.06364 7.9085 -822.93
    ## - waterfront     1   0.09582 7.9407 -821.92
    ## - sqft_above     1   0.20314 8.0480 -818.57
    ## - condition      4   0.45916 8.3040 -816.78
    ## - grade          1   0.44018 8.2851 -811.35
    ## - sqft_living15  1   0.47564 8.3205 -810.28
    ## - sqft_living    1   0.66600 8.5109 -804.65
    ## - view           4   1.03581 8.8807 -800.06
    ## 
    ## Step:  AIC=-824.87
    ## price ~ bedrooms + sqft_living + sqft_lot + grade + sqft_above + 
    ##     yr_built + sqft_living15 + sqft_lot15 + waterfront + view + 
    ##     condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - yr_built       1   0.01237 7.8596 -826.47
    ## - bedrooms       1   0.04154 7.8888 -825.55
    ## - sqft_lot15     1   0.04315 7.8904 -825.50
    ## - sqft_lot       1   0.06320 7.9104 -824.87
    ## <none>                       7.8472 -824.87
    ## - waterfront     1   0.09450 7.9417 -823.89
    ## - condition      4   0.46880 8.3160 -818.42
    ## - sqft_above     1   0.28508 8.1323 -817.98
    ## - grade          1   0.46338 8.3106 -812.58
    ## - sqft_living15  1   0.48015 8.3274 -812.08
    ## - sqft_living    1   0.77546 8.6227 -803.40
    ## - view           4   1.03348 8.8807 -802.06
    ## 
    ## Step:  AIC=-826.47
    ## price ~ bedrooms + sqft_living + sqft_lot + grade + sqft_above + 
    ##     sqft_living15 + sqft_lot15 + waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bedrooms       1   0.03754 7.8971 -827.29
    ## - sqft_lot       1   0.05307 7.9127 -826.80
    ## - sqft_lot15     1   0.05618 7.9158 -826.70
    ## <none>                       7.8596 -826.47
    ## - waterfront     1   0.12253 7.9821 -824.62
    ## - sqft_above     1   0.27364 8.1332 -819.95
    ## - condition      4   0.53651 8.3961 -818.03
    ## - grade          1   0.45665 8.3162 -814.41
    ## - sqft_living15  1   0.47663 8.3362 -813.81
    ## - sqft_living    1   0.76595 8.6255 -805.32
    ## - view           4   1.14552 9.0051 -800.60
    ## 
    ## Step:  AIC=-827.29
    ## price ~ sqft_living + sqft_lot + grade + sqft_above + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot       1   0.04135 7.9385 -827.99
    ## - sqft_lot15     1   0.05162 7.9488 -827.66
    ## <none>                       7.8971 -827.29
    ## - waterfront     1   0.12394 8.0211 -825.41
    ## - sqft_above     1   0.27600 8.1731 -820.73
    ## - condition      4   0.55250 8.4496 -818.45
    ## - sqft_living15  1   0.45657 8.3537 -815.29
    ## - grade          1   0.55203 8.4492 -812.46
    ## - sqft_living    1   0.74819 8.6453 -806.75
    ## - view           4   1.22797 9.1251 -799.30
    ## 
    ## Step:  AIC=-827.99
    ## price ~ sqft_living + grade + sqft_above + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot15     1   0.01904 7.9575 -829.39
    ## <none>                       7.9385 -827.99
    ## - waterfront     1   0.12064 8.0591 -826.23
    ## - sqft_above     1   0.27661 8.2151 -821.46
    ## - condition      4   0.58542 8.5239 -818.27
    ## - sqft_living15  1   0.42732 8.3658 -816.93
    ## - grade          1   0.61716 8.5557 -811.34
    ## - sqft_living    1   0.74338 8.6819 -807.70
    ## - view           4   1.22598 9.1645 -800.23
    ## 
    ## Step:  AIC=-829.39
    ## price ~ sqft_living + grade + sqft_above + sqft_living15 + waterfront + 
    ##     view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## <none>                       7.9575 -829.39
    ## - waterfront     1   0.12205 8.0796 -827.60
    ## - sqft_above     1   0.29089 8.2484 -822.45
    ## - condition      4   0.58993 8.5475 -819.58
    ## - sqft_living15  1   0.49924 8.4568 -816.24
    ## - grade          1   0.63794 8.5955 -812.19
    ## - sqft_living    1   0.73602 8.6936 -809.36
    ## - view           4   1.21104 9.1686 -802.12

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
    ## price ~ sqft_living + grade + sqft_above + sqft_living15 + waterfront + 
    ##     view + condition
    ## 
    ## 
    ##              Step Df    Deviance Resid. Df Resid. Dev       AIC
    ## 1                                      229   7.842403 -821.0190
    ## 2 - sqft_basement  0 0.000000000       229   7.842403 -821.0190
    ## 3     - bathrooms  1 0.002470626       230   7.844874 -822.9405
    ## 4        - floors  1 0.002357162       231   7.847231 -824.8657
    ## 5      - yr_built  1 0.012370680       232   7.859602 -826.4735
    ## 6      - bedrooms  1 0.037541938       233   7.897144 -827.2870
    ## 7      - sqft_lot  1 0.041349213       234   7.938493 -827.9866
    ## 8    - sqft_lot15  1 0.019043017       235   7.957536 -829.3900

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
    ## -0.46183 -0.10340  0.00079  0.09760  0.83588 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   13.33846    0.18606  71.687  < 2e-16 ***
    ## sqft_living    0.12919    0.02750   4.698 4.48e-06 ***
    ## sqft_lot      -0.01767    0.01600  -1.105  0.27050    
    ## grade          0.10271    0.02545   4.036 7.38e-05 ***
    ## sqft_above     0.07209    0.02526   2.854  0.00471 ** 
    ## sqft_living15  0.09607    0.02617   3.670  0.00030 ***
    ## sqft_lot15     0.02237    0.01813   1.234  0.21841    
    ## waterfront1    0.37191    0.19448   1.912  0.05707 .  
    ## view1          0.07928    0.07708   1.029  0.30475    
    ## view2          0.09495    0.04452   2.133  0.03398 *  
    ## view3          0.17659    0.05360   3.295  0.00114 ** 
    ## view4          0.35742    0.06402   5.583 6.56e-08 ***
    ## condition2     0.07607    0.26195   0.290  0.77177    
    ## condition3     0.14192    0.18734   0.758  0.44951    
    ## condition4     0.21312    0.18586   1.147  0.25270    
    ## condition5     0.28520    0.18753   1.521  0.12966    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1841 on 233 degrees of freedom
    ## Multiple R-squared:  0.8335, Adjusted R-squared:  0.8228 
    ## F-statistic: 77.75 on 15 and 233 DF,  p-value: < 2.2e-16

Notice the Final reduced model of backward selection still contains so many variables.

4.4 Mixed selection model
=========================

Let's see if the variable selection option, "both" with BIC criteria produce more parsimonious final models with fewer variables. To select BIC as a model selection criteria, set k=log(the number of rows in data) in the stepAIC function. (By default, the k=2, which is meant for AIC criteria)

``` r
fit_fb = stepAIC(fit, direction="both",k = log(nrow(sdat[train,])))
```

    ## Start:  AIC=-750.67
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## 
    ## Step:  AIC=-750.67
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - condition      4   0.46075 8.3031 -758.52
    ## - bathrooms      1   0.00247 7.8449 -756.11
    ## - floors         1   0.00286 7.8453 -756.10
    ## - yr_built       1   0.00434 7.8467 -756.05
    ## - bedrooms       1   0.03438 7.8768 -755.10
    ## - sqft_lot15     1   0.04318 7.8856 -754.82
    ## - sqft_lot       1   0.06548 7.9079 -754.12
    ## - waterfront     1   0.09829 7.9407 -753.09
    ## <none>                       7.8424 -750.67
    ## - sqft_above     1   0.20295 8.0454 -749.83
    ## - grade          1   0.44038 8.2828 -742.58
    ## - view           4   1.03666 8.8791 -741.83
    ## - sqft_living15  1   0.47788 8.3203 -741.46
    ## - sqft_living    1   0.58872 8.4311 -738.16
    ## 
    ## Step:  AIC=-758.52
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bathrooms      1   0.00089 8.3040 -764.02
    ## - floors         1   0.01257 8.3157 -763.67
    ## - yr_built       1   0.03599 8.3391 -762.96
    ## - sqft_lot15     1   0.04282 8.3460 -762.76
    ## - bedrooms       1   0.05372 8.3569 -762.44
    ## - waterfront     1   0.08340 8.3866 -761.55
    ## - sqft_lot       1   0.12974 8.4329 -760.18
    ## <none>                       8.3031 -758.52
    ## - sqft_above     1   0.20530 8.5084 -757.96
    ## - grade          1   0.37397 8.6771 -753.07
    ## + condition      4   0.46075 7.8424 -750.67
    ## - sqft_living15  1   0.49647 8.7996 -749.58
    ## - view           4   1.15924 9.4624 -748.05
    ## - sqft_living    1   0.62115 8.9243 -746.08
    ## 
    ## Step:  AIC=-764.02
    ## price ~ bedrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + yr_built + sqft_living15 + sqft_lot15 + waterfront + 
    ##     view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - floors         1   0.01199 8.3160 -769.17
    ## - yr_built       1   0.04188 8.3459 -768.28
    ## - sqft_lot15     1   0.04198 8.3460 -768.28
    ## - bedrooms       1   0.05834 8.3624 -767.79
    ## - waterfront     1   0.08269 8.3867 -767.07
    ## - sqft_lot       1   0.12888 8.4329 -765.70
    ## <none>                       8.3040 -764.02
    ## - sqft_above     1   0.20543 8.5095 -763.45
    ## - grade          1   0.37382 8.6779 -758.57
    ## + bathrooms      1   0.00089 8.3031 -758.52
    ## + condition      4   0.45916 7.8449 -756.11
    ## - sqft_living15  1   0.49701 8.8010 -755.06
    ## - view           4   1.15859 9.4626 -753.56
    ## - sqft_living    1   0.71990 9.0239 -748.83
    ## 
    ## Step:  AIC=-769.17
    ## price ~ bedrooms + sqft_living + sqft_lot + grade + sqft_above + 
    ##     yr_built + sqft_living15 + sqft_lot15 + waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot15     1   0.04587 8.3619 -773.32
    ## - bedrooms       1   0.06444 8.3805 -772.77
    ## - waterfront     1   0.07195 8.3880 -772.55
    ## - yr_built       1   0.08008 8.3961 -772.30
    ## - sqft_lot       1   0.12935 8.4454 -770.85
    ## <none>                       8.3160 -769.17
    ## - sqft_above     1   0.24068 8.5567 -767.59
    ## + floors         1   0.01199 8.3040 -764.02
    ## + bathrooms      1   0.00032 8.3157 -763.67
    ## - grade          1   0.40363 8.7197 -762.89
    ## + condition      4   0.46880 7.8472 -761.55
    ## - sqft_living15  1   0.50584 8.8219 -759.99
    ## - view           4   1.15140 9.4674 -758.95
    ## - sqft_living    1   0.88496 9.2010 -749.51
    ## 
    ## Step:  AIC=-773.32
    ## price ~ bedrooms + sqft_living + sqft_lot + grade + sqft_above + 
    ##     yr_built + sqft_living15 + waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bedrooms       1   0.06110 8.4230 -777.03
    ## - waterfront     1   0.06601 8.4279 -776.88
    ## - sqft_lot       1   0.08630 8.4482 -776.28
    ## - yr_built       1   0.11759 8.4795 -775.36
    ## <none>                       8.3619 -773.32
    ## - sqft_above     1   0.26580 8.6277 -771.05
    ## + sqft_lot15     1   0.04587 8.3160 -769.17
    ## + floors         1   0.01588 8.3460 -768.28
    ## + bathrooms      1   0.00005 8.3618 -767.81
    ## + condition      4   0.47152 7.8904 -765.70
    ## - grade          1   0.48011 8.8420 -764.94
    ## - view           4   1.10608 9.4680 -764.46
    ## - sqft_living15  1   0.57347 8.9354 -762.32
    ## - sqft_living    1   0.87062 9.2325 -754.18
    ## 
    ## Step:  AIC=-777.03
    ## price ~ sqft_living + sqft_lot + grade + sqft_above + yr_built + 
    ##     sqft_living15 + waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot       1   0.06498 8.4880 -780.63
    ## - waterfront     1   0.06949 8.4925 -780.50
    ## - yr_built       1   0.10334 8.5263 -779.51
    ## <none>                       8.4230 -777.03
    ## - sqft_above     1   0.26308 8.6861 -774.88
    ## + bedrooms       1   0.06110 8.3619 -773.32
    ## + sqft_lot15     1   0.04253 8.3805 -772.77
    ## + floors         1   0.02239 8.4006 -772.17
    ## + bathrooms      1   0.00136 8.4217 -771.55
    ## + condition      4   0.49353 7.9295 -769.99
    ## - sqft_living15  1   0.53723 8.9602 -767.15
    ## - grade          1   0.57293 8.9959 -766.16
    ## - view           4   1.23030 9.6533 -765.15
    ## - sqft_living    1   0.82073 9.2437 -759.39
    ## 
    ## Step:  AIC=-780.63
    ## price ~ sqft_living + grade + sqft_above + yr_built + sqft_living15 + 
    ##     waterfront + view
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - yr_built       1   0.05948 8.5475 -784.41
    ## - waterfront     1   0.08110 8.5691 -783.78
    ## <none>                       8.4880 -780.63
    ## - sqft_above     1   0.23797 8.7260 -779.26
    ## + sqft_lot       1   0.06498 8.4230 -777.03
    ## + bedrooms       1   0.03978 8.4482 -776.28
    ## + floors         1   0.01854 8.4695 -775.66
    ## + sqft_lot15     1   0.00405 8.4839 -775.23
    ## + bathrooms      1   0.00026 8.4877 -775.12
    ## + condition      4   0.53837 7.9496 -774.88
    ## - sqft_living15  1   0.47245 8.9604 -772.66
    ## - grade          1   0.55598 9.0440 -770.35
    ## - view           4   1.37756 9.8655 -765.25
    ## - sqft_living    1   0.82341 9.3114 -763.09
    ## 
    ## Step:  AIC=-784.41
    ## price ~ sqft_living + grade + sqft_above + sqft_living15 + waterfront + 
    ##     view
    ## 
    ##                 Df Sum of Sq     RSS     AIC
    ## - waterfront     1   0.12735  8.6748 -786.24
    ## <none>                        8.5475 -784.41
    ## - sqft_above     1   0.21236  8.7598 -783.82
    ## + yr_built       1   0.05948  8.4880 -780.63
    ## + floors         1   0.05767  8.4898 -780.58
    ## + condition      4   0.58993  7.9575 -780.15
    ## + bedrooms       1   0.03699  8.5105 -779.97
    ## + sqft_lot15     1   0.02355  8.5239 -779.58
    ## + sqft_lot       1   0.02112  8.5263 -779.51
    ## + bathrooms      1   0.00571  8.5418 -779.06
    ## - sqft_living15  1   0.43084  8.9783 -777.68
    ## - grade          1   0.49669  9.0442 -775.86
    ## - sqft_living    1   0.81226  9.3597 -767.32
    ## - view           4   1.52048 10.0679 -765.71
    ## 
    ## Step:  AIC=-786.24
    ## price ~ sqft_living + grade + sqft_above + sqft_living15 + view
    ## 
    ##                 Df Sum of Sq     RSS     AIC
    ## <none>                        8.6748 -786.24
    ## - sqft_above     1   0.23546  8.9103 -785.09
    ## + waterfront     1   0.12735  8.5475 -784.41
    ## + yr_built       1   0.10573  8.5691 -783.78
    ## + condition      4   0.59522  8.0796 -781.87
    ## + floors         1   0.03882  8.6360 -781.84
    ## + bedrooms       1   0.03775  8.6371 -781.81
    ## + sqft_lot15     1   0.02554  8.6493 -781.46
    ## + sqft_lot       1   0.01911  8.6557 -781.27
    ## + bathrooms      1   0.00314  8.6717 -780.82
    ## - sqft_living15  1   0.41110  9.0859 -780.23
    ## - grade          1   0.52527  9.2001 -777.12
    ## - sqft_living    1   0.76472  9.4395 -770.72
    ## - view           4   1.87876 10.5536 -759.50

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
    ## price ~ sqft_living + grade + sqft_above + sqft_living15 + view
    ## 
    ## 
    ##               Step Df     Deviance Resid. Df Resid. Dev       AIC
    ## 1                                        229   7.842403 -750.6699
    ## 2  - sqft_basement  0 0.0000000000       229   7.842403 -750.6699
    ## 3      - condition  4 0.4607453534       233   8.303149 -758.5245
    ## 4      - bathrooms  1 0.0008884722       234   8.304037 -764.0153
    ## 5         - floors  1 0.0119941984       235   8.316031 -769.1733
    ## 6     - sqft_lot15  1 0.0458716791       236   8.361903 -773.3211
    ## 7       - bedrooms  1 0.0611033888       237   8.423007 -777.0256
    ## 8       - sqft_lot  1 0.0649821277       238   8.487989 -780.6294
    ## 9       - yr_built  1 0.0594755732       239   8.547464 -784.4082
    ## 10    - waterfront  1 0.1273500323       240   8.674814 -786.2431

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
    ##     Min      1Q  Median      3Q     Max 
    ## -0.6091 -0.1061  0.0000  0.1063  0.8061 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   13.42804    0.19687  68.209  < 2e-16 ***
    ## sqft_living    0.18114    0.02258   8.022 4.60e-14 ***
    ## grade          0.13420    0.02559   5.244 3.45e-07 ***
    ## sqft_living15  0.12516    0.02603   4.808 2.69e-06 ***
    ## waterfront1    0.64898    0.19839   3.271  0.00123 ** 
    ## condition2    -0.01836    0.27719  -0.066  0.94726    
    ## condition3     0.07990    0.19859   0.402  0.68779    
    ## condition4     0.17309    0.19720   0.878  0.38096    
    ## condition5     0.25614    0.19881   1.288  0.19887    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1957 on 240 degrees of freedom
    ## Multiple R-squared:  0.8061, Adjusted R-squared:  0.7996 
    ## F-statistic: 124.7 on 8 and 240 DF,  p-value: < 2.2e-16

``` r
coef(fit_fb)
```

    ##   (Intercept)   sqft_living         grade    sqft_above sqft_living15 
    ##   13.52474563    0.12951308    0.09678407    0.06561452    0.08705433 
    ##         view1         view2         view3         view4 
    ##    0.06906967    0.09912781    0.18608136    0.42520118

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
    ##  [1,] 0.03814746 0.3711721 0.3291050 0.3677916 0.03998696 0.3679102
    ##  [2,] 0.03362604 0.3883499 0.3679390 0.3697772 0.03858469 0.3768141
    ##  [3,] 0.03553366 0.3500051 0.3560182 0.3643317 0.03734573 0.3456373
    ##  [4,] 0.03217964 0.3741843 0.3938959 0.3575330 0.03726134 0.3703760
    ##  [5,] 0.03183003 0.3518372 0.3505339 0.3225746 0.03590589 0.3496497
    ##  [6,] 0.02897279 0.3437981 0.3221470 0.3800735 0.03544271 0.3336435
    ##  [7,] 0.03685385 0.3519591 0.3864961 0.3821805 0.04146703 0.3470710
    ##  [8,] 0.03584692 0.3755167 0.3636567 0.3640517 0.04129164 0.3657030
    ##  [9,] 0.03133315 0.3261018 0.3036293 0.3932292 0.03439824 0.3228079
    ## [10,] 0.03654672 0.3532299 0.3581997 0.3571418 0.04014860 0.3486229
    ## [11,] 0.03810296 0.3092235 0.3327261 0.3235834 0.04194704 0.3095343
    ## [12,] 0.03296904 0.3776137 0.3447957 0.3777042 0.03713794 0.3682461
    ## [13,] 0.03580230 0.3683751 0.3546405 0.3234691 0.04165080 0.3624207
    ## [14,] 0.03815898 0.3892211 0.3301834 0.2910672 0.04185927 0.3792320
    ## [15,] 0.03472313 0.3714201 0.3489487 0.3361758 0.03760514 0.3681535
    ## [16,] 0.03255105 0.3915845 0.3148204 0.3610014 0.03624908 0.3854784
    ## [17,] 0.03371050 0.3646863 0.3425948 0.3578209 0.03537577 0.3636498
    ## [18,] 0.03729158 0.3417259 0.3689368 0.3539138 0.04078907 0.3352166
    ## [19,] 0.03536069 0.3560139 0.3749731 0.3121434 0.03860345 0.3549028
    ## [20,] 0.02824231 0.3553039 0.3404358 0.3767707 0.03185459 0.3528702

5. Conclusion
-------------

To conclude, the important factors deciding house price turned out to be whether the house has the view of water front and nice overall view around the house. Therefore, rather than traits of the internal house structure such as area of livingroom, whether the house has a view of water near the house.

reference of image of AIC/BIC: <https://slideplayer.com/slide/10911060/>
