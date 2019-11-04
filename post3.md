Project proposal for Data Incubator Fellowship

1. Purpose

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


2.1 Model selection criteria
============================

![alt text here](https://github.com/hjung9/STAT576_1/blob/master/post3_1.png)

This is a typical multiple regression form, whre X1~Xp are variables and Beta1 to Betap are coefficients of each variable, beta0 is an intercept, e is an error term. It might be tempting to include as many variables as possible to consider many factors that would explain the dependent variable. Indeed, R^2 will increase as you include more variables in a model. But will all the variables actually contribute in predicting the dependent variable? The problem of R^2 is that it blindly increases regardless actual contribution of variables. That means, even if you include any irrelevant variable, R^2 still increases. There are several model selection criteria to handle this issue: AIC, BIC, Cp and adjusted R^2. Among these, AIC is most frequently used.

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


Advantages: - Computationally efficient and fast - Easy to interprete with smaller number of variables included in the final model

Disadvantages: - P-values do not provide meaningful information because of the bias - Not efficient in dealing with variables with high collinearity. It will just randomly remove one variable among highly correlated variables.

Even though I excluded some variables, there are still 13 variables so it needs to go further selection process to make more simpler, accurate model. In the following analysis, I will use forward, backward and bidirectinoal selection model with different model selection criteria.

4. Data analysis



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
```

Using stepAIC function automatically adds variables step by step. To see the results, use object\_name\_of\_model$anova

``` r
fit_fw = stepAIC(fit, direction="forward") # stepwise selection
```

    ## Start:  AIC=-875.29
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
    ##   Step Df Deviance Resid. Df Resid. Dev       AIC
    ## 1                        230    6.35734 -875.2929

This shows the final model of forward selection, but here is a problem. The final model contains all variables in the initial model and no variables were selected at all.

4.3 Backward selection model
============================

Then, what about backward selection method? Let's try.

``` r
fit_bw = stepAIC(fit, direction="backward") # stepwise selection
```

    ## Start:  AIC=-875.29
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## 
    ## Step:  AIC=-875.29
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - yr_built       1   0.00078 6.3581 -877.26
    ## - floors         1   0.01910 6.3764 -876.55
    ## - bedrooms       1   0.01956 6.3769 -876.53
    ## - sqft_lot15     1   0.04745 6.4048 -875.44
    ## <none>                       6.3573 -875.29
    ## - sqft_lot       1   0.05857 6.4159 -875.01
    ## - sqft_living15  1   0.06719 6.4245 -874.68
    ## - waterfront     1   0.20339 6.5607 -869.45
    ## - bathrooms      1   0.22945 6.5868 -868.46
    ## - sqft_above     1   0.33712 6.6945 -864.43
    ## - sqft_living    1   0.41323 6.7706 -861.61
    ## - condition      3   0.56324 6.9206 -860.16
    ## - grade          1   0.45645 6.8138 -860.03
    ## - view           4   0.92610 7.2834 -849.43
    ## 
    ## Step:  AIC=-877.26
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_living15 + sqft_lot15 + waterfront + 
    ##     view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bedrooms       1   0.01883 6.3770 -878.53
    ## - floors         1   0.02726 6.3854 -878.20
    ## - sqft_lot15     1   0.05122 6.4093 -877.26
    ## <none>                       6.3581 -877.26
    ## - sqft_lot       1   0.05781 6.4159 -877.01
    ## - sqft_living15  1   0.07076 6.4289 -876.51
    ## - bathrooms      1   0.23536 6.5935 -870.21
    ## - waterfront     1   0.23786 6.5960 -870.12
    ## - sqft_above     1   0.35273 6.7109 -865.82
    ## - sqft_living    1   0.41246 6.7706 -863.61
    ## - grade          1   0.46113 6.8192 -861.83
    ## - condition      3   0.59541 6.9535 -860.97
    ## - view           4   0.95554 7.3137 -850.40
    ## 
    ## Step:  AIC=-878.53
    ## price ~ bathrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + sqft_living15 + sqft_lot15 + waterfront + view + 
    ##     condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - floors         1   0.03310 6.4101 -879.24
    ## - sqft_lot       1   0.05017 6.4271 -878.57
    ## - sqft_lot15     1   0.05114 6.4281 -878.54
    ## <none>                       6.3770 -878.53
    ## - sqft_living15  1   0.06869 6.4456 -877.86
    ## - bathrooms      1   0.22214 6.5991 -872.00
    ## - waterfront     1   0.24533 6.6223 -871.13
    ## - sqft_above     1   0.36445 6.7414 -866.69
    ## - sqft_living    1   0.40108 6.7780 -865.34
    ## - condition      3   0.60687 6.9838 -861.89
    ## - grade          1   0.54758 6.9245 -860.01
    ## - view           4   0.96817 7.3451 -851.33
    ## 
    ## Step:  AIC=-879.24
    ## price ~ bathrooms + sqft_living + sqft_lot + grade + sqft_above + 
    ##     sqft_living15 + sqft_lot15 + waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot       1   0.04184 6.4519 -879.62
    ## <none>                       6.4101 -879.24
    ## - sqft_lot15     1   0.05308 6.4631 -879.18
    ## - sqft_living15  1   0.06954 6.4796 -878.55
    ## - waterfront     1   0.22284 6.6329 -872.73
    ## - bathrooms      1   0.22349 6.6335 -872.70
    ## - sqft_above     1   0.40389 6.8139 -866.02
    ## - grade          1   0.53267 6.9427 -861.36
    ## - condition      3   0.66197 7.0720 -860.77
    ## - sqft_living    1   0.55820 6.9682 -860.45
    ## - view           4   1.00038 7.4104 -851.13
    ## 
    ## Step:  AIC=-879.62
    ## price ~ bathrooms + sqft_living + grade + sqft_above + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot15     1   0.01535 6.4672 -881.02
    ## <none>                       6.4519 -879.62
    ## - sqft_living15  1   0.06361 6.5155 -879.17
    ## - waterfront     1   0.22782 6.6797 -872.98
    ## - bathrooms      1   0.25378 6.7057 -872.01
    ## - sqft_above     1   0.42132 6.8732 -865.87
    ## - sqft_living    1   0.52831 6.9802 -862.02
    ## - grade          1   0.57909 7.0310 -860.21
    ## - condition      3   0.75471 7.2066 -858.07
    ## - view           4   0.98367 7.4356 -852.28
    ## 
    ## Step:  AIC=-881.02
    ## price ~ bathrooms + sqft_living + grade + sqft_above + sqft_living15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## <none>                       6.4672 -881.02
    ## - sqft_living15  1   0.08466 6.5519 -879.79
    ## - waterfront     1   0.23593 6.7032 -874.10
    ## - bathrooms      1   0.24114 6.7084 -873.91
    ## - sqft_above     1   0.42509 6.8923 -867.17
    ## - sqft_living    1   0.55762 7.0249 -862.43
    ## - grade          1   0.57974 7.0470 -861.65
    ## - condition      3   0.76977 7.2370 -859.02
    ## - view           4   0.96841 7.4357 -854.28

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
    ## price ~ bathrooms + sqft_living + grade + sqft_above + sqft_living15 + 
    ##     waterfront + view + condition
    ## 
    ## 
    ##              Step Df     Deviance Resid. Df Resid. Dev       AIC
    ## 1                                       230   6.357340 -875.2929
    ## 2 - sqft_basement  0 0.0000000000       230   6.357340 -875.2929
    ## 3      - yr_built  1 0.0007822253       231   6.358122 -877.2622
    ## 4      - bedrooms  1 0.0188332965       232   6.376955 -878.5258
    ## 5        - floors  1 0.0330963005       233   6.410052 -879.2368
    ## 6      - sqft_lot  1 0.0418443235       234   6.451896 -879.6166
    ## 7    - sqft_lot15  1 0.0153518103       235   6.467248 -881.0249

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
    ## -0.49685 -0.10028 -0.00010  0.08363  0.87479 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   13.11682    0.17092  76.741  < 2e-16 ***
    ## sqft_living    0.16059    0.02458   6.533 3.99e-10 ***
    ## sqft_lot      -0.02339    0.01466  -1.595 0.112029    
    ## grade          0.11761    0.02485   4.734 3.82e-06 ***
    ## sqft_above     0.08854    0.02361   3.750 0.000223 ***
    ## sqft_living15  0.04485    0.02575   1.742 0.082847 .  
    ## sqft_lot15     0.02218    0.01670   1.328 0.185391    
    ## waterfront1    0.33532    0.10046   3.338 0.000982 ***
    ## view1          0.04308    0.07031   0.613 0.540680    
    ## view2          0.08245    0.04085   2.018 0.044695 *  
    ## view3          0.11414    0.04745   2.405 0.016932 *  
    ## view4          0.31720    0.05886   5.389 1.72e-07 ***
    ## condition3     0.35395    0.17156   2.063 0.040200 *  
    ## condition4     0.41353    0.17168   2.409 0.016780 *  
    ## condition5     0.51461    0.17363   2.964 0.003353 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1684 on 234 degrees of freedom
    ## Multiple R-squared:  0.8597, Adjusted R-squared:  0.8513 
    ## F-statistic: 102.5 on 14 and 234 DF,  p-value: < 2.2e-16

Notice the Final reduced model of backward selection still contains so many variables.

4.4 Mixed selection model
=========================

Let's see if the variable selection option, "both" with BIC criteria produce more parsimonious final models with fewer variables. To select BIC as a model selection criteria, set k=log(the number of rows in data) in the stepAIC function. (By default, the k=2, which is meant for AIC criteria)

``` r
fit_fb = stepAIC(fit, direction="both",k = log(nrow(sdat[train,])))
```

    ## Start:  AIC=-808.46
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_basement + yr_built + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ## 
    ## Step:  AIC=-808.46
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + yr_built + sqft_living15 + sqft_lot15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - yr_built       1   0.00078 6.3581 -813.95
    ## - floors         1   0.01910 6.3764 -813.23
    ## - bedrooms       1   0.01956 6.3769 -813.21
    ## - sqft_lot15     1   0.04745 6.4048 -812.13
    ## - sqft_lot       1   0.05857 6.4159 -811.70
    ## - sqft_living15  1   0.06719 6.4245 -811.36
    ## <none>                       6.3573 -808.46
    ## - waterfront     1   0.20339 6.5607 -806.14
    ## - bathrooms      1   0.22945 6.5868 -805.15
    ## - condition      3   0.56324 6.9206 -803.88
    ## - sqft_above     1   0.33712 6.6945 -801.11
    ## - sqft_living    1   0.41323 6.7706 -798.30
    ## - grade          1   0.45645 6.8138 -796.71
    ## - view           4   0.92610 7.2834 -796.67
    ## 
    ## Step:  AIC=-813.95
    ## price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
    ##     grade + sqft_above + sqft_living15 + sqft_lot15 + waterfront + 
    ##     view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - bedrooms       1   0.01883 6.3770 -818.73
    ## - floors         1   0.02726 6.3854 -818.40
    ## - sqft_lot15     1   0.05122 6.4093 -817.47
    ## - sqft_lot       1   0.05781 6.4159 -817.21
    ## - sqft_living15  1   0.07076 6.4289 -816.71
    ## <none>                       6.3581 -813.95
    ## - bathrooms      1   0.23536 6.5935 -810.41
    ## - waterfront     1   0.23786 6.5960 -810.32
    ## + yr_built       1   0.00078 6.3573 -808.46
    ## - condition      3   0.59541 6.9535 -808.21
    ## - sqft_above     1   0.35273 6.7109 -806.02
    ## - sqft_living    1   0.41246 6.7706 -803.81
    ## - grade          1   0.46113 6.8192 -802.03
    ## - view           4   0.95554 7.3137 -801.16
    ## 
    ## Step:  AIC=-818.73
    ## price ~ bathrooms + sqft_living + sqft_lot + floors + grade + 
    ##     sqft_above + sqft_living15 + sqft_lot15 + waterfront + view + 
    ##     condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - floors         1   0.03310 6.4101 -822.96
    ## - sqft_lot       1   0.05017 6.4271 -822.30
    ## - sqft_lot15     1   0.05114 6.4281 -822.26
    ## - sqft_living15  1   0.06869 6.4456 -821.58
    ## <none>                       6.3770 -818.73
    ## - bathrooms      1   0.22214 6.5991 -815.72
    ## - waterfront     1   0.24533 6.6223 -814.85
    ## + bedrooms       1   0.01883 6.3581 -813.95
    ## + yr_built       1   0.00005 6.3769 -813.21
    ## - condition      3   0.60687 6.9838 -812.65
    ## - sqft_above     1   0.36445 6.7414 -810.41
    ## - sqft_living    1   0.40108 6.7780 -809.06
    ## - view           4   0.96817 7.3451 -805.60
    ## - grade          1   0.54758 6.9245 -803.73
    ## 
    ## Step:  AIC=-822.96
    ## price ~ bathrooms + sqft_living + sqft_lot + grade + sqft_above + 
    ##     sqft_living15 + sqft_lot15 + waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot       1   0.04184 6.4519 -826.85
    ## - sqft_lot15     1   0.05308 6.4631 -826.42
    ## - sqft_living15  1   0.06954 6.4796 -825.79
    ## <none>                       6.4101 -822.96
    ## - waterfront     1   0.22284 6.6329 -819.97
    ## - bathrooms      1   0.22349 6.6335 -819.94
    ## + floors         1   0.03310 6.3770 -818.73
    ## + bedrooms       1   0.02467 6.3854 -818.40
    ## + yr_built       1   0.00646 6.4036 -817.69
    ## - condition      3   0.66197 7.0720 -815.04
    ## - sqft_above     1   0.40389 6.8139 -813.26
    ## - view           4   1.00038 7.4104 -808.92
    ## - grade          1   0.53267 6.9427 -808.60
    ## - sqft_living    1   0.55820 6.9682 -807.68
    ## 
    ## Step:  AIC=-826.85
    ## price ~ bathrooms + sqft_living + grade + sqft_above + sqft_living15 + 
    ##     sqft_lot15 + waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_lot15     1   0.01535 6.4672 -831.78
    ## - sqft_living15  1   0.06361 6.5155 -829.93
    ## <none>                       6.4519 -826.85
    ## - waterfront     1   0.22782 6.6797 -823.73
    ## + sqft_lot       1   0.04184 6.4101 -822.96
    ## - bathrooms      1   0.25378 6.7057 -822.77
    ## + floors         1   0.02477 6.4271 -822.30
    ## + bedrooms       1   0.01567 6.4362 -821.94
    ## + yr_built       1   0.00233 6.4496 -821.43
    ## - sqft_above     1   0.42132 6.8732 -816.62
    ## - condition      3   0.75471 7.2066 -815.86
    ## - view           4   0.98367 7.4356 -813.59
    ## - sqft_living    1   0.52831 6.9802 -812.78
    ## - grade          1   0.57909 7.0310 -810.97
    ## 
    ## Step:  AIC=-831.78
    ## price ~ bathrooms + sqft_living + grade + sqft_above + sqft_living15 + 
    ##     waterfront + view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## - sqft_living15  1   0.08466 6.5519 -834.06
    ## <none>                       6.4672 -831.78
    ## - waterfront     1   0.23593 6.7032 -828.38
    ## - bathrooms      1   0.24114 6.7084 -828.18
    ## + floors         1   0.02990 6.4374 -827.42
    ## + bedrooms       1   0.01994 6.4473 -827.03
    ## + sqft_lot15     1   0.01535 6.4519 -826.85
    ## + yr_built       1   0.00789 6.4594 -826.57
    ## + sqft_lot       1   0.00412 6.4631 -826.42
    ## - sqft_above     1   0.42509 6.8923 -821.45
    ## - condition      3   0.76977 7.2370 -820.33
    ## - view           4   0.96841 7.4357 -819.11
    ## - sqft_living    1   0.55762 7.0249 -816.70
    ## - grade          1   0.57974 7.0470 -815.92
    ## 
    ## Step:  AIC=-834.06
    ## price ~ bathrooms + sqft_living + grade + sqft_above + waterfront + 
    ##     view + condition
    ## 
    ##                 Df Sum of Sq    RSS     AIC
    ## <none>                       6.5519 -834.06
    ## + sqft_living15  1   0.08466 6.4672 -831.78
    ## - bathrooms      1   0.24707 6.7990 -830.36
    ## - waterfront     1   0.24776 6.7997 -830.33
    ## + sqft_lot15     1   0.03640 6.5155 -829.93
    ## + floors         1   0.03506 6.5168 -829.88
    ## + bedrooms       1   0.02118 6.5307 -829.35
    ## + yr_built       1   0.00065 6.5513 -828.57
    ## + sqft_lot       1   0.00000 6.5519 -828.54
    ## - condition      3   0.74200 7.2939 -823.90
    ## - sqft_above     1   0.46540 7.0173 -822.49
    ## - view           4   1.12155 7.6735 -816.78
    ## - sqft_living    1   0.71263 7.2645 -813.87
    ## - grade          1   0.90036 7.4523 -807.52

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
    ## price ~ bathrooms + sqft_living + grade + sqft_above + waterfront + 
    ##     view + condition
    ## 
    ## 
    ##              Step Df     Deviance Resid. Df Resid. Dev       AIC
    ## 1                                       230   6.357340 -808.4613
    ## 2 - sqft_basement  0 0.0000000000       230   6.357340 -808.4613
    ## 3      - yr_built  1 0.0007822253       231   6.358122 -813.9481
    ## 4      - bedrooms  1 0.0188332965       232   6.376955 -818.7291
    ## 5        - floors  1 0.0330963005       233   6.410052 -822.9576
    ## 6      - sqft_lot  1 0.0418443235       234   6.451896 -826.8548
    ## 7    - sqft_lot15  1 0.0153518103       235   6.467248 -831.7805
    ## 8 - sqft_living15  1 0.0846630037       236   6.551911 -834.0595

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
    ## -0.49176 -0.10324 -0.01056  0.08836  0.86159 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   13.13156    0.17910  73.319  < 2e-16 ***
    ## sqft_living    0.20774    0.02205   9.421  < 2e-16 ***
    ## grade          0.16099    0.02450   6.570 3.08e-10 ***
    ## sqft_living15  0.07001    0.02550   2.745  0.00651 ** 
    ## waterfront1    0.60659    0.09081   6.680 1.64e-10 ***
    ## condition3     0.36827    0.18031   2.042  0.04220 *  
    ## condition4     0.43200    0.17976   2.403  0.01701 *  
    ## condition5     0.54242    0.18148   2.989  0.00309 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1786 on 241 degrees of freedom
    ## Multiple R-squared:  0.8374, Adjusted R-squared:  0.8327 
    ## F-statistic: 177.3 on 7 and 241 DF,  p-value: < 2.2e-16

``` r
coef(fit_fb_f)
```

    ##   (Intercept)   sqft_living         grade sqft_living15   waterfront1 
    ##   13.13155848    0.20774237    0.16098958    0.07000783    0.60658992 
    ##    condition3    condition4    condition5 
    ##    0.36826766    0.43200243    0.54242443

Notice this method effectively selected variables. Only 5 variables in this model. In the backward selection model, sqft\_living, sqft\_lot, grade, sqft\_above, sqft\_living15 , sqft\_lot15, waterfront, view and condition were selected. In the mixed model, sqft\_living, grade, sqft\_living15, waterfront and condition were selected, which all overlap with the variables selected in the backward selection model. By using coef(), we can see the coefficients of each variable. In the final model with BIC, waterfront shows the heighest value, followed by condition of level 5th and 4th.

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


fit_bw=lm(price~sqft_living + sqft_lot + grade + sqft_above + sqft_living15 +sqft_lot15 + waterfront + view + condition,data=sdat[train,])

train_pred=fit_bw$fitted.values
test_pred=predict(fit_bw, newx=sdat[test,])

train_mse2=mean((y[train]-train_pred)^2) 
test_mse2=mean((y[test]-test_pred)^2)

bw_err=c(train_mse2,test_mse2)

# mixed selection
fit_fb = lm(price ~ sqft_living + grade + sqft_living15 + waterfront + condition,data=sdat[train,])

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

    ##            Fw.Tr     Fw.Ts      Bw.Tr     Bw.Ts      Mx.Tr     Mx.Ts
    ##  [1,] 0.03854994 0.3566798 0.04107357 0.3545153 0.04349423 0.3479373
    ##  [2,] 0.03316625 0.3079100 0.03349589 0.3070419 0.03675276 0.3037447
    ##  [3,] 0.03120398 0.3182645 0.03236781 0.3222314 0.03418947 0.3173625
    ##  [4,] 0.03210207 0.3716545 0.03311380 0.3706834 0.03575343 0.3661864
    ##  [5,] 0.02997127 0.3482086 0.03024670 0.3497836 0.03277115 0.3546723
    ##  [6,] 0.03170068 0.3671784 0.03200378 0.3669262 0.03427839 0.3619585
    ##  [7,] 0.03153978 0.3615883 0.03288978 0.3612726 0.03523952 0.3588733
    ##  [8,] 0.02764454 0.3446462 0.02820575 0.3439475 0.03201508 0.3376584
    ##  [9,] 0.03266047 0.3313236 0.03335406 0.3333272 0.03703585 0.3259621
    ## [10,] 0.02972202 0.3577073 0.03080810 0.3524599 0.03570493 0.3434115
    ## [11,] 0.02935060 0.3525722 0.02968107 0.3540774 0.03319157 0.3563956
    ## [12,] 0.03586573 0.3513310 0.03647748 0.3492833 0.03954455 0.3427525
    ## [13,] 0.03118063 0.3053622 0.03182535 0.3049862 0.03418977 0.3027629
    ## [14,] 0.03111783 0.3637338 0.03146133 0.3618837 0.03455182 0.3583814
    ## [15,] 0.02759477 0.3803476 0.02876178 0.3770789 0.03252518 0.3713948
    ## [16,] 0.04019125 0.3073241 0.04110097 0.3091522 0.04303316 0.3090333
    ## [17,] 0.02657515 0.3355539 0.02729096 0.3351013 0.03028893 0.3303060
    ## [18,] 0.02634864 0.3434590 0.02677504 0.3408507 0.03018181 0.3387299
    ## [19,] 0.03327167 0.3286990 0.03474313 0.3304023 0.03698597 0.3302760
    ## [20,] 0.02589888 0.3467955 0.02701766 0.3481924 0.03209962 0.3445554

5. Conclusion

To conclude, the important factors deciding house price turned out to be whether the house has the view of water front and overall condition of house. Therefore, rather than traits of the internal house structure such as area of livingroom, whether the house has a view of water near the house and overall condition strongly predicted house price.

reference of image of AIC/BIC: <https://slideplayer.com/slide/10911060/>
