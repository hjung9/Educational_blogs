post2
================
Hwayoung Jung
December 2, 2018

Within-note variation in Tufted titmice (*baeolophus bicolor*) Purpose: To examine weather within-note property of D notes are associated with threat level

1.  Purpose The purpose of this post is to show how to examine presence of hidden subgroup and the subgroups match with a specific group category that is already known. I am studying the vocal behavior of Tufted titmice, commonly found species in Tennessee. The titmice produce various types of calls that convey threat information to other birds to alert them. There are 10 acoustic measurements of the titmice calls and 1 threat context. The variables in the dataset are:

-   Threat: Ordinal level of predation threat with three levels: low, medium and high. This will be regarded as a categorical variable.
-   No\_notes: The total number of calls produced
-   Duration: The temporal duration of calls measured in millisecond
-   Distomax: is temporal duration to measure the time taken to reach the maximum amplitude from begging of the call.
-   RMS: loudness of the call with 3dB deducted from the peak amplitude
-   Peakfreq: Peak frequency (high or low tone) of a call in Khz
-   Peakamp: Peak amplitude (loudness) of a call
-   Fundfreq: fundamental frequency in Khz
-   Minfreq: minimum frequency in Khz
-   Maxfreq: Maximum frequency in Khz
-   Entropy: The level of tonality (clear and pure tone like flute sound, or harsh sound such as cat's hissing call) Among 12 variables, threat will be used as dependent variables and all other variables will be dependent variables.

1.  Overview of analysis techniques 2.1 Hierarchical clustering Hierarchical clustering is a technique by which group similar data points together by using hierarchical tree model, a dendrogram. (Dendrogram plot here) Dendrogram is a figure that shows how each data point is grouped. Bottom of the plot shows all data points called leaves. They are grouped by clades (branch), and the leaves that are combined in the same height of branch are in the same cluster. Height of the branch indicates more dissimilarity between clusters.

There are largely 2 ways of merging groups: agglomerative and divisive. Agglomerative is basically a bottom-up approach. It starts merging leaves into a bigger cluster based on similarity, until all nodes and leaves are merged into one root. Divisive clustering is a top-down approach. It starts dividing one root into nodes and leaves based on dissimilarity until all nodes get separated into leaves.

There are 3 ways of linkage - single linkage: Calculate all pairwise distance and identify the closest distance between 2 clusters. Fast in calculation and works well for data of which order is important. Also good for separating few isolated groups. - Complete linkage Calculate all pairwise distance and identify the farthest distance between groups to maximize dissimilarity between clusters.

Distance is usually calculated with Euclidean distance, but there are many other options such as manhattan, bray-curtis.. etc. - Average linkage calculate all pairwise distance with average distance between 2 clusters

Advantage of using hierarchical clustering: It does not require to fix the number of the cluster from the beginning, unlike K-means clustering. Cons:

2.2 PCA (principal component analysis) PCA is a technique to reduce dimension by calculating principal components that effectively represent variability with smaller numbers of variables. Normal regression analyses just use the variables themselves, but PCA creates new variables (PCs), each of which consists of original variables by differing intensity and direction. Suppose there are P variables X1, X2,.., Xp and you want to make scatter plots for every pair of variables. The number of cases of choosing 2 variables out of P is P(p-1)/2, which seems to be too many. PCA is based on an idea that not all variables contribute for explaining variation of dependent variable. It aims to find out the most important variables. What does the PC look like? Let's take an example of PC1, the PC that explains the largest variance. PC1= (equation here) ----- are "loadings" SS=1 *x*=0

Before running PCA, all variables should be centered or scaled because PCA focuses on variance. Since I am not going to use PCA as a separate method in this posting, and just calculate PCs that are going to be used in K-means clustering, I am not going to show how to interpret results.

2.3 K-means clustering K-means clustering divide observations into K clusters that do not overlap to find out data points that have similar traits. Minimize variance of distance among clusters Partitioning in the way that increase inner similarity of observations in the same cluster It requires data to take Euclidean distance

How K-means cluster works Decide the number of K. we can choose any natural number from 1. K Centroids (the mean center of cluster) will be randomly assigned first, so all centroids will almost overlap. a. In the next step, nearest data points from each centroid will be assigned (labeled) as a cluster that belongs to a cluster. b. Then, centroid will be calculated again and moved to the point, where is the mean of the cluster. As procedure a and b are iterated repeatedly, the centroids will become more stable and eventually stay in the same place.

1.  Pros and cons

2.  Analsysis 4.1 hierarchical clustering 4.1.2 hierarchical clustering with euclidean distance and complete linkage

``` r
dat=read.csv("TUTI_post2_1.csv",header=T)
plot(dat)
#standardizing all variables except the categorical variable, "threat". All acoustic measurements are standardized.
sdat=scale(dat[,2:11])
#dat[,1] is the dependent variable, that is categorical with 3 threat levels: 10m from human (lowest level of threat, 5m from human-intermediate level of threat, 1m from human-highest level of threat)
print(levels(dat[,1]))
```

    ## [1] "10m" "1m"  "5m"

``` r
#the order of dat[,1] is set as 10m,1m and 5m.
#Let's re-order for easier interpretation, according to threat level.

dat[,1]=factor(dat[,1], levels(dat[,1])[c(1,3,2)])
print(levels(dat[,1])) #Now it is ordered as 10m, 5m and 1m. From less threat to highest threat level.
```

    ## [1] "10m" "5m"  "1m"

``` r
# In this post, I will comparing whether clusters match with the 3 threats condition: low, medium, high,
# using hierarchical clustering (euclidean distance and complete linkage and euclidan distance with single linkage), and K-means. Because the number of threat condition is 3, I will fix the number of clusters to be 3 in all methods to match the cluster labels and threat conditions later. 
d = dist(sdat, method = "euclidean") # distance matrix
fit_comp = hclust(d, method="complete")
rect.hclust(fit_comp, k=3, border="red")
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
hc_comp_labels= cutree(fit_comp, k=3)
plot(fit_comp) # display dendogram
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-1-2.png)

4.1.3 hierarchical clustering with euclidean distance and single linkage

``` r
fit_sing = hclust(d, method="single")
hc_sing_labels= cutree(fit_sing, k=3)
plot(fit_sing)
rect.hclust(fit_sing, k=3, border="red")
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-2-1.png) 4.2.1 Kmeans with Principal component axis

``` r
pcs=princomp(sdat,scale=F)#since we already standardized data, x need to do one more time.
```

    ## Warning: In princomp.default(sdat, scale = F) :
    ##  extra argument 'scale' will be disregarded

``` r
plot(pcs,type='l')
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
summary(pcs)
```

    ## Importance of components:
    ##                           Comp.1    Comp.2    Comp.3    Comp.4     Comp.5
    ## Standard deviation     1.5500660 1.4031017 1.3417161 1.2194316 0.86536445
    ## Proportion of Variance 0.2408508 0.1973450 0.1804550 0.1490605 0.07506645
    ## Cumulative Proportion  0.2408508 0.4381958 0.6186508 0.7677114 0.84277780
    ##                            Comp.6     Comp.7     Comp.8     Comp.9
    ## Standard deviation     0.75036329 0.66533693 0.50022168 0.47784037
    ## Proportion of Variance 0.05644051 0.04437425 0.02508261 0.02288829
    ## Cumulative Proportion  0.89921831 0.94359256 0.96867517 0.99156346
    ##                            Comp.10
    ## Standard deviation     0.290107004
    ## Proportion of Variance 0.008436536
    ## Cumulative Proportion  1.000000000

``` r
pcs=prcomp(sdat,scale=F)
pcs1to3=data.frame(pcs$x[,1:3])#extract PC1 to PC3
plot(pcs1to3)
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-3-2.png)

``` r
fit_km_pc=kmeans(pcs1to3,3)
km_pc_labels=fit_km_pc$cluster
plot(pcs1to3,col=fit_km_pc$cluster)
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-4-1.png) 4.2.2 Kmeans with Raw data

``` r
#what happens if we don't use pc axis and just use axis of raw data?
fit_km_r=kmeans(sdat,3)
km_r_labels=fit_km_r$cluster
plot(dat,col=fit_km_r$cluster)
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-5-1.png) 4.3 Prediction accuracy of 4 methods above

So far, we performed 2 hierarchical clustering and K-means analysis combined with PCA. With the clusters obtained in the previous section, we will going to match with actual dependent variable, threat levels to see how accurately the clusters are divided, by using confusion matrix.

``` r
table(hc_comp_labels,dat[,1])#confusion matrix of hierarchical clustering with complete linkage
```

    ##               
    ## hc_comp_labels 10m  5m  1m
    ##              1   0   3 185
    ##              2  33  81  84
    ##              3   6  14   9

``` r
table(hc_sing_labels,dat[,1])#confusion matrix of hierarchical clustering with complete linkage
```

    ##               
    ## hc_sing_labels 10m  5m  1m
    ##              1  39  96 274
    ##              2   0   2   1
    ##              3   0   0   3

``` r
table(km_pc_labels,dat[,1])#confusion matrix of K-means with PC 
```

    ##             
    ## km_pc_labels 10m  5m  1m
    ##            1   0   4 172
    ##            2   0   4  50
    ##            3  39  90  56

``` r
table(km_r_labels,dat[,1])#confusion matrix of K-means analysis on raw data
```

    ##            
    ## km_r_labels 10m  5m  1m
    ##           1   0   0 185
    ##           2   0   4  41
    ##           3  39  94  52

``` r
#I have selected variables that relatively separate clusteres better.
#pairs(dat[,c(1,3,6,7,8)],col=)
pairs(sdat,col=dat$threat)
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
#As we see here, 5m clusters (intermediate level of threat, red) are separated from 1m clusters (high level of threat, green) and 10m clusters (low level of threat, black).
#Duration, Peak frequency, and fundamental frequency increased for 5m clusters (intermediate level of threat).
#Both fundamnental frequency and



#plot(dat$fundfreq,dat$duration,col=k$cluster,pch=16)
#legend("bottomright", legend=levels(as.factor(dat[,1])),cex=1, pch=16, col=c(1:3))
#fit_km = kmeans(sdat, 3)
#km_labels=fit_km$cluster
```

It seems that one variable, "peakamp" is somewhat positively associated with threat level. "RMS" seems also positively correlated with threat level, but less obvious than "peakamp"

``` r
#violin plot
library("ggplot2")
```

    ## Warning: package 'ggplot2' was built under R version 3.4.4

``` r
#Peak amplitude: violin plot overlaid with boxplot
ggplot(dat,aes(x=dat$threat,y=dat[,8],fill=dat$threat))+
  geom_violin(trim=F)+
  geom_boxplot(width=0.1,fill="white")+
  theme_classic()+
  scale_fill_brewer(palette="Reds")+
  labs(title="    Plot of Peak amplitude  by threat level",x="Threat level (Distance bw a bird and a human) ", y = "Peak amplitude")+
  theme(legend.position="none")
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r
#RMS: violin plot overlaid with boxplot
ggplot(dat,aes(x=dat$threat,y=dat[,5],fill=dat$threat))+
  geom_violin(trim=F)+
  geom_boxplot(width=0.03,fill="white")+
  theme_classic()+
  scale_fill_brewer(palette="Reds")+
  labs(title="    Plot of RMS  by threat level",x="Threat level (Distance bw a bird and a human) ", y = "RMS")+
  theme(legend.position="none")
```

![](post2_1_files/figure-markdown_github/unnamed-chunk-7-2.png)

``` r
#as the threat level increases, they increased peak amplitude.
```

(Tentative)conclusions: Because the low prediction accuracy shown in those confusion matrices, most of variables do not seem to contribute to explain variation in the threat level. However, at least peak amplitude (and possibly RMS) changes accross threat level.