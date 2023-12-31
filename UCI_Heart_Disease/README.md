
Heart Disease
![](https://github.com/w7Ali/DataScience_ML/blob/main/UCI_Heart_Disease/wA_heart_Disease_Project.png)

For this exercise we will be using a subset of the UCI Heart Disease dataset, leveraging the fourteen most commonly used attributes. All identifying information about the patient has been scrubbed. 

The dataset includes 14 columns.
**Homogenity Score** describes the closeness of the clustering algorithm to a perfect cluster, where each cluster has data-points that belong to the same class lable (in this case, 1 or 0, sick or healthy). Basically, we can tell how much sick people we classify as healthy, and how many healthy people we classify as sick. The homogenity score is pretty low which means that we have quite a lot of missclassifications. 

**Completeness Score** describes how well we cluster members of a class in the correct class. This means we evalute whether all members of a true class is classified correctly. We get a very similar result to the homogenity score because it is essentially the same as we only have 2 clases. It's pretty low for the same reason the homogenity score is low. 

**V-measure**: is kind of like F1-score but for homogenity score and completeness score. The formual for this is V = (1+β) * [(hc) / (βh+c)]. Beta (β) = 1 is the default value taken. When we take the default beta value, this is just calcualting the harmonic mean between the two scores, which is why we get a low result in the measure. This concludes that we have a large amount of mis-clustering and misclassification.

**Adjusted Rand Score**: describes how similar the clusters are from one another. The range is from -1 to 1. 1 can be considered perfect labeling of the data while 0 is just randomly labeling the data. Basically, it gives a measure of how similar the points that are classified as 1 are to those classified as 0. This model is slightly better than random (0.474 > 0) but it's still far from the perfect labling of 1.

**Adjusted Mutual Information**: is also another measure of similarity. In particular, it measures how much information is shared between the clusters, while correcting for chance. We can see how much information we gain as opposed to splitting on the model. The value of 3.76 is pretty low and we conclude that this model shares quite a lot of information between clusters, and that splitting does not work very well.
