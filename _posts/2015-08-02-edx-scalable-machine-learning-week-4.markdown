---
layout: post
title:  "EdX: Scalable Machine Learning Week 4 Notes"
date:   2015-08-01 15:00:00
---
### **Lecture Highlights**

* **CTR prediction** is the canonical Machine Learning problem involving Logistic Regression

* **Players** involve:
  * *Publishers*: Make money displaying Ads on their sites (e.g. NYTimes)
  * *Matchmakers*: Match publishers with advertisers (e.g. Mopub, Google)
  * *Advertisers*: Pay for their ads to be displayed. Want to attract business and drive conversion (e.g. Marc Jacob)

* CTR prediction using **logistic regression model**
  * Observations are user-ad-publisher triplets
  * Labels are {not-click, click}
  * Given a set of labeled observations, we want to predict whether a new user-ad-publisher triplet will result in a click

* **Evaluation Criterion** is typically 0-1 loss: penalty is 0 for correct prediction, and 1 otherwise
  * 0-1 loss function is not convex, so we approximate it
  * SVM (hinge), Logistic Regression (log-loss), Adaboost (exponential)

* **ROC curve** plots False Positive Rate with True Positive Rate
  * A random classifier will fall on 45 degree line, depending on the threshold. Not hard to think about, since the denominator for FP is negative examples, and TP is positive examples. A random classifier with threshold t would just predict + with t% of the time. This is true for both FP and TP, so the performance would fall on 45 degree line.

* Again, remember the **intuition of log-loss**

#### **Categorical Data and Feature Encoding**: Raw data is often non-numeric, how do we handle them?
* Use Method that support categorical feature (Decision tree, Naive Bayes), but model options are limited
* Convert these categorical features to numeric feature
  * Encode category to numeric. e.g. cat1 = 1, cat2 = 2, catx = x ...etc. Not good!! since this introduces inherent ordering, which might not be there. (ordinal - not ideal, categorial - bad)
  * Better approaches include (One-Hot-Encoding) & (Feature Hashing)

#### **One-Hot-Encoding**

* Step 1: Create OHE Dictionary. See [Slide](https://courses.edx.org/c4x/BerkeleyX/CS190.1x/asset/CS190.1x_week4b.pdf)
* Step 2: Create Features with Dictionary (bag of word dictionary from training set)
* Step 3: Create Sparse OHE feature

{% highlight ruby %}
Animal = {'bear','cat','mouse'}; Color = {'black', tabby'} 
=> 
(Animal, 'bear') => 0, (Animal, 'cat') => 1, (Animal, 'mouse') => 2, (Color, 'black') => 3, ...etc

A1 = ['mouse', 'black', -] => A1 = [0,0,1,1,0,0,0]

A1 = [0,0,1,1,0,0,0] -> A1 = [(2,1), (3,1)]
{% endhighlight %}

OHE suffers from some issues. Statistically: inefficient learning. Computationally: increased communication in parallel algorithm. Can reduce dimension by discarding rare features, but might throw away useful information.

#### **Feature Hashing**

* Use hashing principle to reduce feature dimension
* Obviates need to compute expensive OHE dictionary
* Preserves sparsity
* Theoretical underpinning - approximate OHE dot-product-ing 

{% highlight ruby %}
Daatapoints: 7 feature categories
Hash buckets: m = 4
A1 = ['mouse', 'black', -]
H(Animal, 'mouse') = 3
H(Color, 'black') = 2
A1 = [0 0 1 1]

A2 = ['cat', 'tabby', 'mouse']
H(Animal, ’cat’) = 0
H(Color, ’tabby’) = 0
H(Diet, ’mouse’) = 2
A2 = [2 0 1 0]
{% endhighlight %}
