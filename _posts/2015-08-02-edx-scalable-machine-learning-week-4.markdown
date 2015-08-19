---
layout: post
title:  "EdX: Scalable Machine Learning Week 4 Notes"
date:   2015-08-01 15:00:00
comments: True
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

#### **Categorical Data and Feature Encoding**: 
Raw data is often non-numeric, how do we handle them?

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

OHE suffers from some issues. **Statistically**: inefficient learning. **Computationally**: increased communication in parallel algorithm. Can reduce dimension by discarding rare features, but might throw away useful information.

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

---

### **Labs Highlights**

* A good practice in coding is if you need to do some complex operation/transformation, it's always good to try it out on a few sample examples. Write tests to verify that you understand how it works before moving on and apply it to the whole dataset. Once you understand how it works, you can define the transformation function.

#### **One Hot Encoding**

* Step 1: Extract features
* Step 2: Create a Dictionary
* Step 3: Create the OHE function
* Step 4: Use OHE to create proper training data

**Extract features**

{% highlight python %}
def parsePoint(point):
    """Converts a comma separated string into a list of (featureID, value) tuples.

    Note:
        featureIDs should start at 0 and increase to the number of features - 1.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.

    Returns:
        list: A list of (featureID, value) tuples.
    """
    data = point.split(',')
    label, rawFeats = data[0], data[1:]
    return [(featureId, featureVal) for (featureId, featureVal) in enumerate(rawFeats)]
{% endhighlight %} 

**Automated creation of an OHE dictionary**

{% highlight python %}
def createOneHotDict(inputData):
    """Creates a one-hot-encoder dictionary based on the input data.

    Args:
        inputData (RDD of lists of (int, str)): An RDD of observations where each observation is
            made up of a list of (featureID, value) tuples.

    Returns:
        dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
            unique integers.
    """
    DistinctFeats = (inputData
                     .flatMap(lambda x: x)
                     .distinct())
    OHEDict = (DistinctFeats
               .zipWithIndex()
               .collectAsMap())
    return OHEDict
{% endhighlight %} 

**Create oneHotEncoding function**

{% highlight python %}
def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
"""Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        You should ensure that the indices used to create a SparseVector are sorted.

    Args:
        rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
        OHEDict (dict): A mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length numOHEFeats with indicies equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.
    """
    args = [(OHEDict[(featureID, featureValue)], 1) for (featureID, featureValue) in rawFeats]
    return SparseVector(numOHEFeats, args)  
{% endhighlight %} 

**Generate Training Set using LabeledPoint**

{% highlight python %}
def parseOHEPoint(point, OHEDict, numOHEFeats):
    """Obtain the label and feature vector for this raw observation.

    Note:
        You must use the function `oneHotEncoding` in this implementation or later portions
        of this lab may not function as expected.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.
        OHEDict (dict of (int, str) to int): Mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The number of unique features in the training dataset.

    Returns:
        LabeledPoint: Contains the label for the observation and the one-hot-encoding of the
            raw features based on the provided OHE dictionary.
    """
    data = point.split(',')
    label, rawFeats = data[0], data[1:]
    rawFeatures = [(featureID, featureVal) for (featureID, featureVal) in enumerate(rawFeats)]
    features = oneHotEncoding(rawFeatures, OHEDict, numOHEFeats)
    return LabeledPoint(label, features)
{% endhighlight %}

#### **Logistic Regression Modeling**

* Step 1: Define Compute Log Loss function
* Step 2: Compute Baseline Logloss
* Step 3: Compute Model logloss

**Compute Log Loss**

{% highlight python %}
def computeLogLoss(p, y):
    """Calculates the value of log loss for a given probabilty and label.

    Note:
        log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
        and when p is 1 we need to subtract a small value (epsilon) from it.

    Args:
        p (float): A probabilty between 0 and 1.
        y (int): A label.  Takes on the values 0 and 1.

    Returns:
        float: The log loss value.
    """    
    epsilon = 10e-12
    if p == 0:
        p = p + epsilon
    elif p == 1:
        p = p - epsilon
    return y * -log(p) + (1-y) * (-log(1 - p))
{% endhighlight %}

**Baseline Log Loss**

{% highlight python %}
classOneFracTrain = OHETrainData.map(lambda lp: lp.label).sum() / len(OHETrainData.collect())
print classOneFracTrain

logLossTrBase = OHETrainData.map(lambda lp: computeLogLoss(classOneFracTrain, lp.label)).sum() / len(OHETrainData.collect())
{% endhighlight %}

**Compute Model Logloss**

{% highlight python %}
model0 = LogisticRegressionWithSGD.train(OHETrainData, 
                                         iterations = numIters, 
                                         step = stepSize, 
                                         miniBatchFraction = 1.0, 
                                         initialWeights = None, 
                                         regParam = regParam, 
                                         regType = regType, 
                                         intercept = includeIntercept)
{% endhighlight %}

**Define prediction function**

{% highlight python %}
def getP(x, w, intercept):
    """Calculate the probability for an observation given a set of weights and intercept.

    Note:
        We'll bound our raw prediction between 20 and -20 for numerical purposes.

    Args:
        x (SparseVector): A vector with values of 1.0 for features that exist in this
            observation and 0.0 otherwise.
        w (DenseVector): A vector of weights (betas) for the model.
        intercept (float): The model's intercept.

    Returns:
        float: A probability between 0 and 1.
    """
    rawPrediction = x.dot(w) + intercept

    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    sigmoid = (1 + exp(-rawPrediction)) ** -1
    return sigmoid
    
trainingPredictions = OHETrainData.map(lambda lp: getP(lp.features, model0.weights, model0.intercept))
{% endhighlight %}

**Evaluate the model**

{% highlight python %}
def evaluateResults(model, data):
    """Calculates the log loss for the data given the model.

    Args:
        model (LogisticRegressionModel): A trained logistic regression model.
        data (RDD of LabeledPoint): Labels and features for each observation.

    Returns:
        float: Log loss for the data.
    """
    resultVector = data.map(lambda lp: computeLogLoss(getP(lp.features, model.weights, model.intercept), lp.label))
    return resultVector.sum() / len(resultVector.collect())
    
logLossTrLR0 = evaluateResults(model0, OHETrainData)
{% endhighlight %}

#### **Feature Hashing**

* Step 1: Create Hash function
* Step 2: Create Hashed feature
* Step 3: Train Model based on hashed features

{% highlight python %}
def hashFunction(numBuckets, rawFeats, printMapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use printMapping=True for debug purposes and to better understand how the hashing works.

    Args:
        numBuckets (int): Number of buckets to use as features.
        rawFeats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        printMapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)
{% endhighlight %}

**Create Hashed feature**

{% highlight python %}
def parseHashPoint(point, numBuckets):
    """Create a LabeledPoint for this observation using hashing.

    Args:
        point (str): A comma separated string where the first value is the label and the rest are
            features.
        numBuckets: The number of buckets to hash to.

    Returns:
        LabeledPoint: A LabeledPoint with a label (0.0 or 1.0) and a SparseVector of hashed
            features.
    """
    data = point.split(',')
    label, rawFeats = data[0], data[1:]
    #zipWIthIndex
    rawFeaturePairs = [(featureID, featureVal) for (featureID, featureVal) in list(enumerate(rawFeats))]
    hashedFeature = hashFunction(numBuckets, rawFeaturePairs)
    SparsedHashedFeature = SparseVector(numBuckets, hashedFeature)
    return LabeledPoint(label, SparsedHashedFeature)
{% endhighlight %}

**Train a logistic regression model based on hashed features**

{% highlight python %}
stepSizes = [1, 10]
regParams = [1e-6, 1e-3]
for stepSize in stepSizes:
    for regParam in regParams:
        model = (LogisticRegressionWithSGD
                 .train(hashTrainData, numIters, stepSize, regParam=regParam, regType=regType,
                        intercept=includeIntercept))
        logLossVa = evaluateResults(model, hashValidationData)
        print ('\tstepSize = {0:.1f}, regParam = {1:.0e}: logloss = {2:.3f}'
               .format(stepSize, regParam, logLossVa))
        if (logLossVa < bestLogLoss):
            bestModel = model
            bestLogLoss = logLossVa
{% endhighlight %}