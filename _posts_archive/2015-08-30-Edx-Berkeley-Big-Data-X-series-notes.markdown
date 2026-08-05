---
layout: post
title:  "EdX Berkeley X-series: Big Data, Machine Learning, and Spark Notes"
date:   2015-08-29 15:00:00
comments: True
---
Compiling notes from each week into one giant file

## Lab: Parsing Apache Web Logs

### **Lecture Highlights**

* Semi-structure data: Mainly focus on tabular and .csv files
* Structured data: SQL, brief mentions on pandas and PySpark datagframe. Didn't really show examples

Most of the stuff covered here are pretty standard.

### **Lab Highlights**

Explore Web Server Access Log using NASA's HTTP server log. Some standard questions:

* What are the statistics for content being returned? Sizes, statuses?
* What are the types of return codes?
* How many 404 (page not found) errors are there?
* How many unique hosts per day?
* How many requests per day?
* On average, how many requests per host?
* How many 404 errors per day?

#### Parsing Apache logs ####

**Apache Format log**: See lab for a detailed description of what each field represents
{% highlight python %}
127.0.0.1 - - [01/Aug/1995:00:00:01 -0400] "GET /images/launch-logo.gif HTTP/1.0" 200 1839
{% endhighlight %}

{% highlight python %}
def parse_apache_time(s):
    return datetime.datetime(int(s[7:11]),
                             month_map[s[3:6]],
                             int(s[0:2]),
                             int(s[12:14]),
                             int(s[15:17]),
                             int(s[18:20]))
{% endhighlight %}

{% highlight python %}
def parseApacheLogLine(logline):
  APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)" (\d{3}) (\S+)'
    match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)
    if match is None:
        return (logline, 0)
    size_field = match.group(9)
    if size_field == '-':
        size = long(0)
    else:
        size = long(match.group(9))
    return (Row(
        host          = match.group(1),
        client_identd = match.group(2),
        user_id       = match.group(3),
        date_time     = parse_apache_time(match.group(4)),
        method        = match.group(5),
        endpoint      = match.group(6),
        protocol      = match.group(7),
        response_code = int(match.group(8)),
        content_size  = size
    ), 1)
{% endhighlight %}

#### Spark RDD oeprations ####
From the exercises, I practiced:

* **map**, **filter**, **flatmap**
* **distinct**, **count**
* **cache**: good for caching RDD into memory
* **collect**: transform an RDD into python object

* **groupByKey**: the `groupByKey()` transformation groups all the elements of the RDD with the same key into a single **list** in one of the partitions. See exercise `2b` from scalable ML for more details.
* **reduceByKey**: The `reduceByKey()` transformation gathers together pairs from `pairRDD` that have the same key and applies the function provided to two values at a time, iteratively reducing all of the values to a single value
* **sortByKey**: again, if you have (key, value) pairs, can sort by key

* **zipWithIndex()**: like zip in python. 

* **take**: `take(10)` take and collect 10 records from RDD to python object
* **takeOrdered**: `takeOrdered(10, lambda (host, count): -count)` take 10 records but order by count, in reversed order

* **join**: typical join operation

These are the most basic operations you need to know. To learn more about, check out official [PySpark API] page.

[PySpark API]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD

## Lab: Entity Resolution, Document Similarity

### **Lecture Highlights**

* **Data Quality**
  * Issues comes up during ETL process
  * Data is dirty on its own
  * Transformations corrupt data
  * clean datasets screwed up by integration

* **Data Gathering**
  * How does data enter the system?
    * Experimentation, observation, collection
    * Watch out for manual entry
    * duplicates
    * No uniform standards for content and formats
    * measurement errors
  * Potential solutions
    * preemptive: build in integrity checks
    * retrospective: automated detection of glitches. cleaning afterward

* **Data Delivery**
  * Issues
    * Destroying/mutilating information by bad pre-processing
    * Loss of data
  * Potential solution
    * build reliable transmission protocols
    * verfication: checksums, verification parser
    * data quality commitment from data supplier

* **Data Cleaning**
  * Parsing text into fields (Separator issues)
  * Dealing with missing data
  * Entity Resolution (this lab!)
  * Unit mismatch
  * Fields too long
  * Redundant records
  * formatting issues
  * Distortion, left and right censorship, dependence
  * selection bias: *likelihood of a sample depends on its value*

* **Data Storage**
  * Issues
    * problems in logical storage: poor metadata, inappropriate data models. Ad-hoc modifications. Hardware/software constraints
  * Potential solution
    * metadata specifications
    * Assume everything bad will happen

* **Data Retrieval**
  * plain mistakes: inner join v.s. outer join, not understanding NULL values
  * computational constraints: full history too expensive
  * Incompatibility: ASCII, Unicode

* **Data Mining/Analysis**
  * Scale and performance
  * Confidence bounds
  * Black boxes
  * Attachment to models
  * Insufficient domain expertise
  * Casual empiricism

### [**Lab Highlights**]

Entity Resolution (ER), or "Record linkage" is the term used by statisticians, epidemiologists, and historians, among others, to describe the process of joining records from one data source with another that describe the same entity. Our terms with the same meaning include, entity disambiguation, entity linking, and duplicate detection ...etc.



[**Lab Highlights**]: https://github.com/robert8138/edx_scalable_machine_learning/blob/master/labs/lab3_text_analysis_and_entity_resolution_student.ipynb

### **Bag of Words**
The idea is to treat strings, a.k.a. documents, as unordered collections of words, or tokens, i.e., as bags of words. Tokens become the atomic unit of text comparison. 

If we want to compare two documents, we count how many tokens they share in common. The power of this approach is that it makes string comparisons insensitive to small differences that probably do not affect meaning much, for example, punctuation and word order.

{% highlight python %}
def tokenize(string):
    string = string.lower()
    tokens = [token for token in re.split(split_regex, string) if token != '' and token not in stopwords]
    return tokens
{% endhighlight %}

With this `tokenize` function, we can just count how many tokens the pair of documents shared in common. The more shared tokens they have, the more `similar`.

### **ER as Text similarity - Weighted Bag-of-Words using TF-IDF**

* **TF**: rewards tokens that appear many times in the same document. It's simply the frequency count, normalized by the length of the document. TF is an example of *local* weight
* **IDF**: rewards tokens that are rare overall in a dataset, because it carrys more specific information. IDF is an example of *global* weight.

The product of the two gives us the bag-of-words TF-IDF.

{% highlight python %}
def tf(tokens):
    dict = {}
    denom = len(tokens)
    for token in tokens:
        if token in dict:
            dict[token] += 1.0 / denom
        else:
            dict[token] = 1.0 / denom
    return dict
{% endhighlight %}

For idfs, need a corpus to process to get the *global* weights

{% highlight python %}
def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """
    N = corpus.count()
    uniqueTokens = corpus.flatMap(lambda (name, tokens): [(name, token) for token in tokens]).distinct()
    tokenCountPairTuple = uniqueTokens.map(lambda (name, token): (token, 1))
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a,b: a+b)
    return (tokenSumPairTuple.map(lambda (token, count): (token,  N / (count + 0.0))))
{% endhighlight %}

{% highlight python %}
def tfidf(tokens, idfs):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(tokens)
    tfIdfDict = {}
    for k in tfs.keys():
        tfIdfDict[k] = tfs[k] * idfs[k]
    return tfIdfDict
{% endhighlight %}

### **ER as Text Similarity - Cosine Similarity**

The geometric interpretation is more intuitive. The angle between two document vectors is small if they share many tokens in common, because they are pointing in roughly the same direction. For that case, the cosine of the angle will be large. Otherwise, if the angle is large (and they have few words in common), the cosine is small. Therefore, cosine similarity scales proportionally with our intuitive sense of similarity.

{% highlight python %}
def cosineSimilarity(string1, string2, idfsDictionary):
    """ Compute cosine similarity between two strings
    Args:
        string1 (str): first string
        string2 (str): second string
        idfsDictionary (dictionary): a dictionary of IDF values
    Returns:
        cossim: cosine similarity value
    """
    w1 = tfidf(tokenize(string1), idfsDictionary)
    w2 = tfidf(tokenize(string2), idfsDictionary)
    return cossim(w1, w2) # just dotprod / norm * norm

cossimAdobe = cosineSimilarity('Adobe Photoshop',
                               'Adobe Illustrator',
                               idfsSmallWeights)
{% endhighlight %}

One of the cool trick that was demonstrated in the lab was the usage of broadcast variable. The issue here was the idfs need to be passed down to each of the worker so perform the cosine similarity. A more efficient way is to define a broadcast variable in the master, so workers can just use it in their respective operations. 

{% highlight python %}
idfsSmallWeights = idfsSmall.collectAsMap() #idfsSmall is just the RDD for idf weights
idfsSmallBroadcast = sc.broadcast(idfsSmallWeights)
def computeSimilarityBroadcast(record):
    """ Compute similarity on a combination record, using Broadcast variable
    Args:
        record: a pair, (google record, amazon record)
    Returns:
        pair: a pair, (google URL, amazon ID, cosine similarity value)
    """
    googleRec = record[0]
    amazonRec = record[1]
    googleURL = googleRec[0]
    amazonID = amazonRec[0]
    googleValue = googleRec[1]
    amazonValue = amazonRec[1]
    cs = cosineSimilarity(googleValue, amazonValue, idfsSmallBroadcast.value) # Here is the key!
    return (googleURL, amazonID, cs)
{% endhighlight %}

### **Scalable ER**

* **First, we did a lot of redundant computation of tokens and weights, since each record was reprocessed every time it was compared**
  * **Solution**: The first source of quadratic overhead can be eliminated with precomputation and look-up tables. Again, using broadcast variables
* **Second, we made quadratically many token comparisons between records**
  * **Solution**: An forward index is a data structure that will allow us to avoid making quadratically many token comparisons.It maps each pair of document to the list of common tokens. This allows us to do more efficient dot products for each pair.

#### Compute IDFs and TF-IDFs for the full datasets and store them as broadcast variables

{% highlight python %}
fullCorpusRDD = amazonFullRecToToken.union(googleFullRecToToken)
idfsFull = idfs(fullCorpusRDD)
idfsFullCount = idfsFull.count()

# Recompute IDFs for full dataset
idfsFullWeights = idfsFull.collectAsMap()
idfsFullBroadcast = sc.broadcast(idfsFullWeights)

# Pre-compute TF-IDF weights.  Build mappings from record ID weight vector.
amazonWeightsRDD = amazonFullRecToToken.map(lambda (name, tokens): (name, tfidf(tokens, idfsFullBroadcast.value)))
googleWeightsRDD = googleFullRecToToken.map(lambda (name, tokens): (name, tfidf(tokens, idfsFullBroadcast.value)))

amazonNorms = amazonWeightsRDD.map(lambda (name, vector): (name, norm(vector)))
amazonNormsBroadcast = sc.broadcast(amazonNorms.collectAsMap())

googleNorms = googleWeightsRDD.map(lambda (name, vector): (name, norm(vector)))
googleNormsBroadcast = sc.broadcast(googleNorms.collectAsMap())
{% endhighlight %}

#### Create forward Index 
First, create a inverted index, which we will later reverse back to a forward index

{% highlight python %}
def invert(record):
    """ Invert (ID, tokens) to a list of (token, ID)
    Args:
        record: a pair, (ID, token vector)
    Returns:
        pairs: a list of pairs of token to ID
    """
    id = record[0]
    tokens = record[1]
    pairs = []
    for k in tokens.keys():
        pairs.append((k, id))
    return pairs
{% endhighlight %}

{% highlight python %}
amazonInvPairsRDD = (amazonWeightsRDD
                    .flatMap(invert)
                    .cache())

googleInvPairsRDD = (googleWeightsRDD
                    .flatMap(invert)
                    .cache())
{% endhighlight %}    

Reverse to get back (amazonId, googleID) -> list of common tokens

{% highlight python %}
def swap(record):
    """ Swap (token, (ID, URL)) to ((ID, URL), token)
    Args:
        record: a pair, (token, (ID, URL))
    Returns:
        pair: ((ID, URL), token)
    """
    token = record[0]
    keys = record[1]
    return (keys, token)
{% endhighlight %}    

{% highlight python %}
commonTokens = (amazonInvPairsRDD
                .join(googleInvPairsRDD)
                .map(swap)
                .groupByKey()
                .cache())  
{% endhighlight %}     

Putting everything together to do `fastCosineSimilarity` calculation

{% highlight python %}
amazonWeightsBroadcast = sc.broadcast(amazonWeightsRDD.collectAsMap())
googleWeightsBroadcast = sc.broadcast(googleWeightsRDD.collectAsMap())

def fastCosineSimilarity(record):
    """ Compute Cosine Similarity using Broadcast variables
    Args:
        record: ((ID, URL), token)
    Returns:
        pair: ((ID, URL), cosine similarity value)
    """
    amazonRec = record[0][0]
    googleRec = record[0][1]
    tokens = record[1]
    s = sum([amazonWeightsBroadcast.value[amazonRec][token] * googleWeightsBroadcast.value[googleRec][token] for token in tokens])
    value = s / (amazonNormsBroadcast.value[amazonRec] * googleNormsBroadcast.value[googleRec])
    key = (amazonRec, googleRec)
    return (key, value)

similaritiesFullRDD = (commonTokens
                       .map(fastCosineSimilarity)
                       .cache())
{% endhighlight %}   

### **Analysis on Model Performance**

ER algorithms are evaluated by the common metrics of information retrieval and search called precision, recall, and F-measures. See the code for more details.                      

## Lab: Recommendation, Matrix Factorization, Alternating Least Square

### **Lecture Highlights**

None

### **Lab Highlights**
This lab is relatively straightforward. The emphasis is on using Spark transformations and learning how to use the ALS algorithm to do matrix factorization.

#### Split into Training, validation, and Test sets

{% highlight python %}
trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)
{% endhighlight %}

#### Building a lower rank matrix using Matrix factorization

{% highlight python %}
from pyspark.mllib.recommendation import ALS

validationForPredictRDD = validationRDD.map(lambda (UserID, MovieID, rating): (UserID, MovieID))

seed = 5L
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.03

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD) # define in the code to calculate RMSE
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank
{% endhighlight %}

#### Always compare to the baseline model (predict average rating for all)

{% highlight python %}
trainingAvgRating = trainingRDD.map(lambda (UserID, MovieID, Rating): Rating).reduce(lambda a,b: a+b) / ((trainingRDD).count())
print 'The average rating for movies in the training set is %s' % trainingAvgRating

testForAvgRDD = testRDD.map(lambda (UserID, MovieID, Rating): (UserID, MovieID, trainingAvgRating))
testAvgRMSE = computeError(testForAvgRDD, testRDD)
print 'The RMSE on the average set is %s' % testAvgRMSE
{% endhighlight %}

The last part of the lab included another record for myself where it consists of all the movies that I watched and rated. 
* The row is added to the matrix
* We retrain the lower rank matrix factorization model
* With the model, can then predict my ratings on movies I haven't seen

## Lab: Linear Regression

### **Lecture Hightlights**

* Big Picture: Supervised Learning Pipeline
  
  * Obtain Raw Data
  * Split Data into Training, Validation, and Test sets
  * Feature Extraction
  * Training
  * Validation / Tuning Hyper-parameter
  * only until we are satisfied, evaluate on test set
  * Prediction

* Distributed ML: Computation & Storage

  * Computing closed form solution (i.e. normal equation)
    * Computation: O(nd^2 + d^3) operations
    * Storage: O(nd + d^2) floats
    * When n large, d small, the bottleneck is in storing X and computing X^TX. 
      * Solution: distribute them to clusters
      * Using outer product to compute X^TX
    * When n large, d large, we now have trouble storing and operating on X^TX
      * In big trouble -> motivation for Gradient Descent

  * Gradient Descent: I have enough intuition on the math
    * Parallel gradient descent
      * See slides to see why both storage and computation complexity got reduced

  * 1st rule of thumb: Computation and storage should be linear in (n, d)

  * 2nd rule of thumb: Perform parallel and in memory computation

  * 3rd rule of thumb: Minimize network communication
    * Stay local (model, data parallel)
    * Reduce iteration (mini-batching)

### **Lab Highlights**

In this lab, we get hands on with training a linear regression model in Spark. The task is to predict song release year based on 12 different audio features (I didn't check what they are), and the main emphases are on:

* **Part 1**: Read and parse the initial dataset
* **Part 2**: Create an Evaluate a baseline model
* **Part 3**: Train (via gradient descent) and evaluate a linear regression model
* **Part 4**: Train using MLlib and tune hyperparameters via grid search
* **Part 4**: Add interactions between features

The beauty of this exercise is that it:

* It does a great job to **modularize** the code, so every task is easy to carry out.
* Links to **documentations** are immediately relevant, so I get unblocked quickly.

This is the general analysis style that I should follow. Keep in mind that in real life, it's very easy to 1). implement everything in one big function, and 2) you might need to google for a long time before figuring out the right syntax to unblock yourself.

Let's talk about the code organization for each part

#### Part 1: Read and parse the initial dataset
Load in the text file as raw data
{% highlight python %}
rawData = sc.textFile(fileName, numPartitions)
{% endhighlight %}

create a method to parse the string into LabeledPoint, which is the data structure for differentiating label & features
{% highlight python %}
def parsePoint(line):
    """Converts a comma separated unicode string into a `LabeledPoint`.
    """
{% endhighlight %}

Break data into training, validation, and test set

{% highlight python %}
weights = [.8, .1, .1]
seed = 42
parsedTrainData, parsedValData, parsedTestData = parsedData.randomSplit(weights, seed)
parsedTrainData.cache()
parsedValData.cache()
parsedTestData.cache()
{% endhighlight %}

#### Part 2: Create and Evaluate a baseline model

Before we make predictions using fancy models, it's always a good idea to build a naive baseline model just so we can use it as the benchmark. Any additional efforts should improve our baseline performance.

The beauty here is how we break up the work, we create **getLabeledPrediction**, **squaredError**, **calcRMSE**:

{% highlight python %}
def getLabeledPrediction(weights, observation):
    return (observation.label, weights.dot(observation.features))
{% endhighlight %}

{% highlight python %}
def squaredError(label, prediction):
    return (label - prediction) ** 2
{% endhighlight %}

{% highlight python %}
def calcRMSE(labelsAndPreds):
    sumSquaredError = labelsAndPreds.map(lambda (label, prediction): squaredError(label, prediction)).sum()
    nCount = labelsAndPreds.count()
    return math.sqrt(sumSquaredError / nCount)
{% endhighlight %}

With these two functions, calculating the training, validation, and test error of the baseline model became easy

{% highlight python %}
labelsAndPredsTrain = parsedTrainData.map(lambda lp: (lp.label, averageTrainYear))
rmseTrainBase = calcRMSE(labelsAndPredsTrain)

labelsAndPredsVal = parsedValData.map(lambda lp: (lp.label, averageTrainYear))
rmseValBase = calcRMSE(labelsAndPredsVal)

labelsAndPredsTest = parsedTestData.map(lambda lp: (lp.label, averageTrainYear))
rmseTestBase = calcRMSE(labelsAndPredsTest)
{% endhighlight %}

#### Part 3: Train (via gradient descent) and evaluate a linear regression model 

The third step is to code up gradient descent for linear regression. Again, codes are modularize here

{% highlight python %}
def gradientSummand(weights, lp):
    summand = (weights.dot(lp.features) - lp.label) * lp.features
    return summand
{% endhighlight %}

Then we have the actual gradient descent algorithm

{% highlight python %}
def linregGradientDescent(trainData, numIters):
    # The length of the training data
    n = trainData.count()
    # The number of features in the training data
    d = len(trainData.take(1)[0].features)
    w = np.zeros(d)
    alpha = 1.0
    # We will compute and store the training error after each iteration
    errorTrain = np.zeros(numIters)
    for i in range(numIters):
        # Use getLabeledPrediction from (3b) with trainData to obtain an RDD of (label, prediction)
        # tuples.  Note that the weights all equal 0 for the first iteration, so the predictions will
        # have large errors to start.
        labelsAndPredsTrain = trainData.map(lambda lp: getLabeledPrediction(w, lp))
        errorTrain[i] = calcRMSE(labelsAndPredsTrain)

        # Calculate the `gradient`.  Make use of the `gradientSummand` function you wrote in (3a).
        # Note that `gradient` sould be a `DenseVector` of length `d`.
        gradient = trainData.map(lambda lp: gradientSummand(w, lp)).reduce(lambda x,y : x + y)

        # Update the weights
        alpha_i = alpha / (n * np.sqrt(i+1))
        w -= alpha_i * gradient
    return w, errorTrain
{% endhighlight %}

With this gradient descient algorithm implemented, we can now calculate the learned weights and training errors easily:

{% highlight python %}
numIters = 50
weightsLR0, errorTrainLR0 = linregGradientDescent(parsedTrainData, numIters)
labelsAndPreds = parsedValData.map(lambda lp: getLabeledPrediction(weightsLR0, lp))
rmseValLR0 = calcRMSE(labelsAndPreds)
{% endhighlight %}

#### Part 4: Train using MLlib and perform grid search
The training part is mostly boiler plate, so there is nothing that is too surprising. Once we have the model, we can also do prediction easily, everything in one suite:

{% highlight python %}
from pyspark.mllib.regression import LinearRegressionWithSGD
# Values to use when training the linear regression model
numIters = 500  # iterations
alpha = 1.0  # step
miniBatchFrac = 1.0  # miniBatchFraction
reg = 1e-1  # regParam
regType = 'l2'  # regType
useIntercept = True  # intercept

# build model
firstModel = LinearRegressionWithSGD.train(
                                     data = parsedTrainData, 
                                     iterations = numIters, 
                                     step = alpha, 
                                     miniBatchFraction = miniBatchFrac, 
                                     initialWeights = None, 
                                     regParam = reg, 
                                     regType = regType, 
                                     intercept = useIntercept)

# weightsLR1 stores the model weights; interceptLR1 stores the model intercept
weightsLR1 = firstModel.weights
interceptLR1 = firstModel.intercept
print weightsLR1, interceptLR1

samplePoint = parsedTrainData.take(1)[0]
samplePrediction = firstModel.predict(samplePoint.features)
print samplePrediction
{% endhighlight %}

The more interesting part is tuning hyper-parameter using grid-search, and it leverages all the modular function we wrote above:

{% highlight python %}
bestRMSE = rmseValLR1
bestRegParam = reg
bestModel = firstModel

numIters = 500
alpha = 1.0
miniBatchFrac = 1.0
for reg in [1e-10, 1e-5, 1]:
    model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                          miniBatchFrac, regParam=reg,
                                          regType='l2', intercept=True)
    labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
    rmseValGrid = calcRMSE(labelsAndPreds)
    print rmseValGrid

    if rmseValGrid < bestRMSE:
        bestRMSE = rmseValGrid
        bestRegParam = reg
        bestModel = model
rmseValLRGrid = bestRMSE
{% endhighlight %}

Another example of tuning hyper-parameter, alpha and numberIter at the same time

{% highlight python %}
reg = bestRegParam
modelRMSEs = []

for alpha in [1e-5, 10]:
    for numIters in [500, 5]:
        model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                              miniBatchFrac, regParam=reg,
                                              regType='l2', intercept=True)
        labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
        rmseVal = calcRMSE(labelsAndPreds)
        print 'alpha = {0:.0e}, numIters = {1}, RMSE = {2:.3f}'.format(alpha, numIters, rmseVal)
        modelRMSEs.append(rmseVal)
{% endhighlight %}

#### Part 5: Add interactions between features
Finally, we build a more involved model with quadratic features, so most of the functions we defined above are still usable. First, define a function to create twoWayInteraction features:

{% highlight python %}
import itertools
def twoWayInteractions(lp):
    """Creates a new `LabeledPoint` that includes two-way interactions.
{% endhighlight %}

Then we build a new dataset with the set of expanded features:

{% highlight python %}
# Transform the existing train, validation, and test sets to include two-way interactions.
trainDataInteract = parsedTrainData.map(lambda lp: twoWayInteractions(lp))
valDataInteract = parsedValData.map(lambda lp: twoWayInteractions(lp))
testDataInteract = parsedTestData.map(lambda lp: twoWayInteractions(lp))
{% endhighlight %}

Training and Testing are exactly the same as above, so I will not repeat myself.

## Lab: Click Prediction, Logistic Regression

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
    * Better approaches include (One-Hot-Encoding) & (Feature Hashing). Check out [Feature Hashing] Wikipedia page

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

## Lab: Neural Science, Distributed PCA

### **Lecture Highlights**

#### **PCA recipe**
* Step 1: Center Data
* Step 2: Compute Covariance Matrix
* Step 3: Eigen-decomposition
* Step 4: Compute PCA scores

#### **Distributed PCA**

* Case 1: Big _n_ and small _d_
  * O(d^2) local storage, O(d^3) local computation, O(dk) communication <- this is what our lab does
  * Can use eigen-decomposition routine directly

* Case 2: Big _n_ and big _d_
  * O(dk + n) local storage, computation
  * O(dk + n) communication
  * Iterative Algorithm


#### **[Big _n_ and small _d_]**

* **Step 1: Center Data**
  * Compute d-dimensional vector **m**, where each element is the mean of a column
  * Communicate **m** to all workers
  * Subtract **m** from each data point

* **Step 2: Compute Covariance Matrix**
  * compute this matrix via **Outer Product**
  * O(nd) distributed storage, need O(d^2) local storage for each outer product. O(nd^2) distributed computation
  * O(d^2) local storage and computation for covariance matrix

* **Step 3: Eigen-decomposition**
  * Same as above, but require O(dk) communication of the k eigenvector, each d-dimensional
  * Local computation of eigen-decomposition is **O(d^3)**

* **Step 4: Compute PCA scores**
  * Multiply each point by principal components, P
  * O(dk) computation. Each element is dot product on d, and k dimension

#### **Big _n_ and big _d_**

* The problem is that it's computational intensive to calculate the X^T * X, and eigen-decomposition also is too computationally intensive O(d^3)
* Need to figure out a way to compute eigenvectors without resorting to eigen-decomposition
* Use [Power Iteration]
* See slides more storage/computation requirements



[Big _n_ and small _d_]:https://courses.edx.org/c4x/BerkeleyX/CS190.1x/asset/CS190.1x_week5.pdf
[Big _n_ and big _d_]:https://courses.edx.org/c4x/BerkeleyX/CS190.1x/asset/CS190.1x_week5.pdf

[Power Iteration]: http://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf

### **Lab Highlights**

#### Standard PCA implementation

{% highlight ruby %}
def estimateCovariance(data):
{% endhighlight %}

{% highlight ruby %}
def pca(data, k=2):
{% endhighlight %}

{% highlight ruby %}
def varianceExplained(data, k=1):
{% endhighlight %}

#### Neuroscience & PCA analysis

* Data format is ((coord1, coord2), timeSeries)
* Some basic massaging of data. Normalization...etc
* Feature-based aggregation and PCA is eye-opening
  * Learned about numpy `np.tile`, `np.eye`, `np.ones`
  * Learned about Kronecker product
  * Learned how we can aggregate feature by left multiply these specific matrices from above
  * Did some exercise to create these aggregation matrices using the basic `np.*` mentioned above
* PCA with aggregating feature by time
  * Aggregate feature
  * Apply PCA
  * Plot
* PCA with aggregating feature by direction
  * Aggregate feature
  * Apply PCA
  * Plot
[Feature Hashing]: https://en.wikipedia.org/wiki/Feature_hashing