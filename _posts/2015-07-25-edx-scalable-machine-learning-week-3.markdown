---
layout: post
title:  "EdX: Scalable Machine Learning Week 3 Notes"
date:   2015-07-25 15:00:00
---

### Lecture Summary

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

### Lab 3 Summary

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
parsedTrainData, parsedValData, parsedTestData = parsedData.<FILL IN>
parsedTrainData.<FILL IN>
parsedValData.<FILL IN>
parsedTestData.<FILL IN>

{% endhighlight %}

#### Part 2: Create and Evaluate a baseline model

Before we make predictions using fancy models, it's always a good idea to build a naive baseline model just so we can use it as the benchmark. Any additional efforts should improve our baseline performance.

The beauty here is how we break up the work, we create **getLabeledPrediction**, **squaredError**, **calcRMSE**:

{% highlight python %}
def getLabeledPrediction(weights, observation):
    """Calculates predictions and returns a (label, prediction) tuple.
    """
{% endhighlight %}

{% highlight python %}
def squaredError(label, prediction):
    """Calculates the the squared error for a single prediction.
    """
{% endhighlight %}

{% highlight python %}
def calcRMSE(labelsAndPreds):
    """Calculates the root mean squared error for an `RDD` of (label, prediction) tuples.
    """
{% endhighlight %}

With these two functions, calculating the training, validation, and test error of the baseline model became easy

{% highlight python %}
labelsAndPredsTrain = parsedTrainData.<FILL IN>
rmseTrainBase = <FILL IN>

labelsAndPredsVal = parsedValData.<FILL IN>
rmseValBase = <FILL IN>

labelsAndPredsTest = parsedTestData.<FILL IN>
rmseTestBase = <FILL IN>
{% endhighlight %}

#### Part 3: Train (via gradient descent) and evaluate a linear regression model 

The third step is to code up gradient descent for linear regression. Again, codes are modularize here

{% highlight python %}
def gradientSummand(weights, lp):
    """Calculates the gradient summand for a given weight and `LabeledPoint`.
    """
{% endhighlight %}

Then we have the actual gradient descent algorithm

{% highlight python %}
def linregGradientDescent(trainData, numIters):
    """Calculates the weights and error for a linear regression model trained with gradient descent.
    """
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
        labelsAndPredsTrain = trainData.<FILL IN>
        errorTrain[i] = calcRMSE(labelsAndPredsTrain)

        # Calculate the `gradient`.  Make use of the `gradientSummand` function you wrote in (3a).
        # Note that `gradient` sould be a `DenseVector` of length `d`.
        gradient = <FILL IN> using gradientSummand

        # Update the weights
        alpha_i = alpha / (n * np.sqrt(i+1))
        w -= <FILL IN>
    return w, errorTrain
{% endhighlight %}

With this gradient descient algorithm implemented, we can now calculate the learned weights and training errors easily:

{% highlight python %}
numIters = 50
weightsLR0, errorTrainLR0 = linregGradientDescent(<FILL IN>)
labelsAndPreds = parsedValData.<FILL IN>
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
firstModel = LinearRegressionWithSGD.<FILL IN>

# weightsLR1 stores the model weights; interceptLR1 stores the model intercept
weightsLR1 = <FILL IN>
interceptLR1 = <FILL IN>
print weightsLR1, interceptLR1

samplePoint = parsedTrainData.take(1)[0]
samplePrediction = <FILL IN>
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
for reg in <FILL IN>:
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

for alpha in <FILL IN>:
    for numIters in <FILL IN>:
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
trainDataInteract = <FILL IN>
valDataInteract = <FILL IN>
testDataInteract = <FILL IN>
{% endhighlight %}

Training and Testing are exactly the same as above, so I will not repeat myself.
