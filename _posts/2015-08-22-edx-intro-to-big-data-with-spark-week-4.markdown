---
layout: post
title:  "EdX: Intro to Big Data with Spark Week 4 Notes"
date:   2015-08-22 16:00:00
comments: True
---
Entity Resolution, Bag of Words model, Bag of words with TFIDF weights, cosine similarity, Scalable ER using forward index

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