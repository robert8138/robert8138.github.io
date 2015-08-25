---
layout: post
title:  "EdX: Scalable Machine Learning Week 5 Notes"
date:   2015-08-08 15:00:00
comments: True
---
Learn distributed PCA. Learn numpy np.tile, np.eye, np.ones, and feature aggregation

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