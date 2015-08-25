---
layout: post
title:  "EdX: Intro to Big Data with Spark Week 3 Notes"
date:   2015-08-15 14:00:00
comments: True
---
A summary of Spark RDD operations: collect, groupByKey, reduceByKey, sortByKey, zipWithIndex, takeOrdered, join ...etc

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
