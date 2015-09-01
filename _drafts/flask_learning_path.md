---
layout: post
title: Flask Learning Path
---
A very simple guide to kick start learning Flask 

### **Intro & Motivation**
One of my 2015 Q1 goals is to learn how to build interactive web application for visualization. The purpose is to extend my skillsets and make my analyses more expressive, so the 'insights' are not confined on paper or powerpoint presentation. 


### **Flask**
In the previous quarters, I have spent some time learning the foundation of building web application; I've learned d3.js; and I've encountered several different web frameworks. There are many choices -- Ruby on Rails, Django, flask...etc. Amongst these tools, I find that Flask is very appealing because:

* It's built on top of Python (so I can practice more)
* It's light-weight, easy to learn
* It seems to be the de-facto tool for Insights Data Science and Data Engineering Programs
* There are comprehensive learning materials from Udacity as well as Twitter University
* The combination of flask + d3.js also seem common

As a result, I decided to embark my journey in learning flask. In the following sections, I will highlight some of the learning materials, paving the path of how one should study flask, without talking about the specifics. If you want to be motiviated, you can also read this [quora post]

Flask is an python microframework for web development. Check out this [Intro]. There are a handful of basic concepts you need to master:

#### **Creating an App**
* **Creating a app**: Creating an app is as easy as the following 7 lines of code

{% highlight python %}
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()
{% endhighlight %}

#### **Routes**
* The key construct for surfacing requests with response from users is the view + route combo
* Each `@app.route("path")` is a decorator. When a user enters a particular url, the decorator will respond to the request by executing the function that is wrap within it. To read more about decorator, check out Simeon Franklin's awesome [decorator tutorial]
* Each `def function()` will be executed when the url in the route is being requested. These functions are sometimes called `views` in the MVC framework
* The key principle here is to keep the logic of the view functions as simple as possible. 

#### **Templates**
* Each view function will return resources to be displayed in the UI (e.g. A HTML page), but we don't want to hardcode HTML in the function logic.
* As a result, using templates can be super helpful! Essentially, we create a `template` directory, and we will separate out all the HTML code from the view functions to their own HTML pages.
* We can pass in variables into a template using `{{ }}`, and we can use control flows in a teamplate using `{}`. There are only a handful of control flows that we need to learn.
* `Template inheritance` further simplifies our lives. Often, we would have the same `base.html` that we want to show throughout our web application, having the response of the view function to be embedded in its parent HTML is a useful construct:
	* The way to achieve inheritance is using `{block}` or `{include}` statements

#### **Static Resources**
* Sometimes we would like to serve up static resources for each and every web page (e.g. css + logos). 
* Creating a `static` directory and put every static files there, flask will serve up these resources for us.
* This is where Bootstrap can really help. Download Bootstrap and put everything under static dir, and you will get a properly styled application. Check out Simeon Franklin's Twitter University class (database one).

#### **CRUD & Databases**
* For a sophisticated web application, we want to be able to read/write data from/to a persistent database.
* Similar to the concern of separation of view function logic with HTML, we do not want to hardcode our SQL statements into the function definition.
* Abstraction like SQLAlchemy can really help us there. The main advantage is ORM (Object Relation Mapping), where each python class corresponds to a table, and each instanciation of a class correspond to a row to the table.
* CRUD oeprations correspond really well to the HTTP actions
	* Create -> CREATE
	* Read -> GET
	* Update -> POST
	* Delete -> DELETE
* Check out Twitter University Database + Flask Class

#### **Creating API Endpoints**
* Sometimes we do not want to return a HTML page as the response
* Sotmiems we want to return data that can be parsed by other computers easily. JSON is a very popular choice
* The act of sending data in Json format via HTTP requests = Creating RESTful API
* Check out [Designing a RESTful API with Python and Flask]

#### **Flask + d3.js**
* Once you create a lightweight web application, don't forget the whole point is to integrate it with cool visualization in d3. Here are some examples
	* [Example 1]
	* [Example 2]
	* [Example 3]

### **Other Materials**

* Udacity & Twitter University teaches the very basic, but there seems to be a lot more beyond templates, static resources, and databases, such as login, search, Ajax, internationalization/localization...etc. You can learn more from [The Flask Mega-Tutorial].  
* **Deploying our web application**:
* Twitter University's Deploy Flask at Twitter is a great video, which tells you how to do it the Twitter way, and it teaches things like `pants`, `pex`, `packer`, `aurora`, and `mesos`.
	* You use `pants` to manage dependencies and build a `.pex` file
	* A `.pex` file is just an executable that package all the things you need to run the web application. It's self contained and it has all the libraries and dependencies you need
	* `packer` is just a version control tool that allows you to version your executables.
	* `Aurora` is the client which allows you to schedule jobs on the Twitter internal Cloud
	* `Mesos` is essentially the Twitter version of AWS that allows you to take arbitrary resources and schedule production jobs
* For General deployment, check out [Intro] or [Full Stack Python]

[quora post]: http://www.quora.com/Should-I-learn-Flask-or-Django
[Intro]: http://nbviewer.ipython.org/github/jackgolding/FullStackDataAnalysis/blob/master/Web%20Development%20with%20Flask.ipynb
[decorator tutorial]: http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/
[Designing a RESTful API with Python and Flask]: http://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
[Full Stack Python]: http://www.fullstackpython.com/
[The Flask Mega-Tutorial]: http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
[Example 1]: http://flask.theoryandpractice.org/
[Example 2]: https://realpython.com/blog/python/web-development-with-flask-fetching-data-with-requests/
[Example 3]: http://adilmoujahid.com/posts/2015/01/interactive-data-visualization-d3-dc-python-mongodb/
