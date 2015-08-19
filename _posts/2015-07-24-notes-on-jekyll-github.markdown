---
layout: post
title:  "Resources for builing your own website using Jekyll and Github"
date:   2015-07-24 10:27:08
comments: True
---

# **Resources**

* To build a Jekyll project, simply follow the instruction on [jekyllrb.com]

The basic commands are relatively simple

{% highlight python %}
gem install jekyll
jekyll new my-new-awesome-website
cd my-new-awesome-website
jekyll serve --watch
{% endhighlight %}


* The awesome thing about Jekyll is that a lot of people already pre-built different themes that you can use directly. Here are a few examples

  * [Artist's Theme] from Devtips. This one is pretty fancy, and has interactivity, and is more tailored for designers
  * [Poole] Very minimalistic look
  * [Jekyll Bootstrap Themes] Various Twitter Bootstrap themes


* I also found the following [YouTube] video to be helpful to get started. Allegedly, this is the most viewed Jekyll/Github video I can find on YouTube on this topic.


* You can totally just copy the jekyll directory into your githubUserName.github.io, push the changes, and the site would totally work under this domain. Here is [mine] as an example.

# **TODO**
* Add Quora, Medium signs
* Add presentation from cdk
* Add my own domain
* Add better styling, using Artist themes stuff
* Add Google Analytics Tracking
* Add Comments

## **Set up a custom domain for github pages**

It's a bit confusing what is the right steps to take, so here they are:

### Set up a domain from goDaddy

First, if you don't already have a domain, you should go to [godaddy.com]  to set up one. It will cost a small amount of money to acquire and own a domain. 

![Godaddy](/images/godaddy.png)

### Create a CNAME file in your repo
This is pretty straightforward. In the CNAME (no extension) file, simply list out your website name. 

### Set up cloudfare for redirects

Cloudfare is free, and it allows you to set up redirects. I think the basic redirect flows happens in the following way: ihsiangchang.com -> github's global content delivery network IP -> CNAME -> sees the name of the website -> robert8138.github.io

The important thing in this step is that you need to:

* Set up two type A (?) which links your website to Github's IPs
* Set up CNAME type with name=www and value=robert8138.github.io
* Set up new Nameservers using CloudFlare's nameservers!

![cloudfare](/images/cloudfare.png)

### Update Nameservers
Once you have the new nameserver, you need to go back to godaddy and update them accordingly.

### That's it

After a few minutes up to 48 hours, you should see your domain works like a charm!

[jekyllrb.com]: http://jekyllrb.com/
[Artist's Theme]: https://github.com/DevTips/Artists-Theme
[Poole]: https://github.com/poole/poole
[Jekyll Bootstrap Themes]: http://themes.jekyllbootstrap.com/
[YouTube]: https://www.youtube.com/watch?v=O7NBEFmA7yA
[mine]: https://github.com/robert8138/robert8138.github.io
[godaddy.com]: https://www.godaddy.com
