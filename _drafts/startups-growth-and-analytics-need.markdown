---
layout: post
---
Right out of graduate school, I naturally wanted to do Machine Learning and apply the most sophisticated models possible to do "high impact" work. My very first job was titled "Analytics Engineer" at Washington Post Labs, a small start-up funded by the Posts consists of 40-50 people where the product is a consumption based, news app with social integrations. At the time, there are three people on the analytics team.

* Johnathon: He is the guru in maintaining reporting ETL, and anything user facing
* Olga: She is the guru in data modeling, event tracking 
* Jay: He is the guru in processing data at scale, using Hadoop

As for me, I was just an eager young mind who wanted to do modeling. In retrospective, I had no clear idea the analytics need of the organization, so there was somewhat of an interest misalignment. It is only over time that I realized that companies in different stages have different analytics needs! Twitter, Facebook, Uber, all had a humble start. The Analytics team did not start off by building some fancy ML algorithms, there are more foundational work that needs to be done such as defining KPIs, ETL, data modeling, and reporting. Optimizations only comes after a certain level of maturity is reached.  

This makes me wonder, what exactly is the role of analytics in various stages of a company's growth? How does the emphases change over time, and most importantly, how can a DS use this knowledge to align their interests with the need of the organization?

I started look into this, and I realized that in order to understand the analytics need, I need to understand the cycles of a growing start-ups. Recently, I ran into several posts that really helped me to understand this better.

## For those just started up
Marc Andressen's [essay] is really great, as it explains for a company that just started (literally a start-up), what is the single most important goal/objective they need to achieve. His thesis is basically te following:

> The only thing that matters is getting to product/market fit. Product/market fit means being in a good market with a product that can satisfy that market.

He started off by discussing the three criteria VC use when evaluating a start-up:

* The caliber of a startup team
* The quality of a startup's product
* The size of a startup's market

Which one is the most important? Andreessen argued that most VC would say team is the most important, and people matters most. He took a different stand, and assert that market is the most important.

> In a great market -- a market with lots of real potential customers -- the market pulls product out of the startup. Conversely, in a terrible market, you can have the best product in the world and an absolutely killer team, and it doesn't matter -- you're going to fail. The #1 company-killer is lack of market

He also mentioned that you can always tell if product/market fit is happening.

> You can always feel when product/market fit isn't happening. The customers aren't quite getting value out of the product, word of mouth isn't spreading, usage isn't growing that fast, press reviews are kind of "blah", the sales cycle takes too long, and lots of deals never close.

> And you can always feel product/market fit when it's happening. The customers are buying the product just as fast as you can make it -- or usage is growing just as fast as you can add more servers. Money from customers is piling up in your company checking account. You're hiring sales and customer support staff as fast as you can. Reporters are calling because they've heard about your hot new thing and they want to talk to you about it. You start getting entrepreneur of the year awards from Harvard Business School. Investment bankers are staking out your house. 

Ok, so how is this related to our topic today? If you accept the premise that the single most important objective of a start-up is to find product/market fit, then you need to do wathever is required to acheive this, according to Andreeseen, this means:

> Including changing out people, rewriting your product, moving into a different market, telling customers no when you don't want to, telling customers yes when you don't want to, raising that fourth round of highly dilutive venture capital -- whatever is required.

Whatever is required, does it necessarily include Analytics? I don't know the answer, due to my lack of start-up experience. Here are a few directions which can shed some more lights:

* Look at start-ups that are looking for cofounders or early enigneers. Do they need to hire an Analytics Engineer?
* At what stage does a company need to start building an Analytics team?
* How did companies like Facebook, Twitter, Uber, Airbnb build out their very first Analytics team? What were the motivations.

## Post Product/Market Fit - Growth

Looking back at my two years spent at Twitter's Growth team, I recently realized that I actually never study more in depth the function of a Growth team, how they related to the larger organization. Recently, I came across Josh Schwarzapel's medium post titled "[How to Start a Growth Team]", and I thought it was so interesting. 

While the purpose of the blog post might be to talk about how to start a Growth team, he mentioned quite a few distinct characteristics about a Growth team that I can really echo with.

### Is your product ready for Growth?

> Growth Teams expand the value of, and customer base for a product that is already working through rapid, data-informed experimentation. The corollary to this definition is that you should not start working on growth unless your product has product/market fit.

In Twitter's case, the Growth team has its incarnation in various form throughout the years. By the time I joined Twitter, the particular Growth team that I was on is probably Growth Team in v.3/v.4. Twitter as a product was will beyond product/market fit, and the whole org felt a bit less like a cross-functional Growth than an org with specific product areas.

I did learned about AARRR (Acquisition, Activation, Retention, Revenue, Referral), and each team is somewhat responsible for one of those metrics:

* Acquisition: Signup Engineering, to focus on the top of the funnel
* Activation: START, to focus on New User Experimence (NUX)
* Retention: Notifications, which focus on sending more emails, push, and SMS
* Referrals: Virality, this team was responsible for invites but was later killed

That said, it really was clear to me that each team is faithfully optimizing their respective metric though. In addition, there was no notion of Growth Accounting.

### Who will be your executive champion?

> Here is why getting an executive champion is so important: You're going to work on someone else's product. The product you’re planning to grow already has a product and engineering team, a codebase, a team culture, and often a set of priorities that have nothing to do with your own. You are going to impact all of those things when you insert yourself into their process.

In Twitter's case, the executive champion was Dick. There's often confusion on why Growth is working on a particular product area, and people always complain about ownership. When the team is trying out ideas, iterate fast, learn insights, and improve upon them, people will give usually give benefits of the doubt, especially when a big exec is rallying for the team. Conversely, things can turn bad pretty quickly if the team doesn't learn fast enough or experiment enough or be bold enough.

In our case, the decline of the Growth team was pretty self evident when several teams started to take ownership of Growth initiated project, Instant Timeline is a good example. We also see notification infrastructure team got merged into C&D, largely because those require more sophisticated optimization skill-sets like Machine Learning, which Growth does have.

I also think that the Growth team needs to be willing to give up the projects they've invested heavily upfront, and be willing to extend the project's life cycle by handing it to teams that can execute better.

> You are going to try controversial things

That's another good point. Growth requires experimentation, and experimentation sometimes means trying out new and controversial things. Being able to take intelligent risks, learn quickly, and iterate seem to be an important characteristics of a good Growth team.

### How should you structure your team?

> First, the easy part: a Growth team is just a product team. That means you need awesome engineers, designers, and PMs. One trap I frequently hear people fall into is that they empower a PM to work on growth initiatives with no design or engineering team. The idea is that this person will learn about growth and push for it to get on other teams’ roadmaps. 

I think this certainly true. You cannot have a product without Engineers and Designers.

> In terms of traits, Growth Team members are data-driven and aggressive, with an appetite for risk. People who are open-minded, willing to accept other possibilities. People who are skeptical of what is accepted as "Truth", and quickly learn new things.

Basically people who are data driven

### How should you spend your first 30 days

* Focus the team around one metric
* Generate a few quick, high-impact wins and evangelize them
* Start investing deeply in the tools for analytics & experimentation!

I can't agree more with the last point, I think with proper investment in Analytics & Experimentation, our Growth team could have function more effectively.

### Obviously this post is still very dis-organized, but the eventual goal is to summarize what are the analytics need for the company at different stages, with a personal spin to it.

[essay]: http://web.stanford.edu/class/ee204/ProductMarketFit.html
[How to Start a Growth Team]: https://medium.com/android-news/how-to-start-a-growth-team-ff70cd29c0f2
