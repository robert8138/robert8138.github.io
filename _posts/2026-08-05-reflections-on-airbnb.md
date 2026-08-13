---
layout: post
title:  "Reflections on Airbnb"
date:   2026-08-05 09:00:00
comments: True
ref: reflections-on-airbnb
excerpt: "I left Airbnb a few weeks ago, a decade after joining in early 2016. This post is part about what I think makes Airbnb unique, and part a recollection of lessons learned over the years."
---

## Introduction

I left Airbnb a few weeks ago, a decade after joining in early 2016.

Taking inspiration from [Reflections on Palantir](https://nabeelqu.substack.com/p/reflections-on-palantir) and [Reflections on OpenAI](https://calv.info/openai-reflections), I want to write down some reflections on my time there while the memory is still fresh. This post is part about what I think makes Airbnb unique, and part a recollection of lessons learned over the years. All in all, I left the company with a lot of gratitude, and that's the spirit I'm writing this in.

## Company

### Business

When I joined in early 2016, the company was in full growth mode, and I spent my first few years growing the host side of the marketplace. That's where I learned that running a two-sided marketplace is mostly about designing incentives and rules that don't overly favor one side or the other. Take [Instant Book](https://www.airbnb.com/help/article/523), where guests can book without host approval. It's convenient for guests, and it also means hosts have less control over who stays in their home. Or [House Rules](https://www.airbnb.com/help/article/472), useful for setting expectations. Pile up too many of them and you've created a hassle for guests. Growing a marketplace means the market design matters just as much as the growth curve.

After several years of high growth, Airbnb [announced](https://news.airbnb.com/airbnb-announces-intention-to-become-a-publicly-traded-company-during-2020/) its intention to go public in late 2019. Then COVID hit: by April 2020, gross nights booked were down 72% year over year, and net bookings turned negative, an unprecedented situation, as cancellations skyrocketed. After raising capital and cutting costs everywhere else, Airbnb made the hard decision to lay off about 25% of its workforce, roughly 1,900 people, which remains the largest layoff in Airbnb's history.

As the world slowly adapted to life with COVID, business recovery came faster than it did for most of our peers, as travel demand shifted toward domestic trips and rural destinations. Having supply across [100,000 cities and towns and 220 countries and regions](https://www.sec.gov/Archives/edgar/data/1559720/000155972022000006/abnb-20211231.htm) meant that wherever demand reappeared, we already had listings there. In a miraculous turn of events, Airbnb went public that December, ringing the bell with hosts around the world. After the year everyone had just been through, it was an emotional moment.

Airbnb has spent the years since perfecting the core product, and it's now expanding beyond it. Services, Experiences, and a push into boutique hotels are among the newest bets. It's also exploring what AI can do for the product and the business. Most of these are early and unproven, but a new chapter has begun.

### Culture

For a company whose business is predicated on 99% of humans being good (after all, you're staying at a stranger's home), Airbnb looks for many of those same qualities in hiring. Beyond the standard interview process, Airbnb runs culture interviews to assess a candidate's fit with its [core values](https://careers.airbnb.com/life-at-airbnb/). In mine, I talked about a trip I'd helped plan to Peru. I was told later that the stories I shared embodied the core value of "Be a Host."

It's no surprise, then, that my favorite core value is "Be a Host," and it shows up in small but telling ways. On my first day at Airbnb's HQ, people were opening and holding doors for me and for each other. That kind of behavior extended beyond small daily etiquette into how people actually worked together. My coworkers were brilliant, and brilliant assholes were rare. People were competitive, rarely combative. It was, on the whole, a place of low drama, where people treated each other with a lot of respect.

On the flip side, this desire to be a great host can sometimes get overextended into playing nice, or even conflict avoidance. This shows up often in consensus building, which can be a slow and sometimes painful process. It often comes as a shock to those used to a culture of direct, blunt feedback.

I do think the cofounders [don't want to fuck up the culture](https://medium.com/@bchesky/dont-fuck-up-the-culture-597cde9ee9d4), and they've tried to hold onto the early spirit every time the company outgrows itself. That said, it's hard not to miss some of the early days, when the company was smaller and less formal. Every Friday we had nerds@, where engineers shared what they'd learned or built that week. At world@, product leads frequently showed early snippets of what we were about to launch. Some of that faded as we grew, and I wish we'd kept more of it.

### Operating Model

Given that Brian Chesky, co-founder and CEO, is a designer by training, it's no surprise that he applies that same design mindset to building the company itself, and he's never shy about learning from iconic companies.

On the fourth floor of Airbnb HQ, frames hang on the walls describing the host and guest journey, inspired by how Walt Disney designed the storyline of Snow White. In the years leading up to COVID, Brian drew heavily on Amazon, with the ambition to grow Airbnb into a Trip platform that went beyond just accommodation. More recently, he's turned to Apple, reshaping how we do product marketing, launches, and roadmap planning. Each era brought its own ambitions and its own way of working, often under new leadership too.

Every change, though, came with its own growing pains. Pre-COVID, that meant unfocused sprawl: Airbnb was simultaneously pursuing experiences, magazine, business travel, hotels, and even flights, spread across four largely independent business units. Post-COVID, the pendulum swung toward bi-annual, big-bang releases: high-stakes, all-or-nothing launches that made it harder to isolate what was working. And Brian's [founder mode](https://paulgraham.com/foundermode.html) talk spun up its own debates, internally and externally. Despite this thrash, I appreciate that Brian has the ambition to build an iconic company and isn't afraid to adapt and experiment, treating the company itself as an iterative design problem.

One more operating model choice worth mentioning: Airbnb is one of the few companies that genuinely committed to remote-first since COVID. I've benefited from the flexibility, but I miss standing at a whiteboard with a coworker and working a problem out. Keeping that kind of collaboration alive is a hard problem. We've hired strong people in places we couldn't have reached otherwise, though it's harder to screen for candidates who want the job rather than the flexibility. The remote-versus-office question is challenging, and Airbnb is still working hard to find the right balance.

## Data at Airbnb

### Teams

In the early days, the [entire data team](https://medium.com/airbnb-engineering/at-airbnb-data-science-belongs-everywhere-917250c6beba), known internally as the A-team, was small enough to fit in a single room. It skewed toward early-career PhDs from a mix of backgrounds: economics, statistics, operations research, and the social sciences. All of them were sharp thinkers, deeply data-driven, and product-focused.

Organizationally, individual contributors owned specific domains and became experts in them, but they all reported up to a Head of Data. In more recent years, they report to various leads in engineering in a decentralized fashion. By the time I left, Airbnb had hired its first VP of Data Science, so perhaps the pendulum will swing back toward a more centralized data organization. These swings are not uncommon: LinkedIn and Facebook went through similar evolutions.

Airbnb's relationship with Data Engineering has been a complicated one. The investment started early: many of the founding data engineers came from Facebook, and our early warehouse showed it. The [medallion architecture](https://medium.com/airbnb-engineering/data-infrastructure-at-airbnb-8adfb34f169c) and [core data](https://medium.com/data-science/an-island-of-truth-practical-data-advice-from-facebook-and-airbnb-a0d9c355e5a0) were heavily inspired by what they'd built there. Around 2018, the org dismantled the Data Engineering team, and in my opinion it was one of the costliest mistakes data leadership at Airbnb made.

Luckily, we retained some of the strongest engineers, many of whom moved to Data Platform. Under new leadership, Airbnb reinvested in data engineering hiring in 2019, and the community is strong again today. Airbnb was also one of the first companies at this scale to create an Analytics Engineering organization, thanks in part to its investment in tooling like Minerva, our semantic layer.

Overall, the roles are increasingly specialized: data scientists and analysts focus on product work, analytics and data engineers build company-wide datasets, and software engineers build the underlying platform.

### Platform

Airbnb historically leaned "build" over "buy," and several successful open-source projects were born here, most notably Airflow and Superset.

For offline data, everything lives in a lakehouse: data on S3, stored as Parquet files in [Iceberg](https://www.youtube.com/watch?v=BP9wUnq_OLI) tables. Data is typically date-partitioned, and we compute incrementally wherever possible, though some late-arriving data (think cancellations or alterations) forces a full history rewrite. Spark is the main engine for batch, Flink for streaming, and Trino for interactive queries. For orchestration, Airbnb runs one of the largest Airflow deployments in the world, often at a scale the open source community isn't equipped for.

Airbnb leans heavily on config-driven frameworks, so much so that some joke that our data contracts are entirely built on fragile YAML files. The frameworks built this way are still widely adopted: ML feature platform [Chronon](https://medium.com/airbnb-engineering/chronon-a-declarative-feature-engineering-framework-b7b8ce796e04), Minerva as our semantic layer, and an [experimentation platform](https://medium.com/airbnb-engineering/how-airbnb-safeguards-changes-in-production-9fc9024f3446) called ERF, to name a few. Python is the primary language for building these data frameworks.

We invest just as heavily on the consumption side. [Dataportal](https://medium.com/airbnb-engineering/democratizing-data-at-airbnb-852d76c51770) is a catalog and UI that helps people find the right data, and a unified metadata service sits underneath it, storing the metadata every other data tool depends on, such as ownership, landing times, and asset tagging. More recently we built an internal data agent, and it has shown real promise, largely because the semantic layer and metadata service were already there for it to stand on.

I'm biased here. I think Airbnb's data ecosystem is sophisticated and underrated compared to peer companies. Investment in data was one of the reasons I joined, and it held up.

### Semantic Layer

For seven of my ten years at Airbnb, I worked on the company's semantic layer, [Minerva](https://medium.com/airbnb-engineering/how-airbnb-achieved-metric-consistency-at-scale-f23cc53dea70). People at other companies often ask how we scaled it across the entire organization. There wasn't one standout strategy. It came down to aligning with company initiatives, finding champions, and a relentless ownership mindset.

As early as mid-2018, we were already thinking about consolidating definitions across business metrics and experimentation metrics under a single source of truth. The real catalyst came around 2019, as Airbnb prepared to go public and data quality, or the lack of it, became an existential problem. Years earlier, we'd made the mistake of dismantling our Data Engineering team, and we'd been paying for it ever since. Different teams built their own versions of "bookings," "active listings," and "revenue." When Brian asked for last week's bookings number, he'd get a different answer depending on who he asked. For a company about to report to public markets, that was untenable. Our CTO was so concerned he declared "data bankruptcy."

Out of that urgency came three efforts. First, a [company-wide data quality initiative](https://medium.com/airbnb-engineering/data-quality-at-airbnb-e582465f3ef7) to rebuild the most business-critical data models from the ground up. Rebuilding models once wasn't enough to keep them trustworthy, so we created a [certification process](https://medium.com/airbnb-engineering/data-quality-at-airbnb-870d03080469) called MIDAS to hold data to a consistent bar of quality. Finally, we invested in infrastructure: a way for data producers to define a single source of truth for business metrics and dimensions that could actually be certified. [Minerva](https://medium.com/airbnb-engineering/how-airbnb-achieved-metric-consistency-at-scale-f23cc53dea70) became the natural home for that.

IPO-readiness created the urgency. MIDAS gave us the program and process to act on it, and Minerva ended up being the paved path that came out of it. We worked closely with key leaders early on to position it that way, then expanded team by team as wins built momentum. It took about two years before Minerva became the standard tool for Analytics. In some ways, the analytics engineering role at Airbnb exists because we had this central piece of technology sitting in the middle.

## Work Lessons

### On Building Software

Working at the intersection of software and data engineering, I got to learn from several exceptional software and data engineers who taught me how to think about the craft.

My first big takeaway is that "software engineering is programming integrated over time," a definition popularized by the book [Software Engineering at Google](https://abseil.io/resources/swe-book). Design patterns and abstractions are tools for managing complexity, more than ends in themselves. When a group of people with different levels of understanding and mental models work on the same codebase, having these at our disposal is a proven way to evolve the software while keeping everyone's understanding aligned.

Speaking of design patterns, I learned that they are a useful vocabulary for identifying common problems and possible solutions. Several came up again and again: [Adapter pattern](https://refactoring.guru/design-patterns/adapter), which let us wrap different backend databases behind a consistent interface; [Bridge pattern](https://refactoring.guru/design-patterns/bridge), which we used to decouple the Airflow operator from the unit of work it carries (what we call Step); and [Strategy pattern](https://refactoring.guru/design-patterns/strategy), which became the backbone of our audit framework for checking data quality.

On abstraction: I started off writing procedural code to implement our write-audit-publish ([WAP](https://www.youtube.com/watch?v=fXHdeBnpXrg&t=990s)) pattern, then watched more experienced engineers replace it with abstractions that simplified the code I'd written and made writing new code easier. In another example, we introduced a Spell abstraction that lets developers add codemod capabilities, invoked by both developers and CI for config validation. It replaced a series of very fragile shell scripts strung together over the years. Nowadays, if I find myself writing similar code with unnecessary implementation detail for the task at hand more than once, I pause and ask whether there's a useful abstraction we can introduce to hide that complexity.

I've also picked up several other technical lessons along the way: domain modeling, why [choosing boring technology](https://mcfunley.com/choose-boring-technology) is usually the wiser choice, composition over inheritance, dependency injection and inversion, and snapshot testing. Each one probably deserves its own blog post. The common thread is that all of these took time to internalize. I only appreciated them after working in the same codebase long enough to see why they mattered.

### On Maintaining Software

There was a period where we were heads down building and re-architecting. There are also periods that call for reliability and stability over agility and speed of change. Navigating different phases of a software lifecycle requires shifting how you think and where you focus, and for me, it took a few painful incidents before I really internalized that.

For a while we treated every dataset roughly with the same priority. Once the platform was adopted company-wide, it became clear that financial reporting and executive dashboards were far more critical than someone's ad-hoc report. We introduced a data tiering system that lets us prioritize compute resources according to each dataset's tier and SLA requirements. We also invested in observability. For a long time we didn't know what "healthy" looked like for most of our systems, so we built layered observability: real-time, intraday, daily, and defined "good" for each tier. To keep low-signal alerts from burying the real issues, we routed alerts to the right owners and tuned thresholds until the ones that fired were worth acting on.

To harden our system, we learned from every incident for corrective action. One recurring lesson: big-bang releases are risky because the blast radius is enormous. We leaned into gradual rollouts, feature flags, and staged deployments, and over time deployments got calmer. We also surveyed the team on which parts of the job were intolerably repetitive and automated as much of it as we could. Restarting failed jobs, validating releases, and clearing alerts by hand were the biggest offenders, and we automated most of them over time.

At times, we needed to make much bigger changes. The accumulated weight of new use cases and tech debt made the architecture itself unmaintainable, and the only real fix was to rebuild the foundation. We did that twice with Minerva over seven years, and I expect it will happen again as requirements and use cases continue to evolve.

### On Ownership Mindset

Having worked on Airbnb's Data Platform for many years, I've come to believe that one of the keys separating a successful project from a mediocre one is the level of care contributors bring, the ownership mindset that makes you take pride in your work.

For us, that mindset showed up in everyday choices. While it was often tempting to build the intellectually interesting thing, we pushed back on our own over-designed proposals, and more than once killed them, in favor of something simpler that unblocked users. We cleaned up technical debt when the opportunity came, not because we treated debt as something to avoid at all costs, but because we knew it would help other developers down the line, even if users never noticed. And when users ran into issues, we took pride in unblocking them fast. Our on-call rotation was where we learned what was broken, not just a chore to rotate through.

Ownership mindset mattered even more during difficult times. There were periods when reliability problems hurt our team's reputation, voluntary attrition left us severely under-resourced, and we couldn't prioritize the work users were asking for. In moments like those, we called out the problems honestly and took steps to address them before they turned into fires. When the difficulty was structural, organizational misalignment rather than a technical gap, we weren't shy about rallying leadership to make bigger changes.

I was lucky to work on a team where everyone showed a high level of agency and ownership. That's part of why Minerva stuck with Airbnb's data community for as long as it did.

## Personal Lessons

### Switching Roles

Some of the defining moments of my time at Airbnb came when I stepped outside my existing role and started over from scratch.

From 2016-2019, I worked as a data scientist helping grow our host community across several marketplace tiers (remember Airbnb Plus?). When the company dismantled its data engineering org, I was one of the few data scientists who dove into the wreckage and rebuilt from there, and found I enjoyed the work. Building pipelines and producing high-quality datasets taught me how much leverage comes from the right tooling, so from 2019 to 2021, I pivoted into product management to help scale Airbnb's semantic layer from 0.5 to 100. From 2021 on, I worked as a SWE on the Minerva team to rebuild our stack from the ground up.

A shift in your interests doesn't automatically earn the organization's trust that you can execute in a new role. Every stretch into something new is a bet the org makes on you, and it's on you to make it pay off. My playbook stayed the same each time: do great work, earn a reputation that precedes you, find an adjacent area that interests you, identify a sponsor, then pivot if that move propels growth.

Switching functions re-accelerates your growth. It isn't free, either. It can slow your climb up any single ladder, and the expertise you've built fades into the background, at least for a while. You have to be willing to sit with that discomfort and be honest about your own learning curve (more on that in the next section).

Looking back, this is the main reason I stayed at Airbnb as long as I did. It felt like three different jobs packed into one decade, each giving me a distinct experience and its own growth opportunities. In the age of AI, I think the people who can operate beyond any single role are the ones best positioned to thrive, and I'm curious where that takes me next.

### Growing Pains

With each transition come growing pains. Two stories from my time at Airbnb still stick with me. In one, I leaned in immediately. In the other, I flinched for months before turning it around.

In my first week (yes, first!) as the new product manager for the Minerva team, we had the largest incident in the team's history, internally known as CIM-198. It was big enough that we had to put a moratorium on Minerva, blocking users from contributing new semantics to our platform. My immediate job was to communicate the scope of the incident and how we'd fix it. My broader job was to work with engineers to find the gaps in our system and build both short- and long-term fixes so it couldn't happen again.

The whole thing felt like an extended interview, testing product management skills I barely had yet. Our users were patient, leadership gave us room to work, and I partnered closely with engineering to build a solid plan. We shipped several new features, including offline backfill, which still plays a key role in the architecture today. The incident was memorable enough that we eventually printed T-shirts reading "I survived CIM-198," a badge of honor I hope we never have to award again. I had fond memories of that first week and thought I handled it well.

The second story is from when I first transitioned to software engineering. I worked with a mentor who was extremely capable and had a high bar. His style and presence could be dominating, which made me feel insecure. Unlike the CIM-198 crisis, there was no deadline compelling me to act here, so instead of leaning in, I did the opposite. I worried constantly about people's perception of me, afraid of wasting his time, asking naive questions, and looking dumb. I'd charge ahead on implementations without checking in, only to surface a PR too late for meaningful feedback. It got bad enough that one day he asked me, point blank, "Are you avoiding me?" That was a rude awakening.

Something he told me afterward has stuck with me since: even if my judgment was 99% bad starting out, as long as I was willing to reflect on why, my taste would improve over time. That was the actual path to growth. It took some psychological work, but I realized a strong re-start meant embracing the growing pains and letting go of my ego. I eventually leaned into the discomfort and changed how I conduct myself. I started pair programming more, asking questions earlier, and letting myself look unsure. Six months later, that same mentor told me another respected engineer on the team considered me one of the most reliable people he worked with.

The difference between the two stories was how quickly I let go of my ego and embraced the growing pains. I'd like to think I've gotten faster at that over time.

### Seeking Impact

In most companies, growth is measured by your level on the career ladder. It's a useful external scorecard, but orient your whole career around it and you might end up chasing levels and labels instead of the work that energizes you.

I learned this the hard way as a PM scaling Minerva. By most external measures, things were going well. I had real impact, worked with great people, and the product was growing fast. The day-to-day was wearing me down anyway: constant context switching, no time to think deeply, managing up, down, and sideways all at once. I started seriously thinking about leaving the company.

What changed my mind was a conversation with my partner. She listened to me vent, then said: "It seems like you're still very passionate about the company and the team's mission. You're just in the wrong job." She was right. I still cared about Airbnb, the team, and the mission. What had happened was that I'd drifted into a shape of work that didn't fit me, and I hadn't been honest with myself about it. I'd gotten pulled in by the feeling of having impact and stopped asking whether the work itself was right.

That conversation pushed me to advocate for a move back into a more technical role, one built for deep problem solving. It wasn't a straightforward move on paper, giving up a PM role to start over as an IC in engineering. It was the right call for what I wanted my days to look like, and it set up the most fulfilling chapter of my time at Airbnb.

Nobody will advocate for the work you love more than you will. Your manager has other priorities to juggle, the org has its own gravity, and the career ladder keeps pulling you toward the next level whether or not that's what you want. Knowing yourself well enough to push back, and having the courage to do it, matters more than any promotion I got.

## Parting Thoughts

I never thought I'd stay at Airbnb this long.

Looking back, it felt like working three distinct jobs across three different companies. Careers are non-linear, and staying self-aware is what let me course correct along the way. In the process, I came to understand what I'm good at, what I enjoy, and what I want more of. I made a real impact, and I was lucky to work alongside some of the kindest and most talented people I know, some of whom I now consider dear friends.

I'll miss Airbnb dearly. I'm excited to see what the next chapter looks like, and this decade has given me what I need to navigate it.

—

*Thank you, Jason, for onboarding me to Airbnb, which led to a decade-long friendship. Thank you, Vaughn, for teaching me everything about Airbnb and what it means to do right by the business. Thank you, Aaron, for teaching me Data Engineering. Thank you, Ricardo and Cuky, for believing in me as a Data Scientist. Thank you, Jeff, for betting on me to lead the Minerva team when I had no PM experience. Thank you, Shao, Dave, and Vyl, for helping me transition to SWE when I asked for it. Thank you to the whole Minerva team for working alongside me all those years. Thank you, Philip, Krist, and Clark, for being constants on the Minerva team and going through the ups and downs together. Thank you, Toby, Chris, Barak, and Ginter, for showing me what it means to be an outstanding engineer.*
