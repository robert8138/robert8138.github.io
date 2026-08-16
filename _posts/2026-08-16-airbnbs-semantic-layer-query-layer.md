---
layout: post
title:  "Airbnb's Semantic Layer: MinervaSQL"
date:   2026-08-16 09:00:00 +0800
comments: True
ref: semantic-layer-query-layer
excerpt: "Minerva Compute creates 10,000+ materialized tables, and no user can remember which one to query. This final post is about the query layer that hides them: MinervaSQL, rewriting, source selection, and auto-denormalization."
---

## Introduction

This is the third and final post in the Airbnb Semantic Layer series (see Parts [I](https://robert8138.github.io/2026/08/10/airbnbs-semantic-layer-developer-experience.html) and [II](https://robert8138.github.io/2026/08/13/airbnbs-semantic-layer-compute.html)), covering the engineering decisions that shaped Minerva, our semantic layer, into what it is today. Minerva has 4,000+ consumers at Airbnb, and their use cases vary widely. The query layer is what makes data access possible at that scale. If you want a primer before reading on, Barak gave a [lightning talk](https://www.youtube.com/watch?v=xPFPOEDYV9g) that introduces MinervaSQL, our query layer.

## MinervaSQL

In the previous [post](https://robert8138.github.io/2026/08/13/airbnbs-semantic-layer-compute.html), we highlighted that Minerva Compute creates more than 10,000 materialized tables. At that scale, no user can reasonably remember which tables to query. One common solution is to denormalize data that gets queried together, so users only have to hit a handful of wide tables. The logical extreme of this is the [One Big Table](https://www.ssp.sh/brain/one-big-table/) (OBT) approach, where everything is denormalized into a single wide table. At Minerva's scale, OBT is prohibitively expensive. Imagine how many dependencies such a table would carry, not to mention the time it would take to backfill or the storage it would consume.

Minerva's query layer applies the OBT idea to a virtual table (`magic.all`), so nothing has to be materialized into one giant physical table. Users write queries in a custom dialect called MinervaSQL against that virtual table, and the query layer rewrites them against the physical tables, working out the joins automatically. Metrics are queried through a special `AGG` syntax, which gets rewritten into the right aggregate expression based on the metric definition.

This abstraction cut down how much SQL users had to write to answer their questions. Over time, we added more syntax to the dialect to support sophisticated query patterns. `FILTER` lets users define filtered metrics on the fly. With `SHIFT`, time-over-time comparisons became easy to express. We also added configuration that lets advanced users control which source to query when several are eligible to serve the same data.

### Rewriting

A big challenge of the **One Big Virtual Table** (OBVT) approach is turning a query against the virtual table into queries against the physical tables underneath. We made this possible with **Rewriter**, a pipeline of rewriting rules over SQL ASTs. Each rule receives a syntax tree and returns a transformed one, and at every step the tree remains a valid MinervaSQL expression.

The pipeline has many rules, but the rewriting comes down to a few key steps. The first is temporal and **metric resolution**. Earlier we mentioned the `AGG` expression, which doesn't contain the real metric aggregate expression. The metric resolution rule expands it into that expression and adds a source hint tracking where the metric came from. Those hints are the breadcrumbs the rest of the pipeline follows to expand and rewrite the query.

Working from the **source hints**, the rewriter then tries to create a single `SELECT` per scope, which is what eventually lets us swap entities for physical tables. For a metric from a single source, this is relatively straightforward. Derived metrics and drill-across queries take more work: the source-tagged aggregates get split into one subquery per source, then joined back together with a `FULL OUTER JOIN` on the shared dimensions. The last step replaces all entity references with physical tables, leaving no magic tables and no `@` columns anywhere in the tree.

This only touches the surface of the pipeline. We've added many rules over the years to accommodate how people actually query. Since rules are pure functions and independently testable, we've built detailed test cases and fixtures for different scenarios, which is why we can change the rewriting logic without worrying much about breaking it.

### Source Selection

The section above reads as if there's always exactly one physical table that can answer a metric-by-dimension query. In practice there usually isn't, because Minerva builds precomputed datasets like dimension sets and rollup sources to serve queries from. Paying extra cost on write buys enormous savings on read, and picking the cheapest correct dataset for a given query is a big part of what makes Minerva both performant and trustworthy.

The algorithm walks a fallback chain. Rollup sources get tried first, since pre-aggregated data is the cheapest thing to read. A rollup qualifies when its dimensions are a superset of the query's group-by dimensions, so the query can re-aggregate up to a coarser grain, and when every aggregate in the query decomposes into additive components the rollup already stores.

Plenty of queries fail those checks, whether because they aren't aggregate queries at all, or because a metric or a grain isn't covered. Those go to dimension sets, where we look for a set containing all the dimensions the query touches. Serving from a dimension set still helps enormously, since joins are usually the expensive part of a wide multi-dimension query. If nothing precomputed qualifies, we fall back to joining and aggregating on the fly.

All of this rests on every option returning the same result, since otherwise the numbers users see would depend on which source happened to win. That turns out to be hard, because the pipelines behind these datasets land on different schedules, which leaves the data across them **eventually consistent**.

Inconsistency was a big problem when MinervaSQL first launched, and we introduced `WATERMARK` to address it. Watermarking reads everything as of the oldest available partition, so all sources agree on what "now" means. Dashboards that need both speed and recency get `HYBRID DENORMALIZATION`, which UNIONs precomputed data with recent data at a split date. This area is still under active development, since freshness and consistency trade against each other.

### Auto-denormalization

As described above, precomputed datasets are much more efficient to read from, but curating them can be tedious. In the early days of Minerva, users curated dimension sets by hand. That took time, and different teams would end up creating very similar sets. Minerva 1.0 had no mechanism for pre-aggregated datasets at all, so everything got ingested into Druid instead. That worked for a while, though operating it was always a pain because it required specialized Druid knowledge.

In Minerva 2.0, we flipped the model so that most datasets are created in the background without users knowing about it. An auto-denormalization framework does this by reading MinervaSQL query history, modeling the space of possible denormalizations, and proposing denormalized datasets whose query savings most outweigh their materialization costs.

At Airbnb, query history lives in Elasticsearch, with every query MinervaSQL serves logged as a document. Rewriting a query means the system already knows which sources get touched, which metrics were aggregated, and which dimensions were grouped, so for any given query we can recover what was queried, what was aggregated, and at what grain. From there we de-duplicate queries and collect query counts, which give us a way to estimate query cost. We also track dimension cardinality, row counts, and the number of joins, all of which feed into the materialization cost calculation later.

That leaves the selection algorithm itself. Precomputed datasets aren't free, so the job is to find the sets whose query savings most outweigh what they cost to materialize. Harinarayan, Rajaraman and Ullman worked this out in [Implementing Data Cubes Efficiently](https://web.eecs.umich.edu/~jag/eecs584/papers/implementing_data_cube.pdf), which describes the greedy selection procedure at the heart of our implementation. Glossing over a lot, the selector tells us which precomputed datasets would pay off most, and we create those behind the scenes. It runs on a regular schedule, since query patterns keep shifting.

## Summary

When MinervaSQL was first introduced, it didn't immediately become the workhorse for our users. Many people were still used to writing their own queries against physical datasets. We spent a lot of time evangelizing what MinervaSQL could do, and over time users started to see how complex queries could be expressed in a much simpler dialect, which is when it began to gain traction.

With AI, MinervaSQL became the engine behind data agents. Natural language can be translated into MinervaSQL, which then translates deterministically into the dialect the compute engine understands. That gives us self-serve analytics without giving up correctness. A lot of good engineering went into this, and I think it's what makes Minerva's semantic layer state of the art in the industry. Huge kudos to coworkers like [Barak](https://www.linkedin.com/in/barakalon/), who built this from the ground up.

That concludes the three-part series. If you've read this far, I hope you came away with a better sense of how we built and scaled Minerva at Airbnb. It's been a fun ride, and I feel lucky to have been part of it.
