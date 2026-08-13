---
layout: post
title:  "Airbnb's Semantic Layer: Compute"
date:   2026-08-13 09:00:00 +0800
comments: True
ref: semantic-layer-compute
excerpt: "Minerva keeps 10,000+ sources up-to-date across 20,000+ Airflow DAGs. This second post is about the compute framework behind that: change detection, reconciliation, state tracking, and lifecycle management."
---

## Introduction

This is a follow-up to [Airbnb's Semantic Layer: Developer Experience](https://robert8138.github.io/2026/08/10/airbnbs-semantic-layer-developer-experience.html), where I covered the engineering decisions that shaped Minerva, our semantic layer, into what it is today. Minerva is the largest data framework at Airbnb by Airflow DAG count, and it regularly accounts for more than half of the company's backfill capacity. This post is about the compute side: how we keep 10,000+ Minerva sources up-to-date across 20,000+ Airflow DAGs.

## Compute

Once semantics are defined in the semantic layer, we want to materialize those definitions so the query layer can read from them. Thousands of sources and definitions change constantly, and with 100+ PRs updating these definitions every week, we need a framework that can manage those changes at scale.

To start, we model the entire semantic layer as a **graph**. Each node is a dataset registered in Minerva, and two nodes connect if one was created downstream of the other. That structure lets us identify which subset of the graph changes whenever a definition gets updated. The compute framework then figures out what changed and backfills the affected datasets automatically.

For every source, Minerva continuously **reconciles** the desired state of a dataset against its actual state. Any gap between the two becomes a unit of backfill work, and the framework dispatches it. As datasets get updated, we track their **state**. Compute uses these states to trigger downstream processing, and the query layer relies on them to make sure it only queries up-to-date datasets. When a dataset goes stale or breaks, our **janitor process** handles lifecycle cleanup.

The sections below cover each of these in more detail. We will use *dataset* and *source* interchangeably for the remainder of the post.

### Detecting Change

Each dataset in the Minerva graph has a **data version**, an MD5 hash of every field in the YAML file that encodes the dataset's semantics. Each versioned dataset materializes into an Iceberg table in the warehouse. When a producer edits a config, say by adding a dimension filter or changing a column's projection, the data version updates, telling Minerva to recompute the physical data to match the new definition. That covers a single dataset. Things get more interesting when datasets depend on each other.

Some datasets in Minerva are derived from others. If a user defines an event source from a fact table and another user defines a dimension source from a dimension table, Minerva can pre-join them into a dataset called a **dimension set**, or pre-aggregate the events up to a particular grain to create a **rollup source**. These pre-joined and pre-aggregated datasets depend on the event source, so when that event source changes, they have to be updated accordingly.

We manage these dependencies through **chained data versions**. A downstream source's data version includes the upstream source's data version as one of its inputs. In the example above, the event source's data version is folded into the data versions of both the dimension set and the rollup source. When the event source changes, those two downstream versions change with it. A single YAML edit can turn into a wave of updates across the graph.

This design cascades change well, but its change detection can be overly aggressive. Say we add a new column to the event source. Downstream sources that never reference that column don't need to be recomputed. They get backfilled anyway, because all we know is that the upstream version changed. Workarounds exist, like pinning a source to a fixed data version, though they get complex fast for large-scale changes. This is an area I wish we'd invested in more, since better change detection could meaningfully cut how much Minerva has to backfill.

Some tools have taken this further. SQLMesh has invested heavily in [sophisticated change detection](https://www.tobikodata.com/blog/are-these-sql-queries-the-same), introducing concepts like breaking and non-breaking changes based on comparing canonicalized SQL ASTs. I believe this is the direction transformation tools should go.

### Reconciliation

With change detection covered, we can zoom in on a single dataset and see how the compute framework reconciles changes. Since most datasets at Airbnb are **date-partitioned**, the reconciliation algorithm depends heavily on partition-level operations.

#### Reconciliation Algorithm

Every day, each dataset works out which input partitions exist and which output partitions have already been written. From that difference it generates a plan, batching the missing date partitions into windows and backfilling them in parallel.

In practice, this reconciliation runs per dataset, per day, in an Airflow DAG. A centralized control plane that loops through all datasets could work just as well, since the reconciliation algorithm itself doesn't depend on how the work gets scheduled. The same algorithm handles several scenarios, and each one dispatches work differently.

#### Recurring Runs

The most common scenario is a source with no changes to its definition. New input data lands each day, and Minerva's job is to materialize it. The algorithm notices that all input partitions exist while the output partitions stop at the previous day, so it generates a single batch covering one date partition. Once that partition is written, the dataset is up-to-date.

Partitioning is what makes incremental compute possible. Some datasets, though, have late-arriving events like cancellations or alterations, so their history keeps changing well after the fact and can't be processed incrementally. We reprocess those from the beginning every day, which makes every run a full historical backfill. That's expensive, so we later extended the algorithm to let users specify an output window, where only the most recent X days get reprocessed instead of the whole history. This works in practice because most users don't need data that far back.

#### Offline Backfills

In the first version of Minerva, every change was backfilled directly in production. We soon found that as users changed business definitions more often, Minerva couldn't keep up, since data stayed unavailable while a backfill was running. That caused a string of availability problems and, eventually, a major incident I described in [Reflections on Airbnb](https://robert8138.github.io/2026/08/05/reflections-on-airbnb.html). We learned that users needed an isolated environment to backfill in.

Luckily, each versioned dataset already materializes into its own table, so we can kick off backfills and write new data into a table completely isolated from production. The same reconciliation algorithm applies here. It sees that all input partitions exist while the output table is empty, so it backfills the table from the beginning of time to the current date. This can take a while, so the algorithm splits the work into disjoint batches and runs them concurrently, which cuts backfill time considerably.

Offline backfills mean users can rebuild their datasets without time pressure, and when a dataset is backfilled, promotion to production is instant. Tools like SQLMesh have reached the same conclusion and shipped features such as [Virtual Data Environments](https://www.tobikodata.com/blog/virtual-data-environments).

#### Online Backfills

The last scenario sits between recurring runs and a full historical backfill. Occasionally, users discovered that the input tables feeding Minerva were corrupted. Fixing means restating the affected Minerva data from the corrected input table, without touching the other partitions. This is exactly what online backfill does. Users surgically re-materialize a subset of Minerva partitions in place.

We adapted the self-healing algorithm to handle this. When an online backfill is triggered, we force a re-run of a subset of the output partitions by telling the algorithm they do not exist. From there it takes its usual course, creating batch windows and dispatching backfills accordingly. The data lands in the same production table under the same data version, since no new semantics were introduced.

#### Putting Everything Together

Right before I left Airbnb, the team had been working to unify these disparate workflows into a single workflow with distinct phases. A new dataset typically moves through a dry run, then an offline backfill, and eventually gets promoted to production, where it runs on a recurring basis.

### Tracking Source State

With thousands of sources that need to be processed in the correct order, we need to track the state of each one closely. That matters even more because backfilling is a long, async process that depends on how much compute capacity or orchestration slots are available at any given time.

We addressed this with source state, a suite of API endpoints for writing and reading the current state of a source. It tracks the source's latest fingerprint, data version, and available partitions. Once a source is fully backfilled, we post an update to the API, and compute and the query layer both read from it.

For compute, each dataset is scheduled in its own Airflow DAG, so we built a custom source state sensor to coordinate dependencies between them. It pokes the state of the upstream datasets a DAG depends on, checking whether they've landed, and only passes once they have. That keeps a downstream dataset from starting before its inputs are ready.

The query layer uses source state through a fingerprint, a hash of the physical data that was just written. Unlike a data version, a fingerprint tells you whether the physical data actually changed. For Iceberg tables, it's the `snapshot_id` of the new Iceberg snapshot. Other engines use a SHA256 hash of the table name and the latest partition date instead. With the fingerprint, the query layer can use it as a cache to serve common queries.

### Lifecycle Management

When new datasets come online, older ones go stale, and Minerva runs a janitor process to clean them up. Deletion happens in two phases. Soft deletion comes first, then hard deletion. Candidates for soft deletion are flagged based on usage, aggregated across query logs. When a source is soft deleted, we hide its data assets from search in the catalog and pause the associated pipelines, without deleting any data.

After a grace period, hard deletion happens. Hard deletion cleans up both the configurations and the underlying data. For configuration, we have a set of deleters that walk the Minerva graph and clean everything defined downstream of a source. Once that change merges, the Airflow DAGs get dismantled, and a separate janitor drops the underlying tables and their storage.

We have a robust lifecycle management process today. I wish we'd built it this way earlier in Minerva's history. It's hard to ask users to do this kind of cleanup unless most of it is automated. During my time operating Minerva, we found many opportunities to reduce waste and cost, thanks to these lifecycle management tools.

## Summary

This post covered Minerva's core challenge. Semantic definitions get created, updated, and deleted constantly, and building a compute framework that keeps these datasets up-to-date and consistent sits at the heart of the system.

I walked through the platform's key capabilities: change detection, reconciliation, state tracking, and lifecycle management, and how they fit together to keep the system running. In the next and final post, we'll get into the query layer and see how it reads the datasets Compute produces and serves them to consumers at scale.
