---
layout: post
title:  "Airbnb's Semantic Layer: Developer Experience"
date:   2026-08-10 09:00:00 +0800
comments: True
ref: semantic-layer-developer-experience
excerpt: "Over seven years, Minerva grew into the largest data framework at Airbnb. This first post looks at the developer experience decisions behind it: configuration as code, validation, ownership, and dry runs."
---

## Introduction

By the time I left Airbnb, [Minerva](https://medium.com/airbnb-engineering/how-airbnb-achieved-metric-consistency-at-scale-f23cc53dea70), Airbnb's semantic layer, had reached every part of the company. We had about 200 data producers submitting 100+ PRs per week, close to 10,000 configs generating 20,000+ DAGs and 10,000+ Iceberg tables. Minerva regularly accounted for more than half of Airbnb's backfill capacity and was the largest data framework at the company by Airflow DAG count. On the consumption side, it powered 60,000+ Superset charts and Tableau dashboards for more than 4,000 consumers.

Almost none of that was in view when we started. Over the years we made engineering decisions, some good and some bad, that shaped where the platform is today. I want to use this post to share some of those decisions and what we learned from them.

## Developer Experience

With so many data producers using our tools daily, workflow design mattered a lot. The number one concern from our users was always iteration speed. People wanted to create, test, and iterate on their changes with a fast feedback loop. That became our north star, and it shaped how we treated configuration, validation, review, and dry runs.

### Configuration as Code

Minerva was a configuration-driven framework. Users declared semantic definitions in YAML, and Minerva treated those definitions as code, stored in git, version controlled, and reviewed before they landed. Treating data as code is popular now, thanks partly to tools like [dbt](https://www.getdbt.com/), and Minerva did this at Airbnb scale.

Storing semantics in git creates a real challenge around discoverability. Without conventions, there's no easy way to navigate the files, and when definitions are shared across hundreds of teams, people who can't find existing ones end up rewriting them. In the first version of Minerva, we integrated with [Dataportal](https://medium.com/airbnb-engineering/democratizing-data-at-airbnb-852d76c51770), Airbnb's data catalog, and surfaced configs inline when people searched for metrics and dimensions. The second version went further, adding a dedicated UI, Minerva Studio, that surfaced the same information plus richer metadata.

Readability inside the config was its own problem. The first version had no tool like [sqlglot](https://github.com/tobymao/sqlglot) to parse SQL, so we relied on custom fields and a DSL to define metrics, adding more of both whenever people needed to reuse a definition. Over time this got unwieldy. Understanding a definition meant learning the DSL, then piecing it together across several places in the config. The second version moved metric definitions to plain SQL and used sqlglot to parse and validate them, which simplified things considerably, since you could read a definition directly, inline.

### Validation

Not every configuration was correct, so the framework needed validation to catch problems before merging. We built a validation suite that ran at two points in the developer lifecycle, locally as users iterated on their configs, and in CI when a PR was created.

In the first version of Minerva, validations were strung together through a series of fragile shell scripts. The same scripts ran for both users and CI, so we often had to add branching logic based on the environment. They also lacked single responsibilities. They called other scripts that did specific validations, and there was no easy way to know who owned what. Being shell scripts, they were close to impossible to unit test.

When we developed the second version of Minerva, my colleagues Philip and Krist took on the daunting task of revamping our validation suite. They used [Cerberus](https://docs.python-cerberus.org/), a data validation framework for Python, to standardize the basic YAML validation tasks. In Cerberus you define a [validation schema](https://docs.python-cerberus.org/schemas.html), a mapping of schema keys to schema values, where the values are predefined rules for what each key can take. For example:

```python
schema = {'name': {'type': 'string', 'maxlength': 10}}
```

This says the name field has to be a string and cannot exceed a length of 10. Cerberus handled a large share of this type of standardized validations well, and we only added custom rules when necessary, which kept validations relatively cheap.

Another improvement Krist introduced was what he called the Spell framework, a CLI tool built on top of [Typer](https://typer.tiangolo.com/). Different validation tasks could be implemented as Spells, and each Spell could be unit tested in our codebase. Each Spell was a CLI command with its own arguments, which users could run locally:

```bash
minerva validate
```

The same Spells users ran locally also ran in CI. One particularly nice set was the auto-fix Spells, which detected issues and fixed them on the user's behalf. Some of these operations were slow, so we started tracking how long CI jobs took to make sure we weren't introducing regressions. Every second of CI time showed up in someone's iteration loop.

We also put considerable effort into documenting what each field means, and we added clear error messages along with breadcrumbs that showed users how to fix a misconfiguration. This cut down how often people had to come to the oncall channel.

### Ownership, Review, and PR Approvals

As more business definitions got codified in Minerva, we realized the platform was not just a store for the source of truth. It was also the machine that facilitated change management and data governance at scale. Users had to see what changed and who changed it. They also needed a way to reach the owners when something needed discussion. We treated ownership as a first-class concept from day one, and that had real implications for review, approval, and lifecycle management.

Minerva had a concept called a team. Each team carried information like a list of maintainers, PagerDuty emails, and Slack channels. Teams could be attached as owners to specific datasets, and owning a dataset came with real responsibilities.

Owners were tagged whenever someone else edited one of their datasets, and they could approve or request changes before the edit merged. The PR review process became a place for different teams to debate and reconcile how business semantics should be defined. It did introduce friction into the developer flow, and collecting all the stamps from owners could take a while. We introduced different levels of reviewers, so lighter changes went through a lighter process, and we partnered with analytics engineering to build a Minerva reviewer program, where reviewers could veto or approve changes.

Owners were also the first line of defense for pipeline failures and delays. Any alert triggered by a failure went to the owning team, and they investigated why a pipeline was delayed or failed, escalating to the Minerva team only when they believed there was a wider infrastructure issue. This worked well for teams with dedicated oncall engineers. Teams without an oncall rotation often got fatigued by the alert emails, ignored them, and came straight to our support channel, which became a heavy source of operational load for us.

Owners were also the ones who could answer incoming questions about what a business definition actually meant. They understood the context better than the Minerva platform team did, so we surfaced them on every Minerva asset page in Dataportal. We did the same in SQL Lab, our SQL editor, so someone who hit a problem querying a metric or dimension could reach the owner directly. In the age of AI, it's not hard to imagine that these owners would become the curators and stewards of specific domain knowledge in their respective areas. This I imagine will be a key thing that drives context engineering at scale.

### Dry Run

In the early days of Minerva, we had no mechanism that let users dry run their changes. All changes were tested in production, and that caused real problems. The iteration cycle was slow. Users had to go through the PR review process and get approvals just to test their changes. It was expensive and risky for data integrity too. We wasted compute backfilling data that could be wrong, and we might publish it without consumers knowing the data was bad.

This shortfall was enough that we introduced a new workflow, dry run. The idea was that users should test their changes before putting them into production. When a user put up a PR, we serialized the contents of the YAML configs and diffed them against what was in production. From that diff we calculated which datasets had changed, similar to how git diff works, and surfaced useful information like the diff tree and the estimated backfill cost of the change. Philip demonstrates this in more detail in his talk [here](https://www.youtube.com/watch?v=JHQqi5fdo-s&t=1518s), for anyone curious.

Dry runs did not trigger a full backfill. They ran a targeted date range and wrote the data into a temp namespace isolated from production. When a dry run completed, users ran a command to generate a receipt as proof of work. Without it, validation flagged that the change had not been tested. Dry run was popular once it landed. People no longer had to go through an elaborate review workflow just to test something. They could play with the data in the temp namespace and make further modifications, which sped up the iteration cycle considerably.

Right before I left Airbnb, the team had been investing in the next iteration of the dry run experience. The key change is that users will be able to dry run without even putting up a PR, and the temp data will show up not only in the warehouse, but also in the query layer. This means that users can now query these test data in our BI tools directly! Much of this workflow is CLI driven, which means AI can run it too. The goal is for AI to eventually complete the loop with little human intervention, which will speed iteration even more. I think it could be the next step-function change to developer experience.

## Summary

When you're building a data framework, developer experience matters. This matters less with a handful of users. At a certain scale, small improvements to that experience end up having an outsized impact.

Building Minerva taught us that users care deeply about iteration speed, and that hasn't changed in the age of AI. If anything, the bar is higher now. Keeping semantics as code in git gave us versioning and review for free, though it put the burden on us to make definitions discoverable and readable. We spent a lot of time on validation, catching errors as early as we could, and auto-fixing issues in the background when that made sense. The dry run workflow got heavy investment too, since it let people test changes end to end. Finally, we invested in ownership. We codified it across review and oncall, then put owners front and center in the data catalog, and that's what let us scale ownership in a distributed way.

In the next post I'll go into the second big component of Minerva — compute, and how we built a system that keeps data in the warehouse in sync with the latest business definitions.
