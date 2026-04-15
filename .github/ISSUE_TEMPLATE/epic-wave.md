---
name: Wave Epic (L1)
about: A release-scoped mega-epic grouping theme epics under one milestone.
title: "[Wave N] <title>"
labels: ["type:epic-wave"]
---

## Goal
<!-- One-sentence outcome this wave delivers. Maps to a milestone / release. -->

## Wave / Milestone
- Wave: `wave:N-<slug>`
- Milestone: `vX.Y-<slug>`

## Motivation
<!-- Why this wave now; what it unlocks; what it blocks. -->

## Theme Epics (parallel-safe)
<!-- One section per theme. Sections can run in parallel unless noted. -->

### Section A — <theme> (parallel with B, C, …)
- [ ] #<theme-epic-issue>

### Section B — <theme>
- [ ] #<theme-epic-issue>

## Sequential Dependencies
<!-- e.g. Section A → Section B; Section C can start anytime. -->

## Definition of Done (Wave)
- [ ] All theme epics closed
- [ ] Milestone release cut (tag + changelog)
- [ ] `make test && make lint && make typecheck` green on `main`
- [ ] Docs published

## Relationships
- Blocked by: #
- Blocks: #
- Related: #
