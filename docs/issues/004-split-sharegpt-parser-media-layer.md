# 004 Split ShareGPT Parser Media Layer

Status: deferred

## Problem

`ShareGPTParser` currently handles message normalization, media token
consumption, indexed references, source-relative path resolution, ground truth
fallbacks, and dropped-key diagnostics in one class.

## Desired Outcome

Move media/reference materialization behind a focused module so parser changes
can be reviewed independently from row-level sample normalization.

## Acceptance

- Parser orchestration stays small.
- Media token behavior and `extra_info` reference behavior remain covered by
  tests.
- No schema behavior changes are introduced by the split.
