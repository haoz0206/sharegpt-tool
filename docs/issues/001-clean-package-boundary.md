# 001 Clean Package Boundary

Status: in_progress

## Problem

The root package currently attempts to import optional runtime adapters and
uses a broad exception guard. That can hide real bugs in adapter code.

## Desired Outcome

The root package exports only framework-agnostic utilities. Verl adapters stay
available through explicit module imports such as `agent_dataset`.

## Acceptance

- Importing `sharegpt_utils` does not require verl runtime dependencies.
- Root package does not swallow adapter import errors.
- Adapter smoke imports use explicit module paths.
