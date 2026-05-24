# 003 Remove Import-Time Prints

Status: done

## Problem

JSON backend selection currently prints during module import. In distributed
training workers this creates noisy logs unrelated to dataset content.

## Desired Outcome

Importing dataset utilities is silent by default. Backend information is
available through normal logging only when logging is configured to show it.

## Acceptance

- Importing `sharegpt_utils.io` prints nothing.
- Ordinary dataset status messages use module loggers instead of `print`.
