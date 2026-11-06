# Repository working rules

## Git approval

- The user authorized the initial Git baseline commit of the existing project.
- After that baseline, finish the requested work and relevant checks, then show
  the user the changes and validation results for review.
- Wait for the user's explicit approval of those changes before creating a Git
  commit. Approval of the initial baseline is not approval of later commits.
- Do not amend a commit or otherwise incorporate unapproved work into history.
- Keep unapproved development changes in the working tree for review.
- Do not publish or push this repository unless the user requests it.

## Experimental records

- Preserve the existing pilot, round2, and confirmation results and their source
  snapshots. Put new experiments in separate directories with explicit protocols.
- Keep calibration/validation selection separate from held-out evaluation.
- Distinguish similarity to an uncached output from application quality.
- Keep downloaded models, raw datasets, local environments, full generated image
  sets, and large per-trajectory tensors outside Git. Retain compact statistics,
  aggregate calibration costs, policies, provenance, and summary figures.
- Do not delete ignored local assets as part of repository cleanup.
