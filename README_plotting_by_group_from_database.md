# README: plotting_by_group_from_database.ipynb

## Scope
This README documents the notebook [plotting_by_group_from_database.ipynb](plotting_by_group_from_database.ipynb).

## Current Status (Important)
At the time this README was generated, the notebook has no cells and no code:

- The JSON content is `{"cells": []}`.
- There are no function definitions.
- There are no function calls.
- There is no execution order.
- There are no declared inputs or outputs.

Because of that, there is currently no dependency graph to document for this notebook.

## Function Inventory
Current function inventory in this notebook:

- None

## Dependency Graph
Current dependency graph:

- None (empty notebook)

## Required Inputs
Current required inputs:

- None

## Produced Outputs
Current produced outputs:

- None

## Run Order
Current run order:

1. No runnable cells exist.

## What This Means Practically
This notebook does not currently do any data loading, computation, plotting, statistics, or file export. If you run all cells, nothing will execute because there are no cells to run.

## Recommended Structure To Avoid Future Confusion
When you start adding cells, use this strict order and keep one responsibility per section:

1. Setup section:
- Imports.
- Global config (paths, constants, plotting style).
- Validation checks for required files.

2. Input section:
- Read master file/database table.
- Validate required columns.
- Normalize condition/genotype labels.

3. Processing section:
- Define and run transformation functions.
- Create analysis table(s) used by plots/stats.

4. Plotting section:
- Plot functions only (no data mutation inside plotting functions).
- Save figures to an explicit output folder.

5. Statistics section:
- Run ANOVA/posthoc tests on processed tables.
- Write stats outputs to file.

6. Export section:
- Save final long-format and wide-format analysis tables.
- Save metadata (date, source file path, row counts).

7. Driver section:
- A final "run pipeline" cell that calls the above in order.

## Suggested Function Contract Template
Use this template for each function you add:

- Name:
- Purpose:
- Inputs (columns/types/units):
- Outputs (columns/types/units):
- Depends on functions:
- Called by:
- Side effects (files written/plots generated):
- Failure modes and checks:

## Suggested Cell Header Template
Put this one-line header at the top of each code cell:

- `# Stage: <setup|input|process|plot|stats|export> | Produces: <object names>`

This makes the execution order and data flow visible at a glance.

## Optional Next Step
If you intended to analyze a different notebook (for example [ephys_plotting_by_group.ipynb](ephys_plotting_by_group.ipynb)), generate a separate README for that notebook because it does contain functions and dependencies.
