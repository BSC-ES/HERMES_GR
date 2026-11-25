# AGENTS.md

## 1. Project description

NES (NetCDF for Earth Science) is a Python/MPI library for reading, writing
and processing Earth science datasets. It uses domain decomposition for
parallel I/O and numerical operations on structured grids and point data.
It is used by SNES and other BSC-ES tools, including HERMES_GR.

- `nes/default_nes.py`: base `Nes` class, dataset state and parallel I/O.
- `nes/load_nes.py`, `nes/create_nes.py`: dataset loading and creation.
- `nes/projections/`: concrete grid projections and observation layouts.
- `nes/formats/`: model-specific output adapters.
- `nes/methods/`: interpolation, geospatial operations and cell measures.
- `nes/grid_configuration.py`: grid configuration definitions and validation.
- `nes/utilities/`: shared utilities, including MPI helpers.
- `nes/cli/`: the `nes` command and its subcommands.
- `nes/tests/`: miscellaneous, projection, numerical-method and CLI tests.
- `docs/source/`: Sphinx documentation source. Do not edit generated files in
  `docs/build/` as documentation sources.
- `CHANGELOG.md`: public release history.
- `CONTRIBUTING.md`: contribution workflow and documentation conventions.
- User documentation is also maintained in the separate NES GitHub wiki.

## 2. BSC Earth Sciences (BSC-ES) department conventions

This section records the BSC-ES conventions adopted by this project; it does
not establish policy for other repositories. If a rule below conflicts with
this section, follow this section and ask the user.

### 2.1 Data and file systems

- Never delete, move or overwrite existing input datasets, inventories, model
  outputs or other scientific data without explicit approval. This applies
  above all to shared storage (`/esarchive`, `/gpfs/projects`, other users' folders).
  Requested edits to project source code, configuration and documentation are
  allowed; preserve unrelated user changes.
- Treat `/esarchive` and other shared data archives as **read-only**.
- On HPC systems, write temporary and intermediate data to scratch
  (`/gpfs/scratch/...`), not to home or project space. Scratch may be purged.
  Place requested deliverables in the project or the location specified by
  the user.
- Check how big a dataset is before loading it or copying it. Prefer lazy
  loading or reading only the required variables and domain slices over
  reading everything into memory.
- Never `rm -rf` with a variable or wildcard in the path unless the full
  path has been printed and checked first.

### 2.2 HPC usage (MN5, CIRRUS)

- Login nodes are shared: do not run heavy computations on them (anything
  longer than a few minutes or using many cores/a lot of memory). Use a
  Slurm job (`sbatch`) instead.
- Show the full job script to the user before any `sbatch`, and never submit
  jobs, arrays or chains of jobs without approval.
- Only cancel (`scancel`) jobs created during the current session, unless
  the user asks otherwise.
- Obtain the account, partition and QoS from the user or an approved project
  configuration; do not guess them. No HPC defaults are defined in this file.
- Do not run `ssh` to other machines, or start long-running services, without
  approval.

### 2.3 Software environments

- Never install packages into the conda `base` environment or into shared
  environments (e.g. under `/gpfs/projects/.../shared_conda_environments`).
  Ask before installing anything.
- Check which environment is active before running scripts.
- Use the editable development environment selected by the user as described
  in section 4.1.

### 2.4 Secrets and confidential data

- Never write passwords, API keys or tokens into files, commits or logs. Read
  them from environment variables.
- Do not send project data, unpublished results or code to external services
  (web APIs, online tools, public pastebins) without approval.
- Respect the licence of the input data (e.g. Copernicus, ECMWF, national
  agencies): do not copy restricted data outside its allowed location.

### 2.5 Git

- At the start of a new task, before doing other task work, check the local
  working tree and index for uncommitted changes, including untracked files.
  Follow-up requests, corrections and iterations on the current task are not
  new tasks; do not repeat this checkpoint for them. If the distinction is
  unclear, ask whether the request starts a new task before making edits.
- If uncommitted changes exist, inspect and summarize them and suggest an
  appropriate commit message before proceeding with the new task. This is a
  suggestion only, not authorization to stage files or create a commit.
- After reporting existing changes, read-only work and edits to clean files
  may proceed. Wait for the user's confirmation before editing any file that
  already has uncommitted changes, including staged changes or untracked files.
  Preserve existing changes; do not overwrite or revert them.
- Never `git push`, force-push, rewrite history or delete branches without
  explicit approval.
- Never create a commit unless the user explicitly asks for a commit.
  Requests to implement changes or prepare a release do not authorize commits.
  Commit messages: imperative mood, short subject line.
- Do not stage or unstage files, create or delete tags, merge branches or publish
  releases unless explicitly requested. Preserve the user's existing index.
- Do not commit large data files, model outputs, figures or secrets: update
  `.gitignore` instead.

### 2.6 Working style

- Prefer small, reviewable changes. Explain why a change is needed, not only
  what it does.
- When unsure about anything irreversible or that affects other people, stop
  and ask.
- Report results honestly: if a script fails or a step was skipped, say so.

## 3. Atmospheric Composition (AC) group conventions

This section records the AC-group conventions adopted by this project; it
does not establish policy for other repositories. If a project-specific rule
below conflicts with this section, follow this section and ask the user.

- Comments, documentation and anything shared with others (e.g. GitLab
  issues, merge requests, READMEs) must give the level of detail an expert
  expects, together with a description an intelligent beginner can follow.
- State units, time zone and data sources (with versions) wherever data or
  results are described.
- Explain *why* a choice was made (method, threshold, dataset), not only
  *what* was done.
- Prefer clear, readable code over clever code; define jargon and
  abbreviations the first time they appear.
- Never modify shared model or observation archives; work on copies.

## 4. Project-specific rules

### 4.1 Environments and validation

- The user must specify once which Conda environment to use for editable
  development. Reuse that choice throughout the session and in later sessions
  when it is available in project instructions or conversation context. Ask
  only if no choice is available, the environment becomes unavailable, or the
  user changes it. Do not write personal environment names into shared project
  files unless requested.
- If no environment has been specified, ask before running code that requires
  the project environment. Read-only inspection and documentation edits do not
  require a project environment.
- Confirm the selected environment exists and use it explicitly, with
  `conda run -n <environment> ...` or `conda run -p <environment-path> ...`.
  Do not rely on whichever Python happens to be active.
- Inspect `environment.yml`, `setup.py`, `requirements.txt` and the Makefile
  before changing dependencies or creating an environment. The Makefile has
  editable-install targets; creation, installation and updates require
  approval under section 2.3. Do not invoke installation or cleanup targets
  as routine validation.
- MPI, NetCDF and HDF5 libraries must be compatible. Do not silently replace
  parallel dependencies with serial builds or change MPI implementations.
- New code developments must include tests for the new behavior. Inspect
  related coverage first; extend or parameterize existing tests and reuse
  fixtures when practical. Cover edge cases and regressions rather than
  mirroring implementation details.
- After code changes, run `pytest nes/tests/misc/test_lint.py` and tests related
  to the modified behavior after the final edits. Documentation-only changes
  do not require the Python test suite.
- Consult `.gitlab-ci.yml` and the Makefile for current checks. CI covers
  `nes/tests/misc`, `nes/tests/projections`, `nes/tests/methods` and horizontal
  interpolation proxy tests. CLI tests are currently commented out in CI;
  run relevant tests in `nes/tests/cli` when changing CLI behavior.
- Run broader checks when shared dataset behavior, dependencies or MPI
  execution are affected. For grid configuration round trips, the existing
  two-rank check is:
  `mpiexec -n 2 python -m unittest nes.tests.projections.test_grid_configuration_roundtrip`.
  Run it inside the selected environment and use the approved scheduler on HPC
  when required by section 2.2. Inspect other MPI tests for their rank and
  resource requirements before running them.
- Use small synthetic datasets and temporary output directories. Inspect
  tests and examples for external paths, output locations and data requirements
  before running them; do not launch large production processing as a check.
- Report which checks ran and their results. Explain missing dependencies,
  data, skipped checks and checks pending execution; do not claim they passed.

### 4.2 Code and documentation

- Preserve public imports from `nes`, including `Nes`, projection classes,
  `open_netcdf` and `create_nes`. Use the current internal paths
  (`nes.default_nes`, `nes.projections`, `nes.formats`, `nes.methods`);
  do not reintroduce obsolete package layouts.
- Reuse existing dataset, grid, I/O and numerical helpers before introducing
  parallel implementations or new dependencies.
- Document modules, classes and functions using NumPy-style docstrings.
  Include parameter types, defaults, return values, units, array shapes,
  exceptions and side effects where applicable. Explain collective MPI
  requirements and which ranks must call an operation.
- Limit code lines to 120 characters, docstrings and comments to 79, and
  Markdown paragraph lines to 100. Allow exceptions for URLs, Markdown tables,
  code examples and literal content that cannot reasonably be wrapped.
- Keep code, comments, changelogs and project documentation in English;
  respond to the user in the language they use.
- Preserve scientific conventions: units, CRS and CF projection metadata,
  native coordinate order and bounds, calendars and time handling, missing
  values, masks and numerical precision. Explain and validate changes to them.
- For interpolation changes, distinguish extensive quantities from fluxes per
  unit area. Validate conservation, area normalization, uncovered cells and
  proxy fallback behavior where relevant; use justified numerical tolerances.
- Preserve MPI decomposition and rank-local/global array semantics. Collective
  calls must be reached consistently by participating ranks; avoid deadlocks,
  unnecessary full-array gathers and rank-dependent scientific results.
- Preserve NetCDF round-trip behavior, model-format metadata and versioned
  grid configuration or interpolation-cache compatibility. Explain and test
  intentional compatibility changes.
- Update relevant examples, CLI help and user documentation when public
  behavior changes. Edit Sphinx sources in `docs/source/`; modify the separate
  wiki repository only when relevant to the task and available locally.
- Read `CONTRIBUTING.md` for the issue and review workflow. Git actions remain
  subject to section 2.5 and explicit user instructions; a documented workflow
  does not itself authorize staging, commits, merges or publication.

### 4.3 Changelogs and releases

- Record user-facing developments and bug fixes in `CHANGELOG.md`. Inspect
  existing headings to identify the appropriate upcoming release; preserve
  historical entries and do not invent skipped releases.
- When preparing a release, inspect actual changes against the requested
  base branch and existing changelog entries. Do not infer changes solely
  from commit subjects or assume a release branch without checking.
- Keep `nes/__init__.py` (`__version__`, `__date__`), the Makefile `version`
  and the changelog version and release date consistent. `setup.py` reads
  the package version from `nes/__init__.py`; do not duplicate it there.
- Use the requested release date, or today's date if omitted. Preserve each
  file's existing date format.
- NES currently has one `CHANGELOG.md`; do not assume the HERMES_GR internal
  changelog or its release conventions exist here.
- Inspect `.gitlab-ci.yml` and relevant `.github/` workflows before proposing
  release automation changes. Release tags can trigger external publication;
  do not create tags or trigger publication without explicit approval.
- Preparing a release means editing and checking the release files. Apply
  section 2.5: this does not authorize commits, tags, pushes, merges or remote
  release publication. Finish by showing the changes and validation results.
