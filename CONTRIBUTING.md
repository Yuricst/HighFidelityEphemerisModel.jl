# Contributing to HighFidelityEphemerisModel.jl

Thank you for considering contributing to `HighFidelityEphemerisModel.jl`!
Contributions of all kinds are welcome: bug reports, feature requests,
documentation improvements, new examples, and code.

This document describes how to get set up and what we expect from contributions.
If anything here is unclear, feel free to open an issue and ask.

## Code of conduct

We follow the [Julia Community Standards](https://julialang.org/community/standards/).
Please be respectful and constructive in all interactions in issues, pull requests,
and discussions.

## Reporting bugs

Before opening an issue, please search the
[existing issues](https://github.com/Yuricst/HighFidelityEphemerisModel.jl/issues)
to check whether it has already been reported.

A good bug report includes:

- a short description of the problem and the expected behavior;
- a **minimal working example** (MWE) that reproduces the issue, including the
  parameter construction (`SpiceParameters` / `EphemeridesParameters`) and the
  `ODEProblem` / `solve` call where relevant;
- the SPICE kernels (or ephemeris files) used, by filename;
- the full error message and stack trace, if any;
- the output of `versioninfo()` and `] status` (or `Pkg.status()`).

## Suggesting features

Feature requests are welcome as issues. Please describe the use case (e.g. the
dynamical model, force, or element set you need) and, where possible, point to a
reference (paper, textbook, or mission documentation) that defines the model.
For larger changes, such as a new force model, a new backend, or a change to a
public interface, please open an issue to discuss the design **before** starting
on a pull request.

Note that the package scope is to provide equations of motion and related
utilities compatible with the `OrdinaryDiffEq.jl` ecosystem; integration schemes
themselves are out of scope.

## Development setup

1. Fork the repository on GitHub and clone your fork:

   ```bash
   git clone git@github.com:<your-username>/HighFidelityEphemerisModel.jl.git
   cd HighFidelityEphemerisModel.jl
   ```

2. Instantiate the package environment:

   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

3. To use your local copy from another environment, `dev` it:

   ```julia-repl
   ] dev /path/to/HighFidelityEphemerisModel.jl
   ```

   We recommend using [Revise.jl](https://github.com/timholy/Revise.jl) so that
   changes to the source are picked up without restarting Julia.

The minimum supported Julia version is given by the `julia` entry in the
`[compat]` section of `Project.toml`; CI runs on the Julia LTS release.

## Running the tests

The test suite requires several SPICE kernels. These are **not** distributed with
the repository and must be downloaded once. The tests look for them in one of two
places:

- if the `SPICE` environment variable is set, kernels are loaded from
  `$SPICE/lsk`, `$SPICE/spk`, `$SPICE/pck`, `$SPICE/fk` and
  `$SPICE/misc/dsg_naif` (see `test/utils.jl` for the exact paths);
- otherwise, kernels are loaded from `spice/test/` at the root of the repository.

To populate `spice/test/` (this mirrors what CI does in
`.github/workflows/test.yml`):

```bash
mkdir -p spice/test && cd spice/test
NAIF=https://naif.jpl.nasa.gov/pub/naif
curl -LO $NAIF/generic_kernels/lsk/naif0012.tls
curl -LO $NAIF/generic_kernels/spk/planets/de440.bsp
curl -LO $NAIF/generic_kernels/pck/gm_de440.tpc
curl -LO $NAIF/generic_kernels/pck/moon_pa_de440_200625.bpc
curl -LO $NAIF/generic_kernels/pck/earth_latest_high_prec.bpc
curl -LO $NAIF/generic_kernels/pck/pck00011.tpc
curl -LO $NAIF/generic_kernels/fk/satellites/moon_de440_250416.tf
curl -LO $NAIF/misc/MORE_PROJECTS/DSG/receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp
cd ../..
```

Please do not commit SPICE kernels to the repository.

Then run the tests from the repository root:

```julia-repl
] activate .
] test
```

or from the shell:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Some tests exercise `EnsembleThreads` and `Threads.@threads`; CI runs them with
four threads. To do the same locally, start Julia with `--threads=4` or set
`JULIA_NUM_THREADS=4`.

## Making changes

### Workflow

1. Create a branch off `main` with a descriptive name, e.g.
   `fix-srp-shadow-model` or `42-add-gve-callback` (prefixing the issue number
   is encouraged).
2. Make focused commits with clear messages. Keep unrelated changes in separate
   pull requests.
3. Add or update tests and documentation alongside code changes (see below).
4. Make sure the test suite passes locally.
5. Open a pull request against `main`, describing **what** changed and **why**,
   and link any related issue (e.g. `Closes #42`).

All pull requests are run through CI (tests on Linux, macOS and Windows, plus a
documentation build), and must pass before being merged. A maintainer will review
your pull request; please be prepared to iterate on feedback.

### Tests

- Every bug fix should come with a test that fails without the fix.
- Every new feature should be covered by tests. Tests live in `test/`, and new
  test files must be `include`d from `test/runtests.jl` in the relevant
  `@testset`.
- Where possible, validate dynamics against an independent reference (e.g. SPICE
  ephemerides, a published value, or a different backend/formulation) rather
  than only checking that code runs.
- Keep test runtimes reasonable; prefer short propagation spans and low-degree
  spherical harmonics unless a longer case is essential.

### Code style

- Follow the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/):
  4-space indentation, `snake_case` for functions and variables, `CamelCase`
  for types and modules, and a trailing `!` for functions that mutate their
  arguments (e.g. in-place equations of motion `eom_*!`).
- Keep equations of motion type-generic (avoid hard-coding `Float64`) so that
  they remain compatible with automatic differentiation such as `ForwardDiff.jl`.
- Avoid allocations and type instabilities in performance-critical code paths
  such as the equations of motion; the scripts in `benchmark/` can help check
  for regressions.
- Code reachable from the `EphemeridesParameters` backend must remain
  thread-safe; SPICE calls are not thread-safe and should only be used through
  the SPICE backend.
- Prefer clarity over cleverness, and state units and reference frames in
  variable names, comments, or docstrings (e.g. canonical `DU`/`TU` versus km/s).

### Dependencies

The package strives to keep its dependency list minimal. Please discuss in an
issue before adding a new dependency to `Project.toml`. If a new dependency is
added, include a corresponding `[compat]` entry.

## Documentation

- All exported functions and types should have a docstring following the
  [Julia documentation conventions](https://docs.julialang.org/en/v1/manual/documentation/),
  starting with the signature in an indented code block, followed by a description
  of arguments, keyword arguments, and return values (including units).
- New public API should be added to the appropriate page in `docs/src/`
  (typically `docs/src/api.md`).
- The tutorials are the Jupyter notebooks in `examples/`; they are converted into
  documentation pages at build time by `docs/notebooks.jl`. Edit the notebooks,
  not the generated files under `docs/src/tutorials/`.

To build the documentation locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The generated site is written to `docs/build/`.

To check that the example notebooks still run end to end (requires the `SPICE`
environment variable to point to a kernel directory, see `examples/README.md`):

```bash
julia --project=examples examples/run_notebooks.jl
```

## Changelog

User-facing changes (new features, bug fixes, deprecations, and especially
breaking changes) should be recorded in `NEWS.md` under the `Unreleased`
section. For breaking changes, please describe how users should migrate.

## Versioning and releases

This package follows [Semantic Versioning](https://semver.org/). While the major
version is `0`, a minor version bump (e.g. `0.3` → `0.4`) indicates a breaking
change. Releases are made by the maintainers: the version in `Project.toml` is
bumped, the package is registered in the Julia General registry through
[Registrator](https://github.com/JuliaRegistries/Registrator.jl), and
[TagBot](https://github.com/JuliaRegistries/TagBot) creates the corresponding Git
tag and GitHub release. Contributors do not need to bump the version in their
pull requests.

## License

By contributing to `HighFidelityEphemerisModel.jl`, you agree that your
contributions will be licensed under the [MIT License](LICENSE) of this project.
