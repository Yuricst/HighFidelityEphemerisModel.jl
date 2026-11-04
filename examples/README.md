# Examples

The Jupyter notebooks in this folder are the tutorials of the
[documentation](https://yuricst.github.io/HighFidelityEphemerisModel.jl/).
The tutorial pages are generated from these notebooks when the docs are built.

| Notebook | Content |
|----------|---------|
| [`basics.ipynb`](basics.ipynb) | Setting up parameters, solving an IVP, and propagating the STM |
| [`perturbations.ipynb`](perturbations.ipynb) | Third-body gravity, spherical harmonics, SRP, drag, and the different backends |
| [`gve.ipynb`](gve.ipynb) | Gauss variational equations: propagation in modified equinoctial, Keplerian, and equinoctial elements |
| [`jacobians_hessians.ipynb`](jacobians_hessians.ipynb) | Finite-difference Jacobians & Hessians of the dynamics |
| [`ode_sol_to_spk.ipynb`](ode_sol_to_spk.ipynb) | Writing ODE solutions to SPICE SPK (BSP) files, with maneuver & metadata sidecars |


## SPICE kernels

The notebooks load SPICE kernels from the directory given by the `SPICE` environment
variable, using the following layout:

```
$SPICE/
├── lsk/naif0012.tls
├── spk/de440.bsp
├── pck/gm_de440.tpc
├── pck/pck00011.tpc
├── pck/moon_pa_de440_200625.bpc
└── fk/moon_de440_250416.tf
```

All of these kernels are available from the
[NAIF generic kernels](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/).
Alternatively, edit `spice_dir` in the notebooks to point to your kernels.


## Running the notebooks

The first code cell of each notebook activates the environment defined by
`examples/Project.toml`, which uses the local copy of `HighFidelityEphemerisModel.jl`.

To open the notebooks with Jupyter, install [IJulia](https://github.com/JuliaLang/IJulia.jl)
(e.g. in your global environment) and launch Jupyter from the repository root:

```julia
julia> using IJulia
julia> notebook(dir = "examples")   # or jupyterlab(dir = "examples")
```

The notebooks can also be opened directly in VS Code with the Julia and Jupyter extensions.

To run all notebooks from the command line without Jupyter (e.g. to check that the
examples still work after changing the package), from the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/run_notebooks.jl                          # all notebooks
julia --project=examples examples/run_notebooks.jl examples/basics.ipynb    # selected notebooks
```


## Editing the notebooks

The notebooks are converted to Documenter pages by [`docs/notebooks.jl`](../docs/notebooks.jl).
To keep them rendering nicely both in Jupyter and in the docs:

- Write admonitions as blockquotes starting with `**Note**`, `**Tip**`, `**Warning**`,
  `**Danger**`, `**Info**`, or `**Compat**`, e.g.
  ```markdown
  > **Note**
  >
  > Some note.
  ```
  These become `!!! note` admonitions in the docs.
- Write display math between `$$` lines; it becomes a ```` ```math ```` block in the docs.
- Use relative links (e.g. `[Basics](basics.ipynb)` or `[Overview](../docs/src/overview.md)`);
  they are rewritten to point to the corresponding docs page.
- Tag a cell with `hide-in-docs` to leave it out of the docs, or with `hide-output`
  to leave out its outputs.
- Code cells are not executed during the docs build. Outputs saved in a notebook are
  shown on the docs page, so clear the outputs before committing unless you want them
  displayed.

After editing, rebuild the docs with `julia --project=docs docs/make.jl` and check the result in `docs/build`.
