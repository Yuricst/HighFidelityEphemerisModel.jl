"""
Make documentation with Documenter.jl
"""

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

# const SRC = normpath(joinpath(@__DIR__, "..", "src"))
# include(joinpath(SRC, "HighFidelityEphemerisModel.jl"))

using HighFidelityEphemerisModel
using Documenter
using Literate

# Generate the tutorial pages from the Jupyter notebooks in `examples/` (not executed)
include(joinpath(@__DIR__, "notebooks.jl"))
convert_notebooks(
    joinpath(@__DIR__, "..", "examples"),
    joinpath(@__DIR__, "src", "tutorials");
    repo_url = "https://github.com/Yuricst/HighFidelityEphemerisModel.jl",
    branch = "main",
)

# Generate the example pages from the Literate.jl scripts in `docs/src/examples/`. Their code
# is run by Documenter while building the docs (`@example` blocks), which needs the SPICE
# kernels in `ENV["SPICE"]`; start Julia with several threads to run the ensembles in parallel.
include(joinpath(@__DIR__, "plotly_figures.jl"))   # `docs_figure`, used by the examples
const EXAMPLES_DIR = joinpath(@__DIR__, "src", "examples")
const EXAMPLES = ["nrho_maneuver_dispersion", "earth_flyby_aerobraking"]
for name in EXAMPLES
    Literate.markdown(
        joinpath(EXAMPLES_DIR, name * ".jl"), EXAMPLES_DIR;
        flavor = Literate.DocumenterFlavor(),
        credit = false,
    )
end

makedocs(
    clean = true,
    build = joinpath(@__DIR__, "build"),
	modules  = [HighFidelityEphemerisModel],
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "HighFidelityEphemerisModel.jl",
    # options
    pages = [
		"Home" => "index.md",
        "Overview" => "overview.md",
        "Tutorials" => Any[
            "Basics" => "tutorials/basics.md",
            "Perturbations" => "tutorials/perturbations.md",
            "Gauss variational equations" => "tutorials/gve.md",
            "Jacobians & Hessians" => "tutorials/jacobians_hessians.md",
            "Generating SPICE SPK file" => "tutorials/ode_sol_to_spk.md",
        ],
        "Examples" => Any[
            "Maneuver dispersions on an NRHO" => "examples/nrho_maneuver_dispersion.md",
            "Aerobraking during an Earth flyby" => "examples/earth_flyby_aerobraking.md",
        ],
        "API" => "api.md",
		# "API" => Any[
		# 	"Core" => "api/api_core.md",
		# 	# "Problem Constructor" => "api/api_create_sft_problem.md",
		# 	# "Core Routines" => "api/api_core.md",
		# 	# "Sims-Flanagan Transcription" => "api/api_simsflanagan.md",
		# 	# "Plotting" => "api/api_plot.md",
		# ],
    ],
)