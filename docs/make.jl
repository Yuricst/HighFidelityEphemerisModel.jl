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
            "Jacobians & Hessians" => "tutorials/jacobians_hessians.md",
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