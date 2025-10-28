"""
Make documentation with Documenter.jl
"""

using Documenter

include(joinpath(dirname(@__FILE__), "../src/HighFidelityEphemerisModel.jl"))


makedocs(
    clean = false,
    build = dirname(@__FILE__),
	modules  = [HighFidelityEphemerisModel],
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "HighFidelityEphemerisModel.jl",
    # options
    pages = [
		"Home" => "index.md",
        "Overview" => "overview.md",
        "Tutorials" => Any[
            "Basics" => "tutorials/basics.md",
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