"""
Run the code cells of the example notebooks, top to bottom, without Jupyter.

Usage (from the repository root):

    julia --project=examples examples/run_notebooks.jl                         # all notebooks
    julia --project=examples examples/run_notebooks.jl examples/basics.ipynb   # selected notebooks

Each notebook is run in a fresh module with the notebook's directory as working
directory, mimicking a Jupyter kernel (including soft global scope). The script
exits with a non-zero status if any cell throws.
"""

import JSON
import REPL

cell_source(x) = x isa AbstractString ? String(x) : join(x)

function run_notebook(nb_path::AbstractString)
    nb_path = abspath(nb_path)
    notebook = JSON.parsefile(nb_path)
    # (the module name must not clash with names used in the notebook)
    mod = Module(Symbol("Notebook_", splitext(basename(nb_path))[1]))
    code_cells = [cell_source(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    cd(dirname(nb_path)) do
        for (i, src) in enumerate(code_cells)
            println("── $(basename(nb_path)): cell $i/$(length(code_cells))")
            result = Base.include_string(REPL.softscope, mod, src, "$(nb_path)#cell$(i)")
            # display the value of the last expression, as Jupyter does
            if result !== nothing && !REPL.ends_with_semicolon(src)
                Base.invokelatest(show, stdout, MIME"text/plain"(), result)
                println()
            end
        end
    end
    # unload SPICE kernels so that notebooks do not depend on each other
    isdefined(mod, :kclear) && Base.invokelatest(mod.kclear)
    return nothing
end

notebooks = isempty(ARGS) ?
    filter(f -> endswith(f, ".ipynb"), readdir(@__DIR__; join = true)) :
    ARGS

failed = String[]
for nb in notebooks
    try
        @time run_notebook(nb)
    catch err
        @error "Notebook $(nb) failed" exception = (err, catch_backtrace())
        push!(failed, nb)
    end
end

if isempty(failed)
    println("\nAll $(length(notebooks)) notebook(s) ran successfully.")
else
    println("\nFailed notebooks:\n  ", join(failed, "\n  "))
    exit(1)
end
