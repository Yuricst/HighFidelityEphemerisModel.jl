"""
Display PlotlyBase figures as interactive figures in the pages generated with Literate.jl.

The example scripts in `docs/src/examples/` end each figure block with the line

    #md Main.docs_figure(fig) #hide

which only exists in the generated markdown (in the scripts it is a comment) and is hidden
on the page. `docs_figure` writes the figure data to the RequireJS module
`plotly-figure-<n>.js` in the working directory of the `@example` block (the directory of
the page in `docs/build`) and returns an object displayed as HTML that loads plotly.js and
the figure with RequireJS, which Documenter already uses. Keeping the data out of the page
keeps it out of the search index and the HTML size limits of Documenter.
"""

using PlotlyBase

const PLOTLY_JS_URL = "https://cdn.plot.ly/plotly-2.33.0.min"   # without ".js" (RequireJS path)
const DOCS_BUILD_DIR = joinpath(@__DIR__, "build")
const _DOCS_FIGURE_COUNTER = Ref(0)

struct DocsFigure
    name::String       # id of the HTML element and base name of the data module
    path::String       # path of the data module relative to `docs/build`, with '/'
    height::Int
end

function docs_figure(fig::PlotlyBase.Plot)
    name = "plotly-figure-$(_DOCS_FIGURE_COUNTER[] += 1)"
    open(name * ".js", "w") do io
        print(io, "define(function () { return ", PlotlyBase.JSON.json(fig), "; });\n")
    end
    path = join(splitpath(relpath(joinpath(pwd(), name * ".js"), DOCS_BUILD_DIR)), "/")
    height = something(get(fig.layout.fields, :height, nothing), 450)
    return DocsFigure(name, path, height)
end

function Base.show(io::IO, ::MIME"text/html", f::DocsFigure)
    # `documenterBaseURL` (the relative path to the root of the docs, set by Documenter on
    # every page) makes the path work with and without `prettyurls`
    print(io, """
    <div id="$(f.name)" class="plotly-figure" style="width:100%;height:$(f.height)px;"></div>
    <script>
    require.config({paths: {plotly: "$(PLOTLY_JS_URL)"}});
    require(["plotly", documenterBaseURL + "/$(f.path)"], function (Plotly, figure) {
        Plotly.newPlot("$(f.name)", figure.data, figure.layout, {responsive: true});
    });
    </script>
    """)
end
