"""
Convert the Jupyter notebooks in `examples/` into Markdown pages for Documenter.jl.

The notebooks are the source of truth for the tutorials. Each notebook is
converted into `docs/src/tutorials/<name>.md` at build time (these generated
files are not tracked by git). The conversion:

- turns markdown cells into Documenter markdown:
    - blockquotes starting with `**Note**`, `**Tip**`, `**Warning**`, `**Danger**`,
      `**Info**`, or `**Compat**` become admonitions (`!!! note`, ...),
    - `\$\$ ... \$\$` display math becomes a ```` ```math ```` block,
    - relative links are rewritten so that they resolve from the generated page
      (links to other converted notebooks point to their generated page, links
      to files outside `docs/src` point to the GitHub repository);
- turns code cells into (non-executed) ```` ```julia ```` blocks;
- renders stored cell outputs (text, PNG, SVG), if the notebook was saved with any;
- skips cells tagged `hide-in-docs`, and the outputs of cells tagged `hide-output`.
"""

using JSON
using Base64

const ADMONITION_REGEX = r"^\*\*(Note|Tip|Warning|Danger|Info|Compat)\*\*:?\s*(.*)$"i
const MARKDOWN_LINK_REGEX = r"\]\(([^)\s]+)\)"
const ANSI_REGEX = r"\e\[[0-9;]*[A-Za-z]"


_cell_source(x) = x isa AbstractString ? String(x) : join(x)

function _has_tag(cell, tag)
    metadata = get(cell, "metadata", nothing)
    metadata === nothing && return false
    return tag in get(metadata, "tags", String[])
end

_is_fence(line) = startswith(lstrip(line), "```") || startswith(lstrip(line), "~~~")


"""
Rewrite a relative link `target` found in notebook `nb_path` so that it resolves
from the generated markdown file `md_path`.
"""
function _rewrite_link(target, nb_path, md_path, link_map, docs_src, repo_root, repo_url, branch)
    if occursin(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) || startswith(target, "#") || startswith(target, "@")
        return target   # absolute URL, in-page anchor, or Documenter @ref
    end
    path, anchor = occursin('#', target) ? (split(target, '#'; limit = 2)...,) : (target, "")
    anchor = isempty(anchor) ? "" : "#" * anchor
    resolved = normpath(joinpath(dirname(nb_path), path))
    resolved = get(link_map, resolved, resolved)
    if startswith(resolved, docs_src)
        return relpath(resolved, dirname(md_path)) * anchor
    else
        return repo_url * "/blob/" * branch * "/" * relpath(resolved, repo_root) * anchor
    end
end


"""
Convert the content of a notebook markdown cell into Documenter markdown.
"""
function _convert_markdown(src::AbstractString, rewrite_link)
    lines = split(src, '\n')
    out = String[]
    in_fence = false
    i = 1
    while i <= length(lines)
        line = lines[i]
        if in_fence
            _is_fence(line) && (in_fence = false)
            push!(out, line)
        elseif _is_fence(line)
            in_fence = true
            push!(out, line)
        elseif startswith(line, ">")
            # blockquote: collect all consecutive quoted lines
            block = String[]
            while i <= length(lines) && startswith(lines[i], ">")
                push!(block, replace(lines[i], r"^> ?" => ""))
                i += 1
            end
            m = match(ADMONITION_REGEX, strip(block[1]))
            if m === nothing
                append!(out, ">" .* ifelse.(isempty.(block), "", " " .* block))
            else
                body = isempty(m.captures[2]) ? block[2:end] : [m.captures[2]; block[2:end]]
                while !isempty(body) && isempty(strip(first(body)))
                    popfirst!(body)
                end
                push!(out, "!!! " * lowercase(m.captures[1]))
                push!(out, "")
                for l in split(_convert_markdown(join(body, "\n"), rewrite_link), '\n')
                    push!(out, isempty(l) ? "" : "    " * l)
                end
            end
            continue
        elseif strip(line) == "\$\$"
            # multi-line display math
            push!(out, "```math")
            i += 1
            while i <= length(lines) && strip(lines[i]) != "\$\$"
                push!(out, lines[i])
                i += 1
            end
            push!(out, "```")
        elseif (s = strip(line); length(s) > 4 && startswith(s, "\$\$") && endswith(s, "\$\$"))
            # single-line display math
            append!(out, ["```math", strip(s[3:end-2]), "```"])
        else
            push!(out, replace(line, MARKDOWN_LINK_REGEX => s -> "](" * rewrite_link(s[3:end-1]) * ")"))
        end
        i += 1
    end
    return join(out, "\n")
end


"""
Convert the stored outputs of a notebook code cell into markdown.
"""
function _convert_outputs(outputs, figure_prefix, md_dir, figure_counter)
    blocks = String[]
    plain(text) = push!(blocks, "```\n" * rstrip(replace(text, ANSI_REGEX => "")) * "\n```")
    for output in outputs
        otype = get(output, "output_type", "")
        if otype == "stream"
            plain(_cell_source(output["text"]))
        elseif otype in ("execute_result", "display_data")
            data = output["data"]
            if haskey(data, "image/png") || haskey(data, "image/svg+xml")
                figure_counter[] += 1
                ext = haskey(data, "image/png") ? "png" : "svg"
                name = "$(figure_prefix)-$(figure_counter[]).$(ext)"
                content = _cell_source(data[ext == "png" ? "image/png" : "image/svg+xml"])
                open(joinpath(md_dir, name), "w") do io
                    ext == "png" ? write(io, base64decode(content)) : write(io, content)
                end
                push!(blocks, "![]($(name))")
            elseif haskey(data, "text/plain")
                plain(_cell_source(data["text/plain"]))
            end
        elseif otype == "error"
            plain(string(get(output, "ename", "Error"), ": ", get(output, "evalue", "")))
        end
    end
    return blocks
end


"""
    notebook_to_markdown(nb_path, md_path; kwargs...)

Convert the Jupyter notebook `nb_path` into the Documenter markdown page `md_path`.
"""
function notebook_to_markdown(
    nb_path::AbstractString,
    md_path::AbstractString;
    link_map = Dict{String,String}(),
    docs_src::AbstractString = joinpath(@__DIR__, "src"),
    repo_root::AbstractString = normpath(joinpath(@__DIR__, "..")),
    repo_url::AbstractString,
    branch::AbstractString = "main",
)
    nb_path, md_path = abspath(nb_path), abspath(md_path)
    md_dir = dirname(md_path)
    mkpath(md_dir)
    notebook = JSON.parsefile(nb_path)
    nb_relpath = relpath(nb_path, repo_root)
    nb_url = repo_url * "/blob/" * branch * "/" * nb_relpath

    # remove figures generated by a previous build of this page
    figure_prefix = splitext(basename(md_path))[1]
    for f in readdir(md_dir)
        occursin(Regex("^" * figure_prefix * raw"-\d+\.(png|svg)$"), f) && rm(joinpath(md_dir, f))
    end
    figure_counter = Ref(0)

    rewrite_link(target) = _rewrite_link(
        target, nb_path, md_path, link_map, abspath(docs_src), repo_root, repo_url, branch,
    )

    blocks = String[
        "```@meta\nEditURL = \"$(relpath(nb_path, md_dir))\"\n```",
    ]
    banner = """
    !!! info "Jupyter notebook"

        This page is generated from the Jupyter notebook [`$(nb_relpath)`]($(nb_url)).
        Download it to run the code yourself; see the [examples README]($(repo_url)/blob/$(branch)/examples/README.md) for instructions."""
    banner_inserted = false

    for cell in notebook["cells"]
        _has_tag(cell, "hide-in-docs") && continue
        src = rstrip(_cell_source(cell["source"]))
        if cell["cell_type"] == "markdown"
            md = _convert_markdown(src, rewrite_link)
            if !banner_inserted && startswith(md, "# ")
                # place the banner right after the page title
                title, rest = occursin('\n', md) ? split(md, '\n'; limit = 2) : (md, "")
                push!(blocks, title, banner)
                isempty(strip(rest)) || push!(blocks, rest)
                banner_inserted = true
            else
                push!(blocks, md)
            end
        elseif cell["cell_type"] == "code"
            isempty(strip(src)) || push!(blocks, "```julia\n" * src * "\n```")
            if !_has_tag(cell, "hide-output")
                append!(blocks, _convert_outputs(get(cell, "outputs", []), figure_prefix, md_dir, figure_counter))
            end
        end
    end
    banner_inserted || insert!(blocks, 2, banner)

    open(md_path, "w") do io
        print(io, join(blocks, "\n\n"), "\n")
    end
    return md_path
end


"""
    convert_notebooks(nb_dir, out_dir; kwargs...)

Convert every `*.ipynb` notebook in `nb_dir` into `out_dir/<name>.md`.
Returns the list of generated markdown files.
"""
function convert_notebooks(nb_dir::AbstractString, out_dir::AbstractString; kwargs...)
    notebooks = filter(f -> endswith(f, ".ipynb"), readdir(abspath(nb_dir); join = true))
    link_map = Dict(
        nb => joinpath(abspath(out_dir), splitext(basename(nb))[1] * ".md") for nb in notebooks
    )
    return [notebook_to_markdown(nb, link_map[nb]; link_map = link_map, kwargs...) for nb in notebooks]
end
