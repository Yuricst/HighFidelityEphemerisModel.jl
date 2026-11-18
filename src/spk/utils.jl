"""Utility functions for SPK generation"""

using Printf: @printf


# =============================================================================
# Small internal helpers
# =============================================================================

"""
    _get_et0(parameters)

Reference epoch `parameters.et0`, used by the methods that do not take `et0`
explicitly.
"""
function _get_et0(parameters)
    hasproperty(parameters, :et0) || error("No `et0` argument was provided and `parameters` has no `et0` property. Pass `et0` explicitly.")
    return Float64(parameters.et0)
end

"""
    _resolve_body_code(body; label)

Integer SPICE code of `body`, given either as an integer ID or as a name/numeric
string known to SPICE (resolved with `bods2c`).
"""
_resolve_body_code(body::Integer; label::AbstractString) = Int(body)

function _resolve_body_code(body::AbstractString; label::AbstractString)
    code = SPICE.bods2c(String(body))
    code === nothing && error("Could not resolve `$(label)` = \"$(body)\" to a SPICE ID. Pass an integer ID instead.")
    return Int(code)
end

_resolve_body_code(body; label::AbstractString) =
    error("`$(label)` must be an integer SPICE ID or a SPICE body name, got $(repr(body)).")

"""
    _default_center(parameters)

Default SPK center: the central body of the dynamics, `parameters.naif_ids[1]`.
"""
function _default_center(parameters)
    if hasproperty(parameters, :naif_ids) && !isempty(parameters.naif_ids)
        return string(first(parameters.naif_ids))
    end
    error("`center_id` was not given and `parameters` has no `naif_ids` to take the central body from. Pass `center_id`.")
end

"""
    _default_ref_frame(parameters)

Default SPK reference frame: the integration frame `parameters.naif_frame`
(states of the Cartesian EOMs are expressed in it), else `"J2000"`.

`"ICRF"` (accepted by the Ephemerides.jl backend) is written as `"J2000"`, as in
the backend itself. `"EME2000"` has no SPICE equivalent and is rejected: writing
it as `"J2000"` would silently drop the frame bias.
"""
function _default_ref_frame(parameters)
    hasproperty(parameters, :naif_frame) || return "J2000"
    frame = strip(String(parameters.naif_frame))
    uppercase(frame) == "ICRF" && return "J2000"
    uppercase(frame) == "EME2000" && error("`parameters.naif_frame` is \"EME2000\", which is not a SPICE frame. Pass `ref_frame_name` explicitly.")
    return String(frame)
end

function _body_name(code::Integer)
    name = SPICE.bodc2n(Int(code))
    return name === nothing ? nothing : String(name)
end

# `parameters.name` if that property exists, else `nothing`. Parameters may be
# the package's parameter structs or duck-typed objects such as NamedTuples.
_prop(obj, name::Symbol) = hasproperty(obj, name) ? getproperty(obj, name) : nothing


function _display_path(path)
    path === nothing && return ""
    s = String(path)
    try
        return relpath(s, pwd())
    catch
        return s
    end
end

function _summary_get(d, key, default = nothing)
    d === nothing && return default
    try
        return get(d, key, default)
    catch
        return default
    end
end

function _print_spk_pipeline_summary(; output_spk, segment_count, maneuver_txt = nothing,
    trajectory_maneuver_txt = nothing, metadata_json = nothing, ocp_control_summary = nothing)

    println()
    println("SPK generation complete")
    println("  segments                    : ", segment_count)
    println("  SPK                         : ", _display_path(output_spk))
    maneuver_txt !== nothing && println("  maneuvers (OCP controls)    : ", _display_path(maneuver_txt))
    trajectory_maneuver_txt !== nothing && println("  trajectory-jump maneuvers   : ", _display_path(trajectory_maneuver_txt))
    metadata_json !== nothing && println("  metadata JSON               : ", _display_path(metadata_json))

    ocp_scalar = _summary_get(ocp_control_summary, "total_control_scalar_mps")
    if ocp_scalar !== nothing
        @printf("  total control (OCP scalar)  : %.6e m/s\n", ocp_scalar)
    end

    return nothing
end

function _print_progress(label::AbstractString, i::Integer, n::Integer; enabled::Bool = true, width::Integer = 28)
    enabled || return nothing
    n <= 0 && return nothing

    filled = clamp(round(Int, width * i / n), 0, width)
    bar = repeat("=", filled) * repeat(" ", width - filled)
    print("\r", rpad(label, 22), " [", bar, "] ", lpad(string(i), length(string(n))), "/", n)

    if i >= n
        println()
    end
    flush(stdout)
    return nothing
end

function _json_safe(x)
    x === nothing && return nothing
    x isa Missing && return nothing
    x isa AbstractString && return String(x)
    x isa Symbol && return String(x)
    x isa Bool && return x
    x isa Integer && return x
    x isa AbstractFloat && return isfinite(x) ? Float64(x) : string(x)

    if x isa NamedTuple || x isa AbstractDict
        return Dict{String,Any}(string(k) => _json_safe(v) for (k, v) in pairs(x))
    elseif x isa Tuple
        return [_json_safe(v) for v in x]
    elseif x isa AbstractVector
        return [_json_safe(v) for v in x]
    elseif x isa AbstractMatrix
        return [[_json_safe(x[i, j]) for j in axes(x, 2)] for i in axes(x, 1)]
    else
        return string(x)
    end
end

function _write_json_value(io, x, indent::Integer)
    pad = repeat(" ", indent)
    nextpad = repeat(" ", indent + 2)

    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        if isfinite(x)
            print(io, x)
        else
            _write_json_string(io, string(x))
        end
    elseif x isa AbstractString
        _write_json_string(io, x)
    elseif x isa AbstractDict
        items = collect(pairs(x))
        sort!(items; by = kv -> string(kv.first))

        println(io, "{")
        for (idx, kv) in enumerate(items)
            print(io, nextpad)
            _write_json_string(io, string(kv.first))
            print(io, ": ")
            _write_json_value(io, kv.second, indent + 2)
            idx < length(items) && print(io, ",")
            println(io)
        end
        print(io, pad, "}")
    elseif x isa AbstractVector
        if isempty(x)
            print(io, "[]")
        else
            println(io, "[")
            for (idx, v) in enumerate(x)
                print(io, nextpad)
                _write_json_value(io, v, indent + 2)
                idx < length(x) && print(io, ",")
                println(io)
            end
            print(io, pad, "]")
        end
    else
        _write_json_string(io, string(x))
    end

    return nothing
end

function _write_json_string(io, s::AbstractString)
    escaped = replace(String(s),
        "\\" => "\\\\",
        "\"" => "\\\"",
        "\n" => "\\n",
        "\r" => "\\r",
        "\t" => "\\t",
    )
    print(io, "\"", escaped, "\"")
    return nothing
end
