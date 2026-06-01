# SPK utility helpers.

using Printf: @printf


# =============================================================================
# Small internal helpers
# =============================================================================

function _get_et0_from_parameters(parameters)
    return _property_first(parameters, (:et0, :ET0, :epoch0, :epoch_et0, :et_start, :t0_et))
end

function _property_first(obj, names::Tuple)
    obj === nothing && return nothing
    for name in names
        try
            if hasproperty(obj, name)
                return getproperty(obj, name)
            end
        catch
            # Some objects define unusual property access. Skip and try the next alias.
        end
    end
    return nothing
end

function _add_if_found!(dict::AbstractDict, key::AbstractString, obj, names::Tuple)
    value = _property_first(obj, names)
    value === nothing && return dict
    dict[key] = _json_safe(value)
    return dict
end


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

function _print_spk_pipeline_summary(; output_spk, maneuver_txt, ocp_maneuver_txt, metadata_json,
    segment_count, intermediate_dir, maneuver_summary, ocp_control_summary)

    println()
    println("SPK generation complete")
    println("  segments              : ", segment_count)
    println("  SPK                   : ", _display_path(output_spk))
    maneuver_txt !== nothing && println("  trajectory maneuvers  : ", _display_path(maneuver_txt))
    ocp_maneuver_txt !== nothing && println("  OCP maneuvers         : ", _display_path(ocp_maneuver_txt))
    metadata_json !== nothing && println("  metadata JSON         : ", _display_path(metadata_json))
    intermediate_dir !== nothing && println("  intermediates         : ", _display_path(intermediate_dir))

    jump_total = _summary_get(maneuver_summary, "total_delta_v_mps")
    if jump_total !== nothing
        @printf("  Δv from traj. jumps   : %.6e m/s
", jump_total)
    end

    ocp_scalar = _summary_get(ocp_control_summary, "total_control_scalar_mps")
    if ocp_scalar !== nothing
        @printf("  Δv from OCP scalar    : %.6e m/s
", ocp_scalar)
    end

    ocp_vec = _summary_get(ocp_control_summary, "total_control_vector_norm_mps")
    if ocp_vec !== nothing
        @printf("  Δv from OCP vectors   : %.6e m/s
", ocp_vec)
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

    if x isa NamedTuple
        return Dict(string(k) => _json_safe(v) for (k, v) in pairs(x))
    elseif x isa AbstractDict
        return Dict(string(k) => _json_safe(v) for (k, v) in pairs(x))
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


# =============================================================================
# Utility helpers
# =============================================================================

function list_segment_state_files(states_dir::AbstractString)
    files = filter(f -> occursin(r"^seg_\d{3}_states\.txt$", f), readdir(states_dir))
    sort!(files)
    return [joinpath(states_dir, f) for f in files]
end

function list_segment_setup_files(setup_dir::AbstractString)
    files = filter(f -> occursin(r"^seg_\d{3}_setup\.txt$", f), readdir(setup_dir))
    sort!(files)
    return [joinpath(setup_dir, f) for f in files]
end

function read_segment_boundary_epochs(states_dir::AbstractString)
    boundaries = Float64[]
    for path in list_segment_state_files(states_dir)
        lines = readlines(path)
        data_lines = lines[2:end]
        isempty(data_lines) && continue

        first_fields = split(first(data_lines), ",")
        last_fields  = split(last(data_lines), ",")

        et_start = parse(Float64, strip(first_fields[1]))
        et_end   = parse(Float64, strip(last_fields[1]))

        push!(boundaries, et_start)
        push!(boundaries, et_end)
    end
    return sort(unique(boundaries))
end

function read_segment_boundary_epochs(states_dir::AbstractString, nseg::Integer)
    boundaries = Float64[]
    for seg_idx in 1:nseg
        tag = lpad(string(seg_idx), 3, '0')
        path = joinpath(states_dir, "seg_$(tag)_states.txt")
        isfile(path) || error("Missing states file: $path")

        lines = readlines(path)
        data_lines = lines[2:end]
        isempty(data_lines) && continue

        first_fields = split(first(data_lines), ",")
        last_fields  = split(last(data_lines), ",")

        et_start = parse(Float64, strip(first_fields[1]))
        et_end   = parse(Float64, strip(last_fields[1]))

        push!(boundaries, et_start)
        push!(boundaries, et_end)
    end
    return sort(unique(boundaries))
end

function _make_spk_pipeline_workdir(output_spk_abs::AbstractString, intermediate_parent_dir)
    parent = intermediate_parent_dir === nothing ? dirname(output_spk_abs) : abspath(String(intermediate_parent_dir))
    mkpath(parent)
    return mktempdir(parent; prefix = "scp_to_spk_")
end

function _mkspk_path(path::AbstractString)
    # MKSPK setup files are easier to read and usually safer with forward slashes.
    return replace(String(path), "\\" => "/")
end