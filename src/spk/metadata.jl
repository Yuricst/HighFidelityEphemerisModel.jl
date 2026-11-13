"""Metadata JSON helpers for generated SPK files"""


const _SPK_METADATA_SCHEMA = "HighFidelityEphemerisModel.SPKMetadata.v2"


"""
    build_spk_metadata(; kwargs...) -> Dict

Build the metadata dictionary of a generated SPK: product paths, SPK identifiers
and frame, sampling, scaling units, per-segment coverage, force model and
maneuver summaries.

`windows` is a vector of NamedTuples with `start_et`, `end_et` and optionally
`point_count`, `segment_id` and `coast_window`, one per SPK segment.
"""
function build_spk_metadata(;
    output_spk::AbstractString,
    metadata_json = nothing,
    spice_id = nothing,
    center_id = nothing,
    ref_frame_name = nothing,
    polynom_degree = nothing,
    frame_def_file = nothing,
    dt_sec = nothing,
    segment_gap_sec = nothing,
    windows = NamedTuple[],
    et0 = nothing,
    parameters = nothing,
    force_model_metadata = nothing,
    extra_metadata = nothing,
    trajectory_maneuver_file = nothing,
    maneuver_summary = nothing,
    maneuver_entries = nothing,
    ocp_control_file = nothing,
    ocp_control_summary = nothing,
    ocp_control_entries = nothing,
    merge_maneuvers_into_metadata::Bool = false,
)
    _path(p) = p === nothing ? nothing : abspath(String(p))

    coverage_windows = Any[]
    for (idx, w) in enumerate(windows)
        entry = _json_safe(w)
        entry["segment_index"] = idx
        push!(coverage_windows, entry)
    end

    time_base = Dict{String,Any}()
    et0 !== nothing && (time_base["et0"] = Float64(et0))
    if parameters !== nothing
        for (key, name) in (("TU_seconds", :TU), ("DU_km", :DU), ("VU_kmps", :VU))
            value = _prop(parameters, name)
            value !== nothing && (time_base[key] = _json_safe(value))
        end
    end

    force_model = infer_force_model_metadata(parameters)
    force_model_metadata !== nothing && _deep_merge_dicts!(force_model, _json_safe(force_model_metadata))

    maneuvers = Dict{String,Any}()
    if ocp_control_summary !== nothing
        # Commanded/executed controls (e.g. OCP `solution.u`): the primary maneuver product.
        ocp = _json_safe(ocp_control_summary)
        ocp["file"] = _path(ocp_control_file)
        ocp["primary_cost_key"] = haskey(ocp, "total_control_scalar_mps") ?
            "total_control_scalar_mps" : "total_control_vector_norm_mps"
        merge_maneuvers_into_metadata && (ocp["entries"] = _json_safe(something(ocp_control_entries, Any[])))
        maneuvers["ocp_control"] = ocp
    end
    if maneuver_summary !== nothing
        # Velocity jumps between adjacent coast arcs: a diagnostic reconstructed
        # from the state history, not necessarily the commanded control.
        jumps = _json_safe(maneuver_summary)
        jumps["file"] = _path(trajectory_maneuver_file)
        jumps["units"] = "m/s"
        jumps["format"] = "ETSECONDS,DVX_mps,DVY_mps,DVZ_mps"
        jumps["cost_convention"] = "sum(norm.(dv_i))"
        merge_maneuvers_into_metadata && (jumps["entries"] = _json_safe(something(maneuver_entries, Any[])))
        maneuvers["trajectory_jumps"] = jumps
    end

    meta = Dict{String,Any}(
        "schema" => _SPK_METADATA_SCHEMA,
        "generated_unix_time_sec" => time(),
        "products" => Dict{String,Any}(
            "spk_file" => _path(output_spk),
            "maneuver_file" => _path(ocp_control_file),
            "trajectory_jump_maneuver_file" => _path(trajectory_maneuver_file),
            "metadata_json" => _path(metadata_json),
        ),
        "spk" => Dict{String,Any}(
            "object_id" => spice_id,
            "object_name" => spice_id === nothing ? nothing : _body_name(spice_id),
            "center_id" => center_id,
            "center_name" => center_id === nothing ? nothing : _body_name(center_id),
            "ref_frame_name" => ref_frame_name,
            "spk_type" => 13,
            "polynom_degree" => polynom_degree,
            "frame_def_file" => _path(frame_def_file),
        ),
        "sampling" => Dict{String,Any}(
            "dt_sec" => dt_sec === nothing ? nothing : Float64(dt_sec),
            "segment_gap_sec" => segment_gap_sec === nothing ? nothing : Float64(segment_gap_sec),
            "segment_count" => length(windows),
        ),
        "time_base" => time_base,
        "coverage" => Dict{String,Any}(
            "start_et" => isempty(windows) ? nothing : minimum(w.start_et for w in windows),
            "end_et" => isempty(windows) ? nothing : maximum(w.end_et for w in windows),
            "windows" => coverage_windows,
        ),
        "force_model" => force_model,
        "maneuvers" => maneuvers,
    )
    extra_metadata !== nothing && (meta["extra"] = _json_safe(extra_metadata))

    return meta
end

"""
    write_spk_metadata_json(path, metadata)

Write metadata as JSON without adding a package dependency.
"""
function write_spk_metadata_json(path::AbstractString, metadata)
    open(path, "w") do io
        _write_json_value(io, _json_safe(metadata), 0)
        println(io)
    end
    return path
end

"""
    infer_force_model_metadata(parameters) -> Dict

Force-model metadata read from the dynamics parameters (`SpiceParameters`,
`EphemeridesParameters`, or any object with the same property names): bodies and
GMs, integration frame, spherical harmonics (central body, degree, body-fixed
frame), SRP and drag. Quantities that the parameters do not store (e.g. the EOM
name, or `Cr`, `A/m` and `P0`, which are folded into `k_srp_cannonball`) can be
added through the `force_model_metadata` keyword of the SPK writers.
"""
function infer_force_model_metadata(parameters)
    meta = Dict{String,Any}()
    parameters === nothing && return meta

    if parameters isa HighFidelityEphemerisModelParameters
        meta["parameters_type"] = string(nameof(typeof(parameters)))
    end

    ids = _prop(parameters, :naif_ids)
    ids !== nothing && (meta["naif_ids"] = _json_safe(ids))
    gms = _prop(parameters, :GMs)
    gms !== nothing && (meta["GMs"] = _json_safe(gms))
    frame = _prop(parameters, :naif_frame)
    frame !== nothing && (meta["frame"] = _json_safe(frame))

    if hasproperty(parameters, :spherical_harmonics_data)
        data = parameters.spherical_harmonics_data
        sh = Dict{String,Any}("enabled" => data !== nothing)
        if data !== nothing
            # The harmonics are those of the central body, `naif_ids[1]`.
            ids !== nothing && !isempty(ids) && (sh["body"] = string(first(ids)))
            if data isa AbstractDict
                for (key, name) in (("nmax", "degree"), ("GM", "GM"), ("REFERENCE RADIUS", "reference_radius"))
                    haskey(data, key) && (sh[name] = _json_safe(data[key]))
                end
            end
            frame_pcpf = _prop(parameters, :frame_PCPF)
            frame_pcpf !== nothing && (sh["body_fixed_frame"] = _json_safe(frame_pcpf))
        end
        meta["spherical_harmonics"] = sh
    end

    # Coefficients are only reported for enabled perturbations: the parameter
    # constructors compute them even when the perturbation is off.
    include_srp = _prop(parameters, :include_srp)
    if include_srp !== nothing
        srp = Dict{String,Any}("enabled" => Bool(include_srp))
        if include_srp == true
            k = _prop(parameters, :k_srp_cannonball)
            k !== nothing && (srp["k_cannonball_canonical"] = _json_safe(k))
        end
        meta["SRP"] = srp
    end

    include_drag = _prop(parameters, :include_drag)
    if include_drag !== nothing
        drag = Dict{String,Any}("enabled" => Bool(include_drag))
        if include_drag == true
            k = _prop(parameters, :k_drag)
            k !== nothing && (drag["k_drag_canonical"] = _json_safe(k))
            omega = _prop(parameters, :omega_atm)
            omega !== nothing && (drag["omega_atm_radps"] = _json_safe(omega))
        end
        meta["drag"] = drag
    end

    return meta
end

function _deep_merge_dicts!(base::AbstractDict, override::AbstractDict)
    for (key, value) in override
        if haskey(base, key) && base[key] isa AbstractDict && value isa AbstractDict
            _deep_merge_dicts!(base[key], value)
        else
            base[key] = value
        end
    end
    return base
end
