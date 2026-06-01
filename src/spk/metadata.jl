"""Metadata JSON helpers for generated SPK files"""


"""
    build_spk_metadata(; kwargs...) -> Dict

Build metadata for the generated SPK, including force-model information,
NAIF frame, coverage windows, time/scaling units, and maneuver-file metadata.

# Arguments
- `output_spk::AbstractString`: generated BSP file path
- `epoch_ranges`: vector of segment coverage windows in seconds past J2000
- `parameters`: optional model/scaling object used for metadata extraction
- `force_model_metadata`: optional explicit force-model metadata dictionary
"""
function build_spk_metadata(;
    output_spk::AbstractString,
    maneuver_file = nothing,
    metadata_json = nothing,
    spice_id = nothing,
    object_name = nothing,
    center_id = nothing,
    center_name = nothing,
    ref_frame_name::AbstractString = "J2000",
    producer_id::AbstractString = "HighFidelityEphemerisModel.jl",
    output_spk_type::Integer = 13,
    polynom_degree::Integer = 7,
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 0.0,
    coast_windows = nothing,
    epoch_ranges = Tuple{Float64,Float64}[],
    parameters = nothing,
    et0 = nothing,
    leapseconds_file = nothing,
    frame_def_file = nothing,
    force_model_metadata = nothing,
    extra_metadata = nothing,
    maneuver_summary = nothing,
    ocp_control_summary = nothing,
    ocp_control_file = nothing,
    maneuver_entries = nothing,
    ocp_control_entries = nothing,
    merge_maneuvers_into_metadata::Bool = false,
    merge_ocp_maneuvers_into_metadata::Bool = false,
)
    windows = Any[]
    for (idx, epoch_range) in enumerate(epoch_ranges)
        entry = Dict{String,Any}(
            "segment_index" => idx,
            "start_et" => epoch_range[1],
            "end_et" => epoch_range[2],
        )
        if coast_windows !== nothing && idx <= length(coast_windows)
            cw = coast_windows[idx]
            entry["coast_window"] = [cw[1], cw[2]]
        end
        push!(windows, entry)
    end

    coverage_start = isempty(epoch_ranges) ? nothing : minimum(first.(epoch_ranges))
    coverage_end = isempty(epoch_ranges) ? nothing : maximum(last.(epoch_ranges))

    time_base = Dict{String,Any}()
    et0 !== nothing && (time_base["et0"] = et0)
    if parameters !== nothing
        _add_if_found!(time_base, "TU_seconds", parameters, (:TU, :tu, :time_unit))
        _add_if_found!(time_base, "DU_km", parameters, (:DU, :du, :distance_unit))
        _add_if_found!(time_base, "VU_kmps", parameters, (:VU, :vu, :velocity_unit))
    end

    force_model = force_model_metadata === nothing ?
        infer_force_model_metadata(parameters) :
        _json_safe(force_model_metadata)

    maneuver_metadata = Dict{String,Any}(
        "file" => maneuver_file === nothing ? nothing : abspath(String(maneuver_file)),
        "units" => "m/s",
        "format" => "ETSECONDS,DVX_mps,DVY_mps,DVZ_mps",
        "cost_convention" => "sum(norm.(dv_i))",
        "merged_into_metadata" => merge_maneuvers_into_metadata,
    )

    if maneuver_summary !== nothing
        _deep_merge_dicts!(maneuver_metadata, _json_safe(maneuver_summary))
    end

    if ocp_control_summary !== nothing
        ocp_meta = _json_safe(ocp_control_summary)
        ocp_meta["file"] = ocp_control_file === nothing ? nothing : abspath(String(ocp_control_file))
        ocp_meta["merged_into_metadata"] = merge_ocp_maneuvers_into_metadata
        if merge_ocp_maneuvers_into_metadata
            ocp_meta["entries"] = ocp_control_entries === nothing ? Any[] : _json_safe(ocp_control_entries)
        end
        maneuver_metadata["ocp_control"] = ocp_meta
    end

    if merge_maneuvers_into_metadata
        maneuver_metadata["entries"] = maneuver_entries === nothing ? Any[] : _json_safe(maneuver_entries)
    end

    meta = Dict{String,Any}(
        "schema" => "HighFidelityEphemerisModel.SPKMetadata.v1",
        "generated_unix_time_sec" => time(),
        "products" => Dict{String,Any}(
            "spk_file" => abspath(output_spk),
            "maneuver_file" => maneuver_file === nothing ? nothing : abspath(String(maneuver_file)),
            "ocp_maneuver_file" => ocp_control_file === nothing ? nothing : abspath(String(ocp_control_file)),
            "metadata_json" => metadata_json === nothing ? nothing : abspath(String(metadata_json)),
        ),
        "spk" => Dict{String,Any}(
            "object_id" => spice_id,
            "object_name" => object_name,
            "center_id" => center_id,
            "center_name" => center_name,
            "ref_frame_name" => ref_frame_name,
            "producer_id" => producer_id,
            "output_spk_type" => output_spk_type,
            "polynom_degree" => polynom_degree,
            "leapseconds_file" => leapseconds_file,
            "frame_def_file" => frame_def_file,
        ),
        "sampling" => Dict{String,Any}(
            "dt_sec" => Float64(dt_sec),
            "segment_gap_sec" => Float64(segment_gap_sec),
            "segment_count" => length(epoch_ranges),
        ),
        "time_base" => time_base,
        "coverage" => Dict{String,Any}(
            "start_et" => coverage_start,
            "end_et" => coverage_end,
            "windows" => windows,
        ),
        "force_model" => force_model,
        "maneuvers" => maneuver_metadata,
    )

    if extra_metadata !== nothing
        meta["extra"] = _json_safe(extra_metadata)
    end

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
    infer_force_model_metadata(parameters)

Best-effort metadata extraction from the model/scaling object. For production
runs, pass `force_model_metadata=...` when calling `ode_sol_to_spk` if the
parameter object does not expose clear field names.
"""
function infer_force_model_metadata(parameters)
    meta = Dict{String,Any}()

    parameters === nothing && return meta

    ids = _property_first(parameters, (:naif_ids, :naif_IDs, :body_ids, :bodies, :ids))
    mus = _property_first(parameters, (:GMs, :gms, :μs, :mus, :mu, :μ))
    ids !== nothing && (meta["naif_ids"] = _json_safe(ids))
    mus !== nothing && (meta["mu_values"] = _json_safe(mus))

    srp = Dict{String,Any}()
    _add_if_found!(srp, "enabled", parameters, (:use_srp, :srp_enabled, :SRP_ENABLED, :srp_on))
    _add_if_found!(srp, "Cr", parameters, (:srp_Cr, :SRP_Cr, :Cr))
    _add_if_found!(srp, "A_over_m", parameters, (:srp_Am, :SRP_Am, :A_over_m, :Am))
    _add_if_found!(srp, "P0", parameters, (:srp_P0, :SRP_P0, :P0))
    !isempty(srp) && (meta["SRP"] = srp)

    sh = Dict{String,Any}()
    _add_if_found!(sh, "enabled", parameters, (:use_spherical_harmonics, :spherical_harmonics_enabled, :sh_enabled))
    _add_if_found!(sh, "degree", parameters, (:nmax, :Nmax, :gravity_degree, :spherical_harmonics_degree))
    _add_if_found!(sh, "file", parameters, (:spherical_harmonics_file, :sh_file, :gravity_file))
    _add_if_found!(sh, "body_fixed_frame", parameters, (:frame_PCPF, :pcpf_frame, :body_fixed_frame))
    !isempty(sh) && (meta["spherical_harmonics"] = sh)

    return meta
end


function _build_force_model_keyword_metadata(;
    dynamics_name = nothing,
    srp_enabled = nothing,
    srp_Cr = nothing,
    srp_Am = nothing,
    srp_P0 = nothing,
    spherical_harmonics_enabled = nothing,
    spherical_harmonics_body = nothing,
    spherical_harmonics_file = nothing,
    spherical_harmonics_nmax = nothing,
    spherical_harmonics_frame = nothing,
)
    meta = Dict{String,Any}()

    dynamics_name !== nothing && (meta["dynamics"] = String(dynamics_name))

    srp = Dict{String,Any}()
    srp_enabled !== nothing && (srp["enabled"] = srp_enabled)
    srp_Cr !== nothing && (srp["Cr"] = Float64(srp_Cr))
    srp_Am !== nothing && (srp["A_over_m"] = Float64(srp_Am))
    srp_P0 !== nothing && (srp["P0"] = Float64(srp_P0))
    !isempty(srp) && (meta["SRP"] = srp)

    sh = Dict{String,Any}()
    spherical_harmonics_enabled !== nothing && (sh["enabled"] = spherical_harmonics_enabled)
    spherical_harmonics_body !== nothing && (sh["body"] = String(spherical_harmonics_body))
    spherical_harmonics_file !== nothing && (sh["file"] = String(spherical_harmonics_file))
    spherical_harmonics_nmax !== nothing && (sh["degree"] = Int(spherical_harmonics_nmax))
    spherical_harmonics_frame !== nothing && (sh["body_fixed_frame"] = String(spherical_harmonics_frame))
    !isempty(sh) && (meta["spherical_harmonics"] = sh)

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
