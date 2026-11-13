"""
    ode_sol_to_spk.jl

High-level ODE-solution to native SPICE/SPK pipeline.

- `ode_sol_to_spk` writes one BSP from a collection of coast-arc ODE solutions,
  plus optional maneuver and metadata sidecar files.
- `append_ode_sol_to_spk!` (incremental.jl) appends one ODE solution at a time,
  for long station-keeping loops, and `finalize_spk` writes the sidecar files of
  such a kernel at the end.

Assumptions:
- Each solution has a `.t` vector in nondimensional time and can be called as
  `sol(t_nd)`, returning at least a Cartesian 6-vector `[x, y, z, vx, vy, vz]` in
  nondimensional units (element-based EOMs such as the GVE ones are not supported).
- `parameters` has `TU`, `DU` and `VU` fields; `et0`, `naif_ids` (central body
  first) and `naif_frame` are used as defaults when present.
"""


# A single `ODESolution` has a `.t` time grid; a collection of solutions does not.
_looks_like_single_ode_solution(obj) = hasproperty(obj, :t)


"""
    ode_sol_to_spk(sols, [et0,] parameters; output_spk, spice_id, kwargs...) -> NamedTuple

Write a native SPICE type-13 SPK (BSP) file from one or more coast-arc ODE
solutions `sols` (a single solution is treated as `[sol]`), together with
optional maneuver and metadata sidecar files. `et0` defaults to `parameters.et0`.

By default, one SPK segment is written per coast arc, so that no segment
interpolates across an impulsive maneuver.

# Required keywords
- `output_spk`: output `.bsp` path. Sidecar files are written next to it, named
  after its stem: `<stem>_maneuvers.txt`, `<stem>_trajectory_jump_maneuvers.txt`
  and `<stem>_metadata.json`.
- `spice_id`: SPICE ID of the object, as an integer or a name known to SPICE.

# SPK keywords
- `center_id=nothing`: center of motion (integer or name). Defaults to the
  central body of the dynamics, `parameters.naif_ids[1]`.
- `ref_frame_name=nothing`: frame of the states. Defaults to the integration
  frame `parameters.naif_frame` (else `"J2000"`).
- `dt_sec=1800.0`: sampling interval of each segment, in seconds.
- `segment_gap_sec=1e-7`: the end of every segment but the last is trimmed by
  this many seconds, so that segments do not overlap at arc boundaries.
- `polynom_degree=7`: odd degree of the type-13 Hermite interpolation.
- `segment_id="HFEM_SPK_SEGMENT"`: segment `k` is named `"<segment_id>_<k>"`
  (`k` zero-padded to 3 digits); at most 36 characters (35 for 1000 segments
  or more).
- `coast_windows=nothing`: explicit arc windows such as `[(1, 1), (2, 3)]`. A
  window `(a, b)` with `a < b` merges arcs `a:b` into one segment, so these arcs
  must be continuous in time and state; windows that cross an impulsive
  maneuver, a time gap or an overlap are rejected before anything is written.
- `validate_continuity=true`, `continuity_pos_tol_km=1e-3`,
  `continuity_vel_tol_kmps=1e-6`, `continuity_rtol=0.0`: continuity check of
  merged windows. A state component passes if
  `abs(Δ) <= atol + continuity_rtol * max(abs(x_left), abs(x_right))`.
- `frame_def_file=nothing`: frame kernel furnished before writing, for custom frames.
- `overwrite=true`: replace an existing `output_spk`. The new kernel is staged in a
  temporary file, so the existing one is kept if writing fails.

# Sidecar keywords
- `ocp_control=nothing`, `ocp_control_times=nothing`: commanded/executed controls,
  e.g. an OCP `solution.u` (rows 1:3: Δv vector; optional row 4: scalar magnitude
  used by the objective, all nondimensional) and their nondimensional times. When
  given, `<stem>_maneuvers.txt` is written and summarized in the metadata.
- `write_maneuvers=false`: also write `<stem>_trajectory_jump_maneuvers.txt`, the
  velocity jumps between adjacent arcs (a diagnostic).
- `write_metadata=true`: write `<stem>_metadata.json`.
- `merge_maneuvers_into_metadata=false`: also list the individual maneuvers
  (OCP controls and trajectory jumps) in the metadata.
- `force_model_metadata=nothing`: dictionary deep-merged into the force-model
  metadata inferred from `parameters` (e.g. `Dict("dynamics" => "eom_NbodySH!")`).
- `extra_metadata=nothing`: free-form entry stored under `"extra"`.

# Output keywords
- `verbose=true`: print a header and a summary of the generated products.
- `show_progress=verbose`: show progress bars.

Returns a `NamedTuple` with the product paths (`output_spk`, `maneuver_txt`,
`trajectory_maneuver_txt`, `metadata_json`; `nothing` when not written),
`maneuver_summary`, `ocp_control_summary`, `segment_count`, `coast_windows`,
`epoch_ranges` and `point_counts`.
"""
function ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk::AbstractString,
    spice_id::Union{Integer,AbstractString},
    center_id::Union{Nothing,Integer,AbstractString} = nothing,
    ref_frame_name::Union{Nothing,AbstractString} = nothing,
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 1e-7,
    polynom_degree::Integer = 7,
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    coast_windows = nothing,
    validate_continuity::Bool = true,
    continuity_pos_tol_km::Real = 1e-3,
    continuity_vel_tol_kmps::Real = 1e-6,
    continuity_rtol::Real = 0.0,
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    overwrite::Bool = true,
    ocp_control = nothing,
    ocp_control_times = nothing,
    write_maneuvers::Bool = false,
    write_metadata::Bool = true,
    merge_maneuvers_into_metadata::Bool = false,
    force_model_metadata = nothing,
    extra_metadata = nothing,
    verbose::Bool = true,
    show_progress::Bool = verbose,
)
    dt_sec > 0 || error("`dt_sec` must be positive.")
    segment_gap_sec >= 0 || error("`segment_gap_sec` must be nonnegative.")
    _looks_like_single_ode_solution(sols) && (sols = [sols])
    length(sols) > 0 || error("`sols` is empty; provide at least one coast arc.")
    _check_sidecar_inputs(ocp_control, ocp_control_times, force_model_metadata)

    spk_ids = _resolve_spk_ids(parameters, spice_id, center_id, ref_frame_name)

    # Check every window before touching the filesystem, so a rejected window
    # leaves no partial output behind.
    windows = coast_windows === nothing ? default_coast_windows(sols) : collect(coast_windows)
    _validate_coast_windows(
        sols, windows, parameters;
        validate_continuity = validate_continuity,
        continuity_pos_tol_km = continuity_pos_tol_km,
        continuity_vel_tol_kmps = continuity_vel_tol_kmps,
        continuity_rtol = continuity_rtol,
    )
    output_spk_abs = _prepare_spk_final_path(output_spk; overwrite = overwrite)

    if verbose
        println("SPK generation pipeline")
        @printf("  coast arcs / segments : %d / %d\n", length(sols), length(windows))
        @printf("  sampling              : dt = %.6g s, gap = %.6g s\n", Float64(dt_sec), Float64(segment_gap_sec))
    end

    sampled = sample_segmented_states_for_spk(
        sols, windows, et0, parameters;
        dt_sec = Float64(dt_sec),
        segment_gap_sec = Float64(segment_gap_sec),
        verbose = verbose,
        show_progress = show_progress,
        validate_continuity = false,  # already checked above
    )

    write_spkw13_spk!(
        sampled.segments, output_spk_abs;
        spk_ids...,
        segment_id = segment_id,
        polynom_degree = polynom_degree,
        frame_def_file = frame_def_file,
        overwrite = overwrite,
        verbose = verbose,
        show_progress = show_progress,
    )

    segment_windows = [(
        start_et = seg.first,
        end_et = seg.last,
        point_count = length(seg.epochs),
        segment_id = _spkw13_segment_id(segment_id, k),
        coast_window = seg.coast_window,
    ) for (k, seg) in enumerate(sampled.segments)]

    sidecars = _write_spk_sidecars(
        output_spk_abs, et0, parameters;
        spk_info = (; spk_ids..., polynom_degree = polynom_degree, frame_def_file = frame_def_file,
                    dt_sec = Float64(dt_sec), segment_gap_sec = Float64(segment_gap_sec)),
        windows = segment_windows,
        maneuver_entries = collect_node_to_node_maneuvers_mps(sols, et0, parameters),
        write_maneuvers = write_maneuvers,
        ocp_control = ocp_control,
        ocp_control_times = ocp_control_times,
        write_metadata = write_metadata,
        merge_maneuvers_into_metadata = merge_maneuvers_into_metadata,
        force_model_metadata = force_model_metadata,
        extra_metadata = extra_metadata,
    )

    verbose && _print_spk_pipeline_summary(;
        output_spk = output_spk_abs,
        segment_count = length(sampled.segments),
        maneuver_txt = sidecars.maneuver_txt,
        trajectory_maneuver_txt = sidecars.trajectory_maneuver_txt,
        metadata_json = sidecars.metadata_json,
        ocp_control_summary = sidecars.ocp_control_summary,
    )

    return (
        output_spk = output_spk_abs,
        sidecars...,
        segment_count = length(sampled.segments),
        coast_windows = windows,
        epoch_ranges = sampled.epoch_ranges,
        point_counts = sampled.point_counts,
    )
end

ode_sol_to_spk(sols, parameters; kwargs...) = ode_sol_to_spk(sols, _get_et0(parameters), parameters; kwargs...)


"""
    _resolve_spk_ids(parameters, spice_id, center_id, ref_frame_name)

Integer object/center codes and frame name written into the SPK, with the
center and frame defaulted from `parameters`.
"""
function _resolve_spk_ids(parameters, spice_id, center_id, ref_frame_name)
    center = center_id === nothing ? _default_center(parameters) : center_id
    return (
        spice_id = _resolve_body_code(spice_id; label = "spice_id"),
        center_id = _resolve_body_code(center; label = "center_id"),
        ref_frame_name = ref_frame_name === nothing ? _default_ref_frame(parameters) : String(ref_frame_name),
    )
end


# Check the sidecar inputs before anything is written.
function _check_sidecar_inputs(ocp_control, ocp_control_times, force_model_metadata)
    force_model_metadata === nothing || force_model_metadata isa Union{AbstractDict,NamedTuple} ||
        error("`force_model_metadata` must be a dictionary or NamedTuple (it is merged into the force-model metadata inferred from `parameters`), got $(typeof(force_model_metadata)).")
    return _check_ocp_control(ocp_control, ocp_control_times)
end

function _check_ocp_control(ocp_control, ocp_control_times)
    if ocp_control === nothing
        ocp_control_times === nothing || error("`ocp_control_times` was given without `ocp_control`.")
        return nothing
    end
    ocp_control_times === nothing && error("`ocp_control_times` must be provided together with `ocp_control`.")
    size(ocp_control, 1) >= 3 || error("`ocp_control` must have at least 3 rows for DVX/DVY/DVZ.")
    length(ocp_control_times) == size(ocp_control, 2) || error("`ocp_control_times` length ($(length(ocp_control_times))) must match the number of `ocp_control` columns ($(size(ocp_control, 2))).")
    return nothing
end


"""
    _write_spk_sidecars(output_spk, et0, parameters; spk_info, windows, kwargs...)

Write the maneuver and metadata files of an SPK, named after the stem of
`output_spk`. `maneuver_entries` are the trajectory-jump diagnostics (`nothing`
when the arcs are not available, as in `finalize_spk`).
"""
function _write_spk_sidecars(
    output_spk_abs::AbstractString,
    et0,
    parameters;
    spk_info::NamedTuple,
    windows,
    maneuver_entries = nothing,
    write_maneuvers::Bool = false,
    ocp_control = nothing,
    ocp_control_times = nothing,
    write_metadata::Bool = true,
    merge_maneuvers_into_metadata::Bool = false,
    force_model_metadata = nothing,
    extra_metadata = nothing,
)
    stem = splitext(output_spk_abs)[1]

    maneuver_summary = maneuver_entries === nothing ? nothing : summarize_maneuver_entries_mps(maneuver_entries)
    trajectory_maneuver_txt = nothing
    if write_maneuvers
        maneuver_entries === nothing && error("Internal error: no trajectory-jump maneuver entries.")
        trajectory_maneuver_txt = write_maneuver_entries_mps(stem * "_trajectory_jump_maneuvers.txt", maneuver_entries)
    end

    ocp_control_entries = nothing
    ocp_control_summary = nothing
    maneuver_txt = nothing
    if ocp_control !== nothing
        ocp_control_entries = collect_ocp_control_maneuvers_mps(ocp_control_times, ocp_control, et0, parameters)
        ocp_control_summary = summarize_ocp_control_entries_mps(ocp_control_entries)
        maneuver_txt = write_ocp_control_entries_mps(stem * "_maneuvers.txt", ocp_control_entries)
    end

    metadata_json = nothing
    if write_metadata
        metadata_json = stem * "_metadata.json"
        metadata = build_spk_metadata(;
            output_spk = output_spk_abs,
            metadata_json = metadata_json,
            spk_info...,
            windows = windows,
            et0 = et0,
            parameters = parameters,
            force_model_metadata = force_model_metadata,
            extra_metadata = extra_metadata,
            trajectory_maneuver_file = trajectory_maneuver_txt,
            maneuver_summary = maneuver_summary,
            maneuver_entries = maneuver_entries,
            ocp_control_file = maneuver_txt,
            ocp_control_summary = ocp_control_summary,
            ocp_control_entries = ocp_control_entries,
            merge_maneuvers_into_metadata = merge_maneuvers_into_metadata,
        )
        write_spk_metadata_json(metadata_json, metadata)
    end

    return (
        maneuver_txt = maneuver_txt,
        trajectory_maneuver_txt = trajectory_maneuver_txt,
        metadata_json = metadata_json,
        maneuver_summary = maneuver_summary,
        ocp_control_summary = ocp_control_summary,
    )
end
