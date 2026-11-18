"""Incremental SPK writing: append one ODE solution at a time, then finalize"""

# These functions support station-keeping loops where each propagated arc is
# written to the BSP as soon as it is available, so that the `ODESolution`
# objects of a long Monte Carlo run need not be kept in memory:
#
#     for k in 1:N
#         sol_k = ...   # propagate one arc
#         append_ode_sol_to_spk!(sol_k, parameters; output_spk, spice_id, append = k > 1)
#     end
#     finalize_spk(output_spk, parameters; ocp_control, ocp_control_times)


"""
    append_ode_sol_to_spk!(sol, [et0,] parameters; output_spk, spice_id, append, kwargs...) -> NamedTuple

Write one ODE solution `sol` as one native SPICE type-13 segment of
`output_spk`. With `append = false`, a new kernel is created (replacing any
existing file); with `append = true`, the segment is appended to the existing
kernel. `et0` defaults to `parameters.et0`.

Segments are numbered in file order: the segment ID is `"<segment_id>_<k>"`,
where `k - 1` is the number of segments already in the file. When consecutive
arcs share a boundary epoch, SPICE uses the segment appended last there (the
post-maneuver state).

Keywords `spice_id`, `center_id`, `ref_frame_name`, `dt_sec`, `polynom_degree`,
`segment_id` and `frame_def_file` have the same meaning and defaults as for
[`ode_sol_to_spk`](@ref). No sidecar files are written: call
[`finalize_spk`](@ref) once all arcs have been appended.

Returns a `NamedTuple` with `output_spk`, `segment_index`, `appended`,
`epoch_range` and `point_count`.
"""
function append_ode_sol_to_spk!(
    sol,
    et0,
    parameters;
    output_spk::AbstractString,
    spice_id::Union{Integer,AbstractString},
    append::Bool,
    center_id::Union{Nothing,Integer,AbstractString} = nothing,
    ref_frame_name::Union{Nothing,AbstractString} = nothing,
    dt_sec::Real = 1800.0,
    polynom_degree::Integer = 7,
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    frame_def_file::Union{Nothing,AbstractString} = nothing,
)
    dt_sec > 0 || error("`dt_sec` must be positive.")
    spk_ids = _resolve_spk_ids(parameters, spice_id, center_id, ref_frame_name)

    output_spk_abs = abspath(output_spk)
    splitext(output_spk_abs)[2] == ".bsp" || error("`output_spk` must end in `.bsp`: $(output_spk_abs)")
    if append
        isfile(output_spk_abs) || error("Cannot append: output SPK does not exist: $(output_spk_abs). Use `append = false` for the first arc.")
        segment_index = length(read_spk_segments(output_spk_abs)) + 1
    else
        segment_index = 1
    end

    sampled = sample_segmented_states_for_spk(
        [sol], [(1, 1)], et0, parameters;
        dt_sec = Float64(dt_sec),
        segment_gap_sec = 0.0,
        verbose = false,
        show_progress = false,
    )
    segment = only(sampled.segments)

    kw = (; spk_ids..., segment_id = segment_id, polynom_degree = polynom_degree, frame_def_file = frame_def_file)
    if append
        append_spkw13_segment_to_spk!(segment, output_spk_abs; segment_index = segment_index, kw...)
    else
        # Staged in a temporary file: an existing kernel is only replaced once
        # the new one has been written.
        write_spkw13_spk!([segment], output_spk_abs; kw..., overwrite = true, verbose = false, show_progress = false)
    end

    return (
        output_spk = output_spk_abs,
        segment_index = segment_index,
        appended = append,
        epoch_range = only(sampled.epoch_ranges),
        point_count = only(sampled.point_counts),
    )
end

append_ode_sol_to_spk!(sol, parameters; kwargs...) =
    append_ode_sol_to_spk!(sol, _get_et0(parameters), parameters; kwargs...)


"""
    finalize_spk(output_spk, [et0,] parameters; kwargs...) -> NamedTuple

Write the sidecar files of an SPK built with [`append_ode_sol_to_spk!`](@ref):
the OCP maneuver file `<stem>_maneuvers.txt` (when `ocp_control` is given) and
the metadata file `<stem>_metadata.json`. The SPK itself is not modified. `et0`
defaults to `parameters.et0`.

The object, center, frame, polynomial degree and per-segment coverage recorded
in the metadata are read from the segments of `output_spk`, so the metadata
describes every appended arc. The sampling interval is not stored in the SPK and
is recorded as `null`. Trajectory-jump diagnostics need the ODE solutions and
are only available from [`ode_sol_to_spk`](@ref).

Keywords `ocp_control`, `ocp_control_times`, `write_metadata`,
`merge_maneuvers_into_metadata`, `force_model_metadata`, `extra_metadata` and
`verbose` have the same meaning as for [`ode_sol_to_spk`](@ref).

Returns a `NamedTuple` with `output_spk`, `maneuver_txt`, `metadata_json`,
`ocp_control_summary`, `segment_count`, `epoch_ranges` and `point_counts`.
"""
function finalize_spk(
    output_spk::AbstractString,
    et0,
    parameters;
    ocp_control = nothing,
    ocp_control_times = nothing,
    write_metadata::Bool = true,
    merge_maneuvers_into_metadata::Bool = false,
    force_model_metadata = nothing,
    extra_metadata = nothing,
    verbose::Bool = true,
)
    _check_sidecar_inputs(ocp_control, ocp_control_times, force_model_metadata)
    output_spk_abs = abspath(output_spk)
    segments = read_spk_segments(output_spk_abs)
    isempty(segments) && error("SPK has no segments: $(output_spk_abs)")

    for field in (:spice_id, :center_id, :ref_frame_name)
        values = unique(getproperty.(segments, field))
        length(values) == 1 || error("`finalize_spk` expects all segments of $(output_spk_abs) to share one `$(field)`; found $(values).")
    end
    degrees = unique(s.polynom_degree for s in segments)
    seg1 = first(segments)

    sidecars = _write_spk_sidecars(
        output_spk_abs, et0, parameters;
        spk_info = (
            spice_id = seg1.spice_id,
            center_id = seg1.center_id,
            ref_frame_name = seg1.ref_frame_name,
            polynom_degree = length(degrees) == 1 ? only(degrees) : degrees,
        ),
        windows = [(start_et = s.first, end_et = s.last, point_count = s.point_count, segment_id = s.segment_id)
                   for s in segments],
        ocp_control = ocp_control,
        ocp_control_times = ocp_control_times,
        write_metadata = write_metadata,
        merge_maneuvers_into_metadata = merge_maneuvers_into_metadata,
        force_model_metadata = force_model_metadata,
        extra_metadata = extra_metadata,
    )

    verbose && _print_spk_pipeline_summary(;
        output_spk = output_spk_abs,
        segment_count = length(segments),
        maneuver_txt = sidecars.maneuver_txt,
        metadata_json = sidecars.metadata_json,
        ocp_control_summary = sidecars.ocp_control_summary,
    )

    return (
        output_spk = output_spk_abs,
        maneuver_txt = sidecars.maneuver_txt,
        metadata_json = sidecars.metadata_json,
        ocp_control_summary = sidecars.ocp_control_summary,
        segment_count = length(segments),
        epoch_ranges = [(s.first, s.last) for s in segments],
        point_counts = [s.point_count for s in segments],
    )
end

finalize_spk(output_spk::AbstractString, parameters; kwargs...) =
    finalize_spk(output_spk, _get_et0(parameters), parameters; kwargs...)
