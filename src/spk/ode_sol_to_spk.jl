"""
    ode_sol_to_spk.jl

Single-file SCP-solution to MKSPK/SPK pipeline.

- segmented MKSPK STATES file writing
- exact per-segment MKSPK setup file writing
- sequential `mkspk` create/append calls
- node-to-node maneuver text-file writing
- automatic segment counting

Typical use from a notebook or script:

```julia
include("../src/ode_sol_to_spk.jl")

result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 399,
    ref_frame_name = "J2000",
    mkspk_cmd = ".\\mkspk.exe",
    dt_sec = 1800.0,
    segment_gap_sec = 1e-7,
    keep_intermediates = false,
)
```

Assumptions:
- `sols` is a vector of coast-arc solutions.
- Each coast arc has a `.t` vector in nondimensional time.
- Each coast arc can be called as `sol(t_nd)` and returns a 6-vector
  `[x, y, z, vx, vy, vz]` in nondimensional units.
- `parameters` has fields/properties `TU`, `DU`, and `VU`.
"""
# =============================================================================
# High-level public wrapper
# =============================================================================

"""
    ode_sol_to_spk(sols, et0, parameters; kwargs...) to NamedTuple

Generate a combined SPK/BSP file directly from SCP coast-arc solutions.

Required keyword:
- `output_spk`: output `.bsp` path.

Important optional keywords:
- `spice_id=nothing`: SPICE object ID to write into the SPK setup files.
  Pass either `spice_id` or `object_name`.
- `center_id=nothing`: SPICE center ID to write into the SPK setup files.
  Pass either `center_id` or `center_name`.
- `mkspk_cmd="mkspk"`: path/command for NAIF `mkspk`; on Windows this can be
  something like `".\\mkspk.exe"`.
- `dt_sec=1800.0`: sampling interval for each segment states file.
- `segment_gap_sec=1e-7`: right-end trim/gap between adjacent SPK segments.
  The next segment still starts at the true boundary; the previous segment ends
  `segment_gap_sec` seconds before that boundary. Thus each segment starts at its true initial epoch and, except for the final segment, ends slightly before its terminal boundary epoch.
- `write_maneuvers=true`: also write a node-to-node maneuver text file.
- `maneuver_txt=nothing`: path for the maneuver file. If `nothing`, the file is
  written next to the SPK as `<output name>_maneuvers.txt`.
- `merge_maneuvers_into_metadata=false`: if `true`, the maneuver rows are also
  embedded in the metadata JSON under `maneuvers.entries`. Set
  `write_maneuvers=false` with this option for a JSON-only maneuver record.
- `ocp_control=nothing`: optional OCP control matrix, e.g. `solution.u`. If
  provided, the metadata JSON also records the optimizer-side control cost
  separately from the reconstructed node-to-node velocity-jump cost.
- `ocp_control_times=nothing`: nondimensional time vector aligned with columns of
  `ocp_control`. Required when writing an OCP maneuver file or embedding OCP
  maneuver entries into the metadata JSON.
- `write_ocp_maneuvers=false`: if `true`, write a second maneuver file from
  `ocp_control`, representing the optimizer-commanded impulses.
- `ocp_maneuver_txt=nothing`: path for the OCP maneuver file. If `nothing`,
  the file is written next to the SPK as `<output name>_ocp_maneuvers.txt`.
- `merge_ocp_maneuvers_into_metadata=nothing`: if `true`, embed OCP maneuver
  rows in the metadata JSON. If `nothing`, this follows
  `merge_maneuvers_into_metadata` only when `ocp_control` is provided.
- `suppress_mkspk_output=true`: hide MKSPK banner/status text during each
  segment write. Set this to `false` when debugging MKSPK failures.
- `print_summary=true`: print one compact product/cost summary at the end.
- `keep_intermediates=false`: keep or delete temporary states/setup files.
- `intermediate_parent_dir=nothing`: parent directory for temporary run folder.
- `coast_windows=nothing`: optional explicit arc windows like `[(1,1),(2,2)]`.
  If `nothing`, one SPK segment is written per coast arc automatically.

Returns a `NamedTuple` with paths/counts for the generated products.
"""
function ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk::AbstractString,
    spice_id::Union{Nothing,Integer} = nothing,
    object_name::Union{Nothing,AbstractString} = nothing,
    center_id::Union{Nothing,Integer} = nothing,
    center_name::Union{Nothing,AbstractString} = nothing,
    ref_frame_name::AbstractString = "J2000",
    producer_id::AbstractString = "HighFidelityEphemerisModel.jl",
    output_spk_type::Integer = 13,
    polynom_degree::Integer = 7,
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    leapseconds_file::AbstractString = "naif0012.tls",
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    mkspk_cmd::AbstractString = "mkspk",
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 1e-7,
    coast_windows = nothing,
    keep_intermediates::Bool = false,
    intermediate_parent_dir::Union{Nothing,AbstractString} = nothing,
    overwrite::Bool = true,
    write_maneuvers::Bool = true,
    maneuver_txt::Union{Nothing,AbstractString} = nothing,
    merge_maneuvers_into_metadata::Bool = false,
    ocp_control = nothing,
    ocp_control_times = nothing,
    write_ocp_maneuvers::Bool = false,
    ocp_maneuver_txt::Union{Nothing,AbstractString} = nothing,
    merge_ocp_maneuvers_into_metadata::Union{Nothing,Bool} = nothing,
    write_metadata::Bool = true,
    metadata_json::Union{Nothing,AbstractString} = nothing,
    force_model_metadata = nothing,
    dynamics_name::Union{Nothing,AbstractString} = nothing,
    srp_enabled::Union{Nothing,Bool} = nothing,
    srp_Cr::Union{Nothing,Real} = nothing,
    srp_Am::Union{Nothing,Real} = nothing,
    srp_P0::Union{Nothing,Real} = nothing,
    spherical_harmonics_enabled::Union{Nothing,Bool} = nothing,
    spherical_harmonics_body::Union{Nothing,AbstractString} = nothing,
    spherical_harmonics_file::Union{Nothing,AbstractString} = nothing,
    spherical_harmonics_nmax::Union{Nothing,Integer} = nothing,
    nmax::Union{Nothing,Integer} = nothing,
    spherical_harmonics_frame::Union{Nothing,AbstractString} = nothing,
    extra_metadata = nothing,
    suppress_mkspk_output::Bool = true,
    print_summary::Bool = true,
    verbose::Bool = true,
    show_progress::Bool = true,
)
    length(sols) > 0 || error("`sols` is empty; provide at least one coast arc.")
    dt_sec > 0 || error("`dt_sec` must be positive.")
    segment_gap_sec >= 0 || error("`segment_gap_sec` must be nonnegative.")

    output_spk_abs = prepare_spk_output!(output_spk; overwrite = overwrite)

    workdir = _make_spk_pipeline_workdir(output_spk_abs, intermediate_parent_dir)
    states_dir = joinpath(workdir, "states_segmented")
    setup_dir  = joinpath(workdir, "setup_segmented")

    windows = coast_windows === nothing ?
        default_coast_windows(sols) :
        collect(coast_windows)

    if verbose
        println("SCP-to-SPK pipeline")
        @printf("  coast arcs / segments : %d / %d\n", length(sols), length(windows))
        @printf("  sampling              : dt = %.6g s, gap = %.6g s\n", Float64(dt_sec), Float64(segment_gap_sec))
        println("  temporary dir         : ", _display_path(workdir))
    end

    state_result = write_segmented_states_for_spk!(
        sols,
        windows,
        et0,
        parameters;
        dt_sec = Float64(dt_sec),
        segment_gap_sec = Float64(segment_gap_sec),
        outdir = states_dir,
        verbose = verbose,
        show_progress = show_progress,
    )

    state_files = state_result.state_files
    nseg = length(state_files)
    nseg > 0 || error("No segment state files were written.")

    setup_files = write_full_setups_for_state_files_exact!(
        state_files;
        outdir = setup_dir,
        output_spk_type = output_spk_type,
        object_id = spice_id,
        object_name = object_name,
        center_id = center_id,
        center_name = center_name,
        ref_frame_name = ref_frame_name,
        producer_id = producer_id,
        data_delimiter = ",",
        lines_per_record = 1,
        time_wrapper = "# ETSECONDS",
        ignore_first_line = 1,
        leapseconds_file = leapseconds_file,
        frame_def_file = frame_def_file,
        polynom_degree = polynom_degree,
        segment_id = segment_id,
        segment_id_per_seg = segment_id_per_seg,
        verbose = verbose,
        show_progress = show_progress,
    )

    run_mkspk_for_segments!(
        setup_files,
        state_files,
        output_spk_abs;
        mkspk_cmd = mkspk_cmd,
        verbose = verbose,
        show_progress = show_progress,
        suppress_mkspk_output = suppress_mkspk_output,
    )

    isfile(output_spk_abs) || error("mkspk finished, but output SPK was not found: $output_spk_abs")

    # Compute maneuver entries once so the text file and metadata JSON agree exactly.
    maneuver_entries = collect_node_to_node_maneuvers_mps(sols, et0, parameters)
    maneuver_summary = summarize_maneuver_entries_mps(maneuver_entries)

    effective_merge_ocp = merge_ocp_maneuvers_into_metadata === nothing ?
        (merge_maneuvers_into_metadata && ocp_control !== nothing) :
        Bool(merge_ocp_maneuvers_into_metadata)

    ocp_control_entries = nothing
    ocp_control_summary = nothing
    if ocp_control !== nothing
        if ocp_control_times !== nothing
            ocp_control_entries = collect_ocp_control_maneuvers_mps(ocp_control_times, ocp_control, et0, parameters)
            ocp_control_summary = summarize_ocp_control_entries_mps(ocp_control_entries)
        else
            if write_ocp_maneuvers || effective_merge_ocp
                error("`ocp_control_times` must be provided when `write_ocp_maneuvers=true` or OCP maneuver entries are merged into metadata.")
            end
            ocp_control_summary = summarize_ocp_control_mps(ocp_control, parameters)
        end
    elseif write_ocp_maneuvers || effective_merge_ocp
        error("`ocp_control` must be provided when writing or merging OCP maneuver entries.")
    end

    maneuver_path = nothing
    if write_maneuvers
        maneuver_path = maneuver_txt === nothing ?
            string(splitext(output_spk_abs)[1], "_maneuvers.txt") :
            abspath(maneuver_txt)
        mkpath(dirname(maneuver_path))
        write_maneuver_entries_mps(maneuver_path, maneuver_entries)
        verbose && !print_summary && println("Wrote maneuver file: ", _display_path(maneuver_path))
    end

    ocp_maneuver_path = nothing
    if write_ocp_maneuvers
        ocp_control_entries === nothing && error("Internal error: no OCP maneuver entries were created.")
        ocp_maneuver_path = ocp_maneuver_txt === nothing ?
            string(splitext(output_spk_abs)[1], "_ocp_maneuvers.txt") :
            abspath(ocp_maneuver_txt)
        mkpath(dirname(ocp_maneuver_path))
        write_ocp_control_entries_mps(ocp_maneuver_path, ocp_control_entries)
        verbose && !print_summary && println("Wrote OCP maneuver file: ", _display_path(ocp_maneuver_path))
    end

    metadata_path = nothing
    if write_metadata
        metadata_path = metadata_json === nothing ?
            string(splitext(output_spk_abs)[1], "_metadata.json") :
            abspath(metadata_json)
        mkpath(dirname(metadata_path))

        keyword_force_metadata = _build_force_model_keyword_metadata(
            dynamics_name = dynamics_name,
            srp_enabled = srp_enabled,
            srp_Cr = srp_Cr,
            srp_Am = srp_Am,
            srp_P0 = srp_P0,
            spherical_harmonics_enabled = spherical_harmonics_enabled,
            spherical_harmonics_body = spherical_harmonics_body,
            spherical_harmonics_file = spherical_harmonics_file,
            spherical_harmonics_nmax = spherical_harmonics_nmax === nothing ? nmax : spherical_harmonics_nmax,
            spherical_harmonics_frame = spherical_harmonics_frame,
        )

        effective_force_model_metadata = force_model_metadata
        if !isempty(keyword_force_metadata)
            effective_force_model_metadata = force_model_metadata === nothing ?
                infer_force_model_metadata(parameters) :
                _json_safe(force_model_metadata)
            _deep_merge_dicts!(effective_force_model_metadata, keyword_force_metadata)
        end

        metadata = build_spk_metadata(
            output_spk = output_spk_abs,
            maneuver_file = maneuver_path,
            metadata_json = metadata_path,
            spice_id = spice_id,
            object_name = object_name,
            center_id = center_id,
            center_name = center_name,
            ref_frame_name = ref_frame_name,
            producer_id = producer_id,
            output_spk_type = output_spk_type,
            polynom_degree = polynom_degree,
            dt_sec = Float64(dt_sec),
            segment_gap_sec = Float64(segment_gap_sec),
            coast_windows = windows,
            epoch_ranges = state_result.epoch_ranges,
            parameters = parameters,
            et0 = Float64(et0),
            leapseconds_file = leapseconds_file,
            frame_def_file = frame_def_file,
            force_model_metadata = effective_force_model_metadata,
            extra_metadata = extra_metadata,
            maneuver_summary = maneuver_summary,
            ocp_control_summary = ocp_control_summary,
            ocp_control_file = ocp_maneuver_path,
            maneuver_entries = merge_maneuvers_into_metadata ? maneuver_entries : nothing,
            ocp_control_entries = effective_merge_ocp ? ocp_control_entries : nothing,
            merge_maneuvers_into_metadata = merge_maneuvers_into_metadata,
            merge_ocp_maneuvers_into_metadata = effective_merge_ocp,
        )

        write_spk_metadata_json(metadata_path, metadata)
        verbose && !print_summary && println("Wrote metadata JSON: ", _display_path(metadata_path))
    end

    kept_workdir = keep_intermediates
    if !keep_intermediates
        rm(workdir; recursive=true, force=true)
        verbose && !print_summary && println("Deleted intermediate directory: ", _display_path(workdir))
    else
        verbose && !print_summary && println("Kept intermediate directory: ", _display_path(workdir))
    end

    if verbose && print_summary
        _print_spk_pipeline_summary(
            output_spk = output_spk_abs,
            maneuver_txt = maneuver_path,
            ocp_maneuver_txt = ocp_maneuver_path,
            metadata_json = metadata_path,
            segment_count = nseg,
            intermediate_dir = kept_workdir ? workdir : nothing,
            maneuver_summary = maneuver_summary,
            ocp_control_summary = ocp_control_summary,
        )
    end

    return (
        output_spk = output_spk_abs,
        maneuver_txt = maneuver_path,
        ocp_maneuver_txt = ocp_maneuver_path,
        metadata_json = metadata_path,
        maneuver_summary = maneuver_summary,
        ocp_control_summary = ocp_control_summary,
        segment_count = nseg,
        coast_windows = windows,
        intermediate_dir = kept_workdir ? workdir : nothing,
        states_dir = kept_workdir ? states_dir : nothing,
        setup_dir = kept_workdir ? setup_dir : nothing,
        state_files = kept_workdir ? state_files : String[],
        setup_files = kept_workdir ? setup_files : String[],
        epoch_ranges = state_result.epoch_ranges,
    )
end

"""
    ode_sol_to_spk(sols, parameters; kwargs...)

Convenience method when `parameters.et0` exists.
"""
function ode_sol_to_spk(sols, parameters; kwargs...)
    et0 = _get_et0_from_parameters(parameters)
    et0 === nothing && error("No `et0` argument was provided and `parameters` has no recognized ET0 property. Tried: et0, ET0, epoch0, et_start, t0_et.")
    return ode_sol_to_spk(sols, Float64(et0), parameters; kwargs...)
end
