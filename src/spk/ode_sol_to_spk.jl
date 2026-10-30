"""
    ode_sol_to_spk.jl

High-level ODE-solution to native SPICE/SPK pipeline.

The public `ode_sol_to_spk` function writes a combined BSP from one or more
coast-arc ODE solutions using SPICE-native type-13 routines (`spkopn`,
`spkopa`, `spkw13`, and `spkcls`). Lower-level state sampling, SPK writing, maneuver,
metadata, and incremental helpers are split into the other files in `src/spk`.

Typical use:

```julia
result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 399,
    ref_frame_name = "J2000",
    dt_sec = 1800.0,
)
```

Assumptions:
- `sols` is a vector of coast-arc ODE solutions.
- Each solution has a `.t` vector in nondimensional time.
- Each solution can be called as `sol(t_nd)` and returns at least a 6-vector
  `[x, y, z, vx, vy, vz]` in nondimensional units.
- `parameters` has `TU`, `DU`, and `VU` fields.
"""


# A single `ODESolution` has a `.t` time grid. Vectors of solutions do not use
# this dispatch path and are handled by the batch writer below.
function _looks_like_single_ode_solution(obj)
    return hasproperty(obj, :t)
end


"""
    ode_sol_to_spk(sols, et0, parameters; kwargs...) to NamedTuple

Generate a combined SPK/BSP file directly from ODE coast-arc solutions.

Required keyword:
- `output_spk`: output `.bsp` path.

Important optional keywords:
- `spice_id=nothing`: SPICE object ID written into the SPK. Pass either
  `spice_id` or `object_name`.
- `center_id=nothing`: SPICE center ID written into the SPK. Pass either
  `center_id` or `center_name`.
- `dt_sec=1800.0`: sampling interval for each SPK segment.
- `segment_gap_sec=1e-7`: right-end trim/gap between adjacent SPK segments.
  The next segment still starts at the true boundary; the previous segment ends
  `segment_gap_sec` seconds before that boundary.
- `write_maneuvers=false`: optionally write a trajectory-jump diagnostic maneuver file.
- `write_metadata=true`: also write a metadata JSON file.
- `keep_intermediates=false`: keep or delete the temporary work directory.
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
    leapseconds_file::Union{Nothing,AbstractString} = nothing,
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    internal_file_name::Union{Nothing,AbstractString} = nothing,
    ncomch::Integer = 0,
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 1e-7,
    coast_windows = nothing,
    keep_intermediates::Bool = false,
    intermediate_parent_dir::Union{Nothing,AbstractString} = nothing,
    overwrite::Bool = true,
    append::Union{Nothing,Bool} = nothing,
    segment_index::Union{Nothing,Integer} = nothing,
    write_maneuvers::Bool = false,
    maneuver_txt::Union{Nothing,AbstractString} = nothing,
    merge_maneuvers_into_metadata::Bool = false,
    ocp_control = nothing,
    ocp_control_times = nothing,
    write_ocp_maneuvers::Bool = ocp_control !== nothing,
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
    print_summary::Bool = true,
    verbose::Bool = true,
    show_progress::Bool = true,
)
    output_spk_type == 13 || error("Native SPICE writer currently supports only type-13 SPK output. Got output_spk_type=$(output_spk_type).")
    dt_sec > 0 || error("`dt_sec` must be positive.")
    segment_gap_sec >= 0 || error("`segment_gap_sec` must be nonnegative.")

    if _looks_like_single_ode_solution(sols)
        # Single-solution input is the incremental append mode used by SK loops.
        coast_windows === nothing || error("`coast_windows` is only supported when passing a collection of ODE solutions. For incremental single-solution calls, pass one ODE solution segment at a time.")

        return _ode_sol_to_spk_incremental(
            sols,
            et0,
            parameters;
            output_spk = output_spk,
            spice_id = spice_id,
            object_name = object_name,
            center_id = center_id,
            center_name = center_name,
            ref_frame_name = ref_frame_name,
            producer_id = producer_id,
            output_spk_type = output_spk_type,
            polynom_degree = polynom_degree,
            segment_id = segment_id,
            segment_id_per_seg = segment_id_per_seg,
            leapseconds_file = leapseconds_file,
            frame_def_file = frame_def_file,
            internal_file_name = internal_file_name,
            ncomch = ncomch,
            dt_sec = Float64(dt_sec),
            segment_gap_sec = Float64(segment_gap_sec),
            overwrite = overwrite,
            append = append,
            segment_index = segment_index,
            write_maneuvers = write_maneuvers,
            maneuver_txt = maneuver_txt,
            merge_maneuvers_into_metadata = merge_maneuvers_into_metadata,
            ocp_control = ocp_control,
            ocp_control_times = ocp_control_times,
            write_ocp_maneuvers = write_ocp_maneuvers,
            ocp_maneuver_txt = ocp_maneuver_txt,
            merge_ocp_maneuvers_into_metadata = merge_ocp_maneuvers_into_metadata,
            write_metadata = write_metadata,
            metadata_json = metadata_json,
            force_model_metadata = force_model_metadata,
            dynamics_name = dynamics_name,
            srp_enabled = srp_enabled,
            srp_Cr = srp_Cr,
            srp_Am = srp_Am,
            srp_P0 = srp_P0,
            spherical_harmonics_enabled = spherical_harmonics_enabled,
            spherical_harmonics_body = spherical_harmonics_body,
            spherical_harmonics_file = spherical_harmonics_file,
            spherical_harmonics_nmax = spherical_harmonics_nmax,
            nmax = nmax,
            spherical_harmonics_frame = spherical_harmonics_frame,
            extra_metadata = extra_metadata,
            print_summary = print_summary,
            verbose = verbose,
            show_progress = show_progress,
        )
    end

    append === nothing || error("`append` is only supported when passing a single ODE solution segment. For a collection of solutions, `ode_sol_to_spk` builds a fresh combined SPK.")
    segment_index === nothing || error("`segment_index` is only supported when passing a single ODE solution segment.")

    # Batch mode keeps the previous API: pass a collection of coast arcs and
    # write all segments into one newly-created BSP.
    length(sols) > 0 || error("`sols` is empty; provide at least one coast arc.")

    output_spk_abs = _prepare_spk_final_path(output_spk; overwrite = overwrite)

    workdir = _make_spk_pipeline_workdir(output_spk_abs, intermediate_parent_dir)
    tmp_spk = joinpath(workdir, basename(output_spk_abs))

    windows = coast_windows === nothing ?
        default_coast_windows(sols) :
        collect(coast_windows)

    if verbose
        println("SPK generation pipeline")
        @printf("  coast arcs / segments : %d / %d\n", length(sols), length(windows))
        @printf("  sampling              : dt = %.6g s, gap = %.6g s\n", Float64(dt_sec), Float64(segment_gap_sec))
        println("  SPK writer            : native SPICE type 13")
        println("  temporary dir         : ", _display_path(workdir))
    end

    # Sample states in memory. The native writer does not need setup files or
    # external command-line tools.
    state_result = sample_segmented_states_for_spk(
        sols,
        windows,
        et0,
        parameters;
        dt_sec = Float64(dt_sec),
        segment_gap_sec = Float64(segment_gap_sec),
        verbose = verbose,
        show_progress = show_progress,
    )

    segments = state_result.segments
    nseg = length(segments)
    nseg > 0 || error("No SPK segments were sampled.")

    # Create the combined BSP with native SPICE type-13 segment writes.
    write_spkw13_spk!(
        segments,
        output_spk_abs;
        spice_id = spice_id,
        object_name = object_name,
        center_id = center_id,
        center_name = center_name,
        ref_frame_name = ref_frame_name,
        segment_id = segment_id,
        segment_id_per_seg = segment_id_per_seg,
        polynom_degree = polynom_degree,
        producer_id = producer_id,
        internal_file_name = internal_file_name,
        ncomch = ncomch,
        frame_def_file = frame_def_file,
        overwrite = overwrite,
        tmp_spk = tmp_spk,
        verbose = verbose,
        show_progress = show_progress,
    )

    isfile(output_spk_abs) || error("SPICE finished, but output SPK was not found: $(output_spk_abs)")

    # Compute maneuver entries once so the text file and metadata JSON agree exactly.
    maneuver_entries = collect_node_to_node_maneuvers_mps(sols, et0, parameters)
    maneuver_summary = summarize_maneuver_entries_mps(maneuver_entries)

    # If the user asks for merged maneuver entries, mirror that behavior for
    # OCP controls unless it was overridden explicitly.
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

    # Optional diagnostic from jumps between adjacent coast arcs.
    trajectory_maneuver_path = nothing
    if write_maneuvers
        trajectory_maneuver_path = (maneuver_txt !== nothing && !write_ocp_maneuvers) ?
            abspath(maneuver_txt) :
            string(splitext(output_spk_abs)[1], "_trajectory_jump_maneuvers.txt")
        mkpath(dirname(trajectory_maneuver_path))
        write_maneuver_entries_mps(trajectory_maneuver_path, maneuver_entries)
        verbose && !print_summary && println("Wrote trajectory-jump diagnostic maneuver file: ", _display_path(trajectory_maneuver_path))
    end

    # Primary maneuver product when OCP/executed controls are supplied.
    ocp_maneuver_path = nothing
    if write_ocp_maneuvers
        ocp_control_entries === nothing && error("Internal error: no OCP maneuver entries were created.")
        ocp_maneuver_path = ocp_maneuver_txt !== nothing ?
            abspath(ocp_maneuver_txt) :
            (maneuver_txt !== nothing ? abspath(maneuver_txt) : string(splitext(output_spk_abs)[1], "_maneuvers.txt"))
        mkpath(dirname(ocp_maneuver_path))
        write_ocp_control_entries_mps(ocp_maneuver_path, ocp_control_entries)
        verbose && !print_summary && println("Wrote maneuver file from OCP controls: ", _display_path(ocp_maneuver_path))
    end

    main_maneuver_path = ocp_maneuver_path !== nothing ? ocp_maneuver_path : trajectory_maneuver_path

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
            maneuver_file = trajectory_maneuver_path,
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
        rm(workdir; recursive = true, force = true)
        verbose && !print_summary && println("Deleted intermediate directory: ", _display_path(workdir))
    else
        verbose && !print_summary && println("Kept intermediate directory: ", _display_path(workdir))
    end

    if verbose && print_summary
        _print_spk_pipeline_summary(
            output_spk = output_spk_abs,
            maneuver_txt = main_maneuver_path,
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
        maneuver_txt = main_maneuver_path,
        ocp_maneuver_txt = ocp_maneuver_path,
        trajectory_maneuver_txt = trajectory_maneuver_path,
        metadata_json = metadata_path,
        maneuver_summary = maneuver_summary,
        ocp_control_summary = ocp_control_summary,
        segment_count = nseg,
        coast_windows = windows,
        intermediate_dir = kept_workdir ? workdir : nothing,
        states_dir = nothing,
        setup_dir = nothing,
        state_files = String[],
        setup_files = String[],
        epoch_ranges = state_result.epoch_ranges,
        point_counts = state_result.point_counts,
    )
end


function _ode_sol_to_spk_incremental(
    sol,
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
    leapseconds_file::Union{Nothing,AbstractString} = nothing,
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    internal_file_name::Union{Nothing,AbstractString} = nothing,
    ncomch::Integer = 0,
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 0.0,
    overwrite::Bool = true,
    append::Union{Nothing,Bool} = nothing,
    segment_index::Union{Nothing,Integer} = nothing,
    write_maneuvers::Bool = false,
    maneuver_txt::Union{Nothing,AbstractString} = nothing,
    merge_maneuvers_into_metadata::Bool = false,
    ocp_control = nothing,
    ocp_control_times = nothing,
    write_ocp_maneuvers::Bool = ocp_control !== nothing,
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
    print_summary::Bool = true,
    verbose::Bool = true,
    show_progress::Bool = true,
)
    output_spk_type == 13 || error("Native SPICE writer currently supports only type-13 SPK output. Got output_spk_type=$(output_spk_type).")

    output_spk_abs = abspath(output_spk)
    splitext(output_spk_abs)[2] == ".bsp" || error("`output_spk` must end in `.bsp`: $(output_spk_abs)")
    mkpath(dirname(output_spk_abs))

    # Default append behavior follows the file system: create the first segment
    # when the BSP is absent, append when it already exists.
    append_flag = append === nothing ? isfile(output_spk_abs) : Bool(append)
    seg_idx = segment_index === nothing ? 1 : Int(segment_index)
    seg_idx > 0 || error("`segment_index` must be positive.")

    if verbose
        println("SPK incremental generation pipeline")
        @printf("  mode                  : %s\n", append_flag ? "append existing kernel" : "create new kernel")
        @printf("  segment index         : %d\n", seg_idx)
        @printf("  sampling              : dt = %.6g s, gap = %.6g s\n", Float64(dt_sec), Float64(segment_gap_sec))
        println("  SPK writer            : native SPICE type 13")
        println("  output SPK            : ", _display_path(output_spk_abs))
    end

    # This call performs the requested incremental append. The current solution
    # can be discarded after this point by the calling simulation.
    append_result = append_solution_segment_to_spk!(
        sol,
        et0,
        parameters;
        output_spk = output_spk_abs,
        segment_index = seg_idx,
        append = append_flag,
        overwrite = overwrite,
        dt_sec = Float64(dt_sec),
        segment_gap_sec = Float64(segment_gap_sec),
        spice_id = spice_id,
        object_name = object_name,
        center_id = center_id,
        center_name = center_name,
        ref_frame_name = ref_frame_name,
        segment_id = segment_id,
        segment_id_per_seg = segment_id_per_seg,
        polynom_degree = polynom_degree,
        producer_id = producer_id,
        internal_file_name = internal_file_name,
        ncomch = ncomch,
        frame_def_file = frame_def_file,
    )

    isfile(output_spk_abs) || error("SPICE finished, but output SPK was not found: $(output_spk_abs)")

    # A single appended arc has no adjacent coast-arc jump by itself. Keep the
    # diagnostic summary present but empty for a stable return/metadata schema.
    maneuver_entries = Any[]
    maneuver_summary = summarize_maneuver_entries_mps(maneuver_entries)

    # If the user asks for merged maneuver entries, mirror that behavior for
    # OCP controls unless it was overridden explicitly.
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

    # Optional diagnostic from jumps between adjacent coast arcs.
    trajectory_maneuver_path = nothing
    if write_maneuvers
        trajectory_maneuver_path = (maneuver_txt !== nothing && !write_ocp_maneuvers) ?
            abspath(maneuver_txt) :
            string(splitext(output_spk_abs)[1], "_trajectory_jump_maneuvers.txt")
        mkpath(dirname(trajectory_maneuver_path))
        write_maneuver_entries_mps(trajectory_maneuver_path, maneuver_entries)
        verbose && !print_summary && println("Wrote empty trajectory-jump diagnostic maneuver file for single-solution call: ", _display_path(trajectory_maneuver_path))
    end

    # Primary maneuver product when OCP/executed controls are supplied.
    ocp_maneuver_path = nothing
    if write_ocp_maneuvers
        ocp_control_entries === nothing && error("Internal error: no OCP maneuver entries were created.")
        ocp_maneuver_path = ocp_maneuver_txt !== nothing ?
            abspath(ocp_maneuver_txt) :
            (maneuver_txt !== nothing ? abspath(maneuver_txt) : string(splitext(output_spk_abs)[1], "_maneuvers.txt"))
        mkpath(dirname(ocp_maneuver_path))
        write_ocp_control_entries_mps(ocp_maneuver_path, ocp_control_entries)
        verbose && !print_summary && println("Wrote maneuver file from OCP controls: ", _display_path(ocp_maneuver_path))
    end

    main_maneuver_path = ocp_maneuver_path !== nothing ? ocp_maneuver_path : trajectory_maneuver_path

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

        # Record the append mode details in metadata so downstream products can
        # distinguish single-arc append calls from batch writes.
        incremental_extra = Dict{String,Any}(
            "incremental_append" => Dict{String,Any}(
                "enabled" => true,
                "appended" => append_flag,
                "segment_index" => seg_idx,
            )
        )
        if extra_metadata !== nothing
            incremental_extra["user_extra"] = _json_safe(extra_metadata)
        end

        metadata = build_spk_metadata(
            output_spk = output_spk_abs,
            maneuver_file = trajectory_maneuver_path,
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
            coast_windows = [(1, 1)],
            epoch_ranges = [append_result.epoch_range],
            parameters = parameters,
            et0 = Float64(et0),
            leapseconds_file = leapseconds_file,
            frame_def_file = frame_def_file,
            force_model_metadata = effective_force_model_metadata,
            extra_metadata = incremental_extra,
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

    if verbose && print_summary
        _print_spk_pipeline_summary(
            output_spk = output_spk_abs,
            maneuver_txt = main_maneuver_path,
            ocp_maneuver_txt = ocp_maneuver_path,
            metadata_json = metadata_path,
            segment_count = 1,
            intermediate_dir = nothing,
            maneuver_summary = maneuver_summary,
            ocp_control_summary = ocp_control_summary,
        )
    end

    return (
        output_spk = output_spk_abs,
        maneuver_txt = main_maneuver_path,
        ocp_maneuver_txt = ocp_maneuver_path,
        trajectory_maneuver_txt = trajectory_maneuver_path,
        metadata_json = metadata_path,
        maneuver_summary = maneuver_summary,
        ocp_control_summary = ocp_control_summary,
        segment_count = 1,
        segment_index = seg_idx,
        appended = append_flag,
        coast_windows = [(1, 1)],
        intermediate_dir = nothing,
        states_dir = nothing,
        setup_dir = nothing,
        state_files = String[],
        setup_files = String[],
        epoch_ranges = [append_result.epoch_range],
        point_counts = [append_result.point_count],
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