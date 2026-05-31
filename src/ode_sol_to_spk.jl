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

using Printf: @printf

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

# =============================================================================
# Segmentation / state-file writing
# =============================================================================

"""
    default_coast_windows(sols)

Default segmentation: one SPK segment per coast arc.
This avoids interpolating across impulsive velocity jumps.
"""
default_coast_windows(sols) = [(k, k) for k in 1:length(sols)]

"""
    build_segment_epochs(et_start, et_end; dt_sec=1800.0)

Build a uniform epoch grid from `et_start` to `et_end`, always including the
endpoint exactly once.
"""
function build_segment_epochs(et_start::Float64, et_end::Float64; dt_sec::Float64 = 1800.0)
    @assert et_end > et_start

    ts = Float64[et_start]
    t = et_start + dt_sec
    tol = 1e-13

    while t < et_end - tol
        push!(ts, t)
        t += dt_sec
    end

    if abs(ts[end] - et_end) > tol
        push!(ts, et_end)
    end

    return ts
end

function write_mkspk_states_file(
    filepath::AbstractString,
    ts_et::Vector{Float64},
    Y::Matrix{Float64},
)
    @assert size(Y, 1) == 6
    @assert length(ts_et) == size(Y, 2)

    open(filepath, "w") do io
        println(io, "# ETSECONDS")
        for k in 1:length(ts_et)
            @printf(io, "%.20f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n",
                ts_et[k],
                Y[1, k], Y[2, k], Y[3, k],
                Y[4, k], Y[5, k], Y[6, k],
            )
        end
    end

    return filepath
end

"""
    write_segmented_states_for_spk!(sols, coast_windows, et0, parameters; ...)

Write one MKSPK `STATES` file per `coast_windows` entry and return all written
file paths.
"""
function write_segmented_states_for_spk!(
    sols,
    coast_windows,
    et0,
    parameters;
    dt_sec::Float64 = 1800.0,
    segment_gap_sec::Float64 = 1e-7,
    outdir::AbstractString = "states_segmented",
    verbose::Bool = true,
    show_progress::Bool = true,
)
    isdir(outdir) || mkpath(outdir)

    tol_nd = 1e-10
    tol_et = 1e-9
    nseg = length(coast_windows)
    progress_enabled = verbose && show_progress

    state_files = String[]
    epoch_ranges = Tuple{Float64,Float64}[]

    for (seg_idx, window) in enumerate(coast_windows)
        a, b = window
        @assert 1 <= a <= b <= length(sols) "Bad coast window $(window) for $(length(sols)) coast arcs."

        seg_sols = sols[a:b]

        et_start_true = Float64(et0 + seg_sols[1].t[1]     * parameters.TU)
        et_end_true   = Float64(et0 + seg_sols[end].t[end] * parameters.TU)

        et_start = et_start_true
        et_end = seg_idx < nseg ? et_end_true - segment_gap_sec : et_end_true

        if et_end <= et_start + tol_et
            error("Segment $(seg_idx) arcs[$a,$b] has bad span after applying segment_gap_sec=$(segment_gap_sec).")
        end

        ts_et = build_segment_epochs(et_start, et_end; dt_sec = dt_sec)

        cols = Vector{Vector{Float64}}()
        sol_ptr = 1

        for et in ts_et
            t_nd = (et - et0) / parameters.TU

            while sol_ptr < length(seg_sols) && t_nd > seg_sols[sol_ptr].t[end] + tol_nd
                sol_ptr += 1
            end

            if t_nd < seg_sols[sol_ptr].t[1] - 1e-8 || t_nd > seg_sols[sol_ptr].t[end] + 1e-8
                @printf("ERROR seg %03d: epoch not covered by sols. et=%.20f t_nd=%.15e sol_ptr=%d sol=[%.15e, %.15e]\n",
                    seg_idx, et, t_nd, sol_ptr, seg_sols[sol_ptr].t[1], seg_sols[sol_ptr].t[end])
                error("Segment coverage failure: check coast_windows boundaries vs sols.")
            end

            x_nd = seg_sols[sol_ptr](t_nd)
            length(x_nd) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_nd)).")

            r_km   = x_nd[1:3] .* parameters.DU
            v_kmps = x_nd[4:6] .* parameters.VU

            push!(cols, vcat(Float64.(r_km), Float64.(v_kmps)))
        end

        @assert length(ts_et) > 1 "Segment $seg_idx produced <2 points. Reduce dt_sec or check segment duration."

        Y = reduce(hcat, cols)
        tag = lpad(string(seg_idx), 3, '0')
        outpath = joinpath(outdir, "seg_$(tag)_states.txt")
        write_mkspk_states_file(outpath, ts_et, Y)

        push!(state_files, outpath)
        push!(epoch_ranges, (ts_et[1], ts_et[end]))

        _print_progress("writing state files", seg_idx, nseg; enabled = progress_enabled)
    end

    verbose && !show_progress && println("All state files written to: ", _display_path(outdir))

    return (
        outdir = String(outdir),
        state_files = state_files,
        epoch_ranges = epoch_ranges,
    )
end

"""
    write_segmented_states_from_coast_windows_epsilon_trim!(...)

Backward-compatible wrapper around `write_segmented_states_for_spk!` using the
old keyword name `eps_sec`. Returns only `outdir`, matching the older helper.
"""
function write_segmented_states_from_coast_windows_epsilon_trim!(
    sols,
    coast_windows,
    et0,
    parameters;
    dt_sec::Float64 = 1800.0,
    eps_sec::Float64 = 1e-10,
    outdir::String = "states_segmented",
)
    write_segmented_states_for_spk!(
        sols,
        coast_windows,
        et0,
        parameters;
        dt_sec = dt_sec,
        segment_gap_sec = eps_sec,
        outdir = outdir,
        verbose = true,
    )
    return outdir
end

# =============================================================================
# MKSPK setup generation / execution
# =============================================================================

function _epoch_range_from_states_file(states_file::AbstractString)
    first_epoch = nothing
    last_epoch = nothing

    open(states_file, "r") do io
        first_line_skipped = false
        for line in eachline(io)
            if !first_line_skipped
                first_line_skipped = true
                continue
            end

            s = strip(line)
            isempty(s) && continue

            idx = findfirst(==(','), s)
            idx === nothing && continue

            epoch = parse(Float64, s[1:idx-1])
            if first_epoch === nothing
                first_epoch = epoch
            end
            last_epoch = epoch
        end
    end

    @assert first_epoch !== nothing && last_epoch !== nothing "No epochs found in $states_file"
    return Float64(first_epoch), Float64(last_epoch)
end

"""
    write_full_mkspk_setup_exact(setup_path; states_file_for_epochs, ...)

Write a full MKSPK setup file with exact `EARLIEST_EPOCH` and `LATEST_EPOCH`
computed directly from the corresponding segment states file.
"""
function write_full_mkspk_setup_exact(
    setup_path::AbstractString;
    segment_id::String,
    states_file_for_epochs::AbstractString,
    output_spk_type::Integer = 13,
    object_id::Union{Nothing,Integer} = nothing,
    object_name::Union{Nothing,AbstractString} = nothing,
    center_id::Union{Nothing,Integer} = nothing,
    center_name::Union{Nothing,AbstractString} = nothing,
    ref_frame_name::AbstractString = "J2000",
    producer_id::AbstractString = "HighFidelityEphemerisModel.jl",
    data_delimiter::AbstractString = ",",
    lines_per_record::Integer = 1,
    time_wrapper::AbstractString = "# ETSECONDS",
    ignore_first_line::Integer = 1,
    leapseconds_file::AbstractString = "naif0012.tls",
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    polynom_degree::Integer = 7,
)
    earliest, latest = _epoch_range_from_states_file(states_file_for_epochs)

    open(setup_path, "w") do io
        @printf(io, "\\begindata\n")
        @printf(io, "   INPUT_DATA_TYPE   = 'STATES'\n")
        @printf(io, "   OUTPUT_SPK_TYPE   = %d\n", output_spk_type)

        if object_id !== nothing
            @printf(io, "   OBJECT_ID         = %d\n", object_id)
        elseif object_name !== nothing
            @printf(io, "   OBJECT_NAME       = '%s'\n", object_name)
        else
            error("Either object_id or object_name must be provided.")
        end

        if center_id !== nothing
            @printf(io, "   CENTER_ID         = %d\n", center_id)
        elseif center_name !== nothing
            @printf(io, "   CENTER_NAME       = '%s'\n", center_name)
        else
            error("Either center_id or center_name must be provided.")
        end

        @printf(io, "   REF_FRAME_NAME    = '%s'\n", ref_frame_name)
        @printf(io, "   PRODUCER_ID       = '%s'\n", producer_id)
        @printf(io, "   DATA_ORDER        = 'EPOCH X Y Z VX VY VZ'\n")
        @printf(io, "   INPUT_DATA_UNITS  = ('ANGLES=DEGREES' 'DISTANCES=km')\n")
        @printf(io, "   DATA_DELIMITER    = '%s'\n", data_delimiter)
        @printf(io, "   LINES_PER_RECORD  = %d\n", lines_per_record)
        @printf(io, "   TIME_WRAPPER      = '%s'\n", time_wrapper)
        @printf(io, "   IGNORE_FIRST_LINE = %d\n", ignore_first_line)
        @printf(io, "   LEAPSECONDS_FILE  = '%s'\n", _mkspk_path(leapseconds_file))

        if frame_def_file !== nothing
            @printf(io, "   FRAME_DEF_FILE    = '%s'\n", _mkspk_path(frame_def_file))
        end

        @printf(io, "   POLYNOM_DEGREE    = %d\n", polynom_degree)
        @printf(io, "   SEGMENT_ID        = '%s'\n", segment_id)
        @printf(io, "   EARLIEST_EPOCH    = %.15f\n", earliest)
        @printf(io, "   LATEST_EPOCH      = %.15f\n", latest)
        @printf(io, "\\begintext\n")
    end

    return setup_path
end

"""
    write_full_setups_for_state_files_exact!(state_files; kwargs...) -> Vector{String}

Write setup files for an already known vector of states files.
"""
function write_full_setups_for_state_files_exact!(
    state_files::Vector{String};
    outdir::AbstractString = "setup_segmented",
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
)
    isdir(outdir) || mkpath(outdir)
    progress_enabled = verbose && show_progress

    setup_files = String[]
    for (s, states_path) in enumerate(state_files)
        tag = lpad(string(s), 3, '0')
        setup_path = joinpath(outdir, "seg_$(tag)_setup.txt")
        @assert isfile(states_path) "Missing states file: $states_path"

        segid = segment_id_per_seg ? "$(segment_id)_$(tag)" : String(segment_id)
        write_full_mkspk_setup_exact(
            setup_path;
            segment_id = segid,
            states_file_for_epochs = states_path,
            kwargs...,
        )

        push!(setup_files, setup_path)
        _print_progress("writing setup files", s, length(state_files); enabled = progress_enabled)
    end

    return setup_files
end

"""
    write_full_setups_for_segments_exact!(nseg; ...)

Backward-compatible wrapper for the older manual-count workflow. New code should
prefer `write_full_setups_for_state_files_exact!` or `ode_sol_to_spk`.
"""
function write_full_setups_for_segments_exact!(
    nseg::Int;
    outdir::String = "setup_segmented",
    state_dir::String = "states_segmented",
    segment_id::String = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    kwargs...,
)
    state_files = [joinpath(state_dir, "seg_$(lpad(string(s), 3, '0'))_states.txt") for s in 1:nseg]
    write_full_setups_for_state_files_exact!(
        state_files;
        outdir = outdir,
        segment_id = segment_id,
        segment_id_per_seg = segment_id_per_seg,
        kwargs...,
    )
    return outdir
end

"""
    wrap_mkspk(filepath_set, filepath_in, filepath_out; mkspk_cmd="mkspk", append=false, overwrite=false)

Non-interactive version of the old `wrap_mkspk`. It never prompts the user.
"""
function wrap_mkspk(
    filepath_set,
    filepath_in,
    filepath_out;
    mkspk_cmd = "mkspk",
    append::Bool = false,
    overwrite::Bool = false,
    verbose::Bool = true,
    suppress_output::Bool = true,
)
    @assert splitext(filepath_out)[2] == ".bsp"

    if !append && isfile(filepath_out)
        if overwrite
            rm(filepath_out; force = true)
        else
            error("Output BSP already exists: $filepath_out. Use overwrite=true or remove it first.")
        end
    end

    cmd = if append
        @assert isfile(filepath_out) "Cannot append: output BSP does not exist: $filepath_out"
        `$mkspk_cmd -append -setup $filepath_set -input $filepath_in -output $filepath_out`
    else
        `$mkspk_cmd -setup $filepath_set -input $filepath_in -output $filepath_out`
    end

    if suppress_output
        run(pipeline(cmd; stdout = devnull, stderr = devnull))
    else
        run(cmd)
    end

    if isfile(filepath_out)
        if append
            verbose && println("Successfully appended to $(filepath_out)!")
        else
            verbose && println("Successfully generated $(filepath_out)!")
        end
    else
        error("mkspk did not create expected output: $filepath_out")
    end

    return filepath_out
end

function run_mkspk_for_segments!(
    setup_files::Vector{String},
    state_files::Vector{String},
    output_spk::AbstractString;
    mkspk_cmd::AbstractString = "mkspk",
    verbose::Bool = true,
    show_progress::Bool = true,
    suppress_mkspk_output::Bool = true,
)
    length(setup_files) == length(state_files) || error("setup_files and state_files must have the same length.")
    length(setup_files) > 0 || error("No setup/state files provided.")
    progress_enabled = verbose && show_progress

    for s in eachindex(setup_files)
        setup_path = setup_files[s]
        state_path = state_files[s]
        @assert isfile(setup_path) "Missing setup file: $setup_path"
        @assert isfile(state_path) "Missing states file: $state_path"

        wrap_mkspk(
            setup_path,
            state_path,
            output_spk;
            mkspk_cmd = mkspk_cmd,
            append = s > 1,
            overwrite = false,
            verbose = false,
            suppress_output = suppress_mkspk_output,
        )
        _print_progress("running mkspk", s, length(setup_files); enabled = progress_enabled)
    end

    verbose && !show_progress && println("SPK complete: ", _display_path(output_spk))
    return output_spk
end

# =============================================================================
# Maneuver writer
# =============================================================================

"""
    collect_node_to_node_maneuvers_mps(sols, et0, parameters)

Collect nominal node-to-node delta-Vs. Each entry is the instantaneous velocity
jump between `sols[k]` end and `sols[k+1]` start, converted to m/s.

The returned vector is used for both the maneuver text file and metadata JSON so
those two products remain consistent.
"""
function collect_node_to_node_maneuvers_mps(sols, et0, parameters)
    entries = Any[]

    for k in 1:(length(sols) - 1)
        t_end = sols[k].t[end]

        x_end = sols[k](t_end)
        x_start = sols[k + 1](sols[k + 1].t[1])

        dv_nd = x_start[4:6] - x_end[4:6]
        dv_kmps = dv_nd .* parameters.VU
        dv_mps = dv_kmps .* 1000.0
        dv_norm_mps = sqrt(sum(abs2, dv_mps))

        et = et0 + t_end * parameters.TU

        push!(entries, (
            index = k,
            et = Float64(et),
            dvx_mps = Float64(dv_mps[1]),
            dvy_mps = Float64(dv_mps[2]),
            dvz_mps = Float64(dv_mps[3]),
            dv_norm_mps = Float64(dv_norm_mps),
            dv_norm_cmps = Float64(100.0 * dv_norm_mps),
        ))
    end

    return entries
end

"""
    summarize_maneuver_entries_mps(entries)

Summarize maneuver entries using station-keeping cost convention
`total_delta_v = sum(norm.(dv_i))`.
"""
function summarize_maneuver_entries_mps(entries)
    if isempty(entries)
        return Dict{String,Any}(
            "count" => 0,
            "total_delta_v_mps" => 0.0,
            "total_delta_v_cmps" => 0.0,
            "max_delta_v_mps" => 0.0,
            "max_delta_v_cmps" => 0.0,
            "mean_delta_v_mps" => 0.0,
            "mean_delta_v_cmps" => 0.0,
            "rms_delta_v_mps" => 0.0,
            "rms_delta_v_cmps" => 0.0,
        )
    end

    dvs = [Float64(e.dv_norm_mps) for e in entries]
    total = sum(dvs)
    meanv = total / length(dvs)
    rmsv = sqrt(sum(abs2, dvs) / length(dvs))
    maxv = maximum(dvs)

    return Dict{String,Any}(
        "count" => length(dvs),
        "total_delta_v_mps" => total,
        "total_delta_v_cmps" => 100.0 * total,
        "max_delta_v_mps" => maxv,
        "max_delta_v_cmps" => 100.0 * maxv,
        "mean_delta_v_mps" => meanv,
        "mean_delta_v_cmps" => 100.0 * meanv,
        "rms_delta_v_mps" => rmsv,
        "rms_delta_v_cmps" => 100.0 * rmsv,
    )
end


"""
    summarize_ocp_control_mps(ocp_control, parameters)

Summarize the optimizer-side control matrix, e.g. `solution.u`, separately from
velocity jumps reconstructed from adjacent coast arcs. If the matrix has at
least four rows, row 4 is treated as the OCP scalar magnitude/slack variable.
Rows 1:3 are always summarized as vector-norm controls when available.
"""
function summarize_ocp_control_mps(ocp_control, parameters)
    nrow = size(ocp_control, 1)
    ncol = size(ocp_control, 2)
    vu_to_mps = Float64(parameters.VU) * 1000.0

    summary = Dict{String,Any}(
        "source" => "ocp_control keyword",
        "count" => ncol,
        "units" => "m/s",
    )

    if nrow >= 3
        vec_norms_nd = [sqrt(sum(abs2, ocp_control[1:3, k])) for k in 1:ncol]
        vec_norms_mps = Float64.(vec_norms_nd) .* vu_to_mps
        total_vec = sum(vec_norms_mps)

        summary["total_control_vector_norm_mps"] = total_vec
        summary["total_control_vector_norm_cmps"] = 100.0 * total_vec
        summary["max_control_vector_norm_mps"] = isempty(vec_norms_mps) ? 0.0 : maximum(vec_norms_mps)
        summary["max_control_vector_norm_cmps"] = 100.0 * summary["max_control_vector_norm_mps"]
        summary["mean_control_vector_norm_mps"] = isempty(vec_norms_mps) ? 0.0 : total_vec / ncol
        summary["mean_control_vector_norm_cmps"] = 100.0 * summary["mean_control_vector_norm_mps"]
        summary["cost_convention_vector_norm"] = "sum(norm.(ocp_control[1:3,k])) * VU * 1000"
    end

    if nrow >= 4
        scalar_nd = Float64.(ocp_control[4, :])
        scalar_mps = scalar_nd .* vu_to_mps
        total_scalar = sum(scalar_mps)

        summary["total_control_scalar_mps"] = total_scalar
        summary["total_control_scalar_cmps"] = 100.0 * total_scalar
        summary["max_control_scalar_mps"] = isempty(scalar_mps) ? 0.0 : maximum(scalar_mps)
        summary["max_control_scalar_cmps"] = 100.0 * summary["max_control_scalar_mps"]
        summary["mean_control_scalar_mps"] = isempty(scalar_mps) ? 0.0 : total_scalar / ncol
        summary["mean_control_scalar_cmps"] = 100.0 * summary["mean_control_scalar_mps"]
        summary["cost_convention_scalar"] = "sum(ocp_control[4,:]) * VU * 1000"
    end

    return summary
end


"""
    collect_ocp_control_maneuvers_mps(ocp_control_times, ocp_control, et0, parameters)

Collect optimizer-commanded impulse entries from an OCP control matrix, e.g.
`solution.u`, and a matching nondimensional time vector. Rows 1:3 are treated
as the commanded vector impulse. Row 4, when present, is treated as the scalar
magnitude/slack variable used by the objective.
"""
function collect_ocp_control_maneuvers_mps(ocp_control_times, ocp_control, et0, parameters)
    nrow = size(ocp_control, 1)
    ncol = size(ocp_control, 2)
    nrow >= 3 || error("`ocp_control` must have at least 3 rows for DVX/DVY/DVZ.")
    length(ocp_control_times) == ncol || error("`ocp_control_times` length ($(length(ocp_control_times))) must match number of OCP control columns ($(ncol)).")

    vu_to_mps = Float64(parameters.VU) * 1000.0
    entries = Any[]

    for k in 1:ncol
        t_nd = Float64(ocp_control_times[k])
        et = Float64(et0 + t_nd * parameters.TU)

        dvx_mps = Float64(ocp_control[1, k]) * vu_to_mps
        dvy_mps = Float64(ocp_control[2, k]) * vu_to_mps
        dvz_mps = Float64(ocp_control[3, k]) * vu_to_mps
        dv_norm_mps = sqrt(dvx_mps^2 + dvy_mps^2 + dvz_mps^2)

        dv_scalar_mps = nrow >= 4 ? Float64(ocp_control[4, k]) * vu_to_mps : nothing

        push!(entries, (
            index = k,
            t_nd = t_nd,
            et = et,
            dvx_mps = dvx_mps,
            dvy_mps = dvy_mps,
            dvz_mps = dvz_mps,
            dv_norm_from_vector_mps = dv_norm_mps,
            dv_norm_from_vector_cmps = 100.0 * dv_norm_mps,
            dv_scalar_u4_mps = dv_scalar_mps,
            dv_scalar_u4_cmps = dv_scalar_mps === nothing ? nothing : 100.0 * dv_scalar_mps,
        ))
    end

    return entries
end

"""
    summarize_ocp_control_entries_mps(entries)

Summarize OCP-commanded maneuver entries. Includes both the sum of vector norms
and, when row 4 exists, the sum of the scalar/slack magnitude used in the OCP
objective.
"""
function summarize_ocp_control_entries_mps(entries)
    summary = Dict{String,Any}(
        "source" => "ocp_control keyword",
        "count" => length(entries),
        "units" => "m/s",
        "format" => "ETSECONDS,DVX_mps,DVY_mps,DVZ_mps,DV_norm_from_vector_mps,DV_scalar_u4_mps",
    )

    vec_norms = [Float64(e.dv_norm_from_vector_mps) for e in entries]
    if isempty(vec_norms)
        summary["total_control_vector_norm_mps"] = 0.0
        summary["total_control_vector_norm_cmps"] = 0.0
        summary["max_control_vector_norm_mps"] = 0.0
        summary["max_control_vector_norm_cmps"] = 0.0
        summary["mean_control_vector_norm_mps"] = 0.0
        summary["mean_control_vector_norm_cmps"] = 0.0
    else
        total_vec = sum(vec_norms)
        max_vec = maximum(vec_norms)
        summary["total_control_vector_norm_mps"] = total_vec
        summary["total_control_vector_norm_cmps"] = 100.0 * total_vec
        summary["max_control_vector_norm_mps"] = max_vec
        summary["max_control_vector_norm_cmps"] = 100.0 * max_vec
        summary["mean_control_vector_norm_mps"] = total_vec / length(vec_norms)
        summary["mean_control_vector_norm_cmps"] = 100.0 * summary["mean_control_vector_norm_mps"]
    end
    summary["cost_convention_vector_norm"] = "sum(norm.(ocp_control[1:3,k])) * VU * 1000"

    scalars = Float64[]
    for e in entries
        if e.dv_scalar_u4_mps !== nothing
            push!(scalars, Float64(e.dv_scalar_u4_mps))
        end
    end

    if !isempty(scalars)
        total_scalar = sum(scalars)
        max_scalar = maximum(scalars)
        summary["total_control_scalar_mps"] = total_scalar
        summary["total_control_scalar_cmps"] = 100.0 * total_scalar
        summary["max_control_scalar_mps"] = max_scalar
        summary["max_control_scalar_cmps"] = 100.0 * max_scalar
        summary["mean_control_scalar_mps"] = total_scalar / length(scalars)
        summary["mean_control_scalar_cmps"] = 100.0 * summary["mean_control_scalar_mps"]
        summary["cost_convention_scalar"] = "sum(ocp_control[4,:]) * VU * 1000"
    end

    return summary
end

"""
    write_ocp_control_entries_mps(outpath, entries)

Write OCP-commanded maneuver entries to a CSV-like text file. This is separate
from the trajectory-jump maneuver file because it represents the optimizer-side
control variables, not reconstructed velocity discontinuities between coast arcs.
"""
function write_ocp_control_entries_mps(outpath::AbstractString, entries)
    open(outpath, "w") do io
        println(io, "# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps,DV_norm_from_vector_mps,DV_scalar_u4_mps")
        for entry in entries
            scalar = entry.dv_scalar_u4_mps === nothing ? NaN : entry.dv_scalar_u4_mps
            @printf(io, "%.9f,%.15e,%.15e,%.15e,%.15e,%.15e\n",
                entry.et,
                entry.dvx_mps,
                entry.dvy_mps,
                entry.dvz_mps,
                entry.dv_norm_from_vector_mps,
                scalar,
            )
        end
    end

    return outpath
end

"""
    write_maneuver_entries_mps(outpath, entries)

Write maneuver entries to a CSV-like text file.

Output format:
`# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps`
"""
function write_maneuver_entries_mps(outpath::AbstractString, entries)
    open(outpath, "w") do io
        println(io, "# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps")
        for entry in entries
            @printf(io, "%.9f,%.15e,%.15e,%.15e\n",
                entry.et, entry.dvx_mps, entry.dvy_mps, entry.dvz_mps)
        end
    end

    return outpath
end

"""
    write_node_to_node_maneuvers_mps(outpath, sols, et0, parameters)

Write nominal node-to-node delta-Vs. Each row is the instantaneous velocity jump
between `sols[k]` end and `sols[k+1]` start.

Output format:
`# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps`
"""
function write_node_to_node_maneuvers_mps(
    outpath::AbstractString,
    sols,
    et0,
    parameters,
)
    entries = collect_node_to_node_maneuvers_mps(sols, et0, parameters)
    return write_maneuver_entries_mps(outpath, entries)
end


# =============================================================================
# Incremental SPK writing helpers for Monte-Carlo / station-keeping recursion
# =============================================================================

"""
    prepare_spk_output!(output_spk; overwrite=true)

Prepare a final `.bsp` file path for a new SPK build. This removes an existing
file only when `overwrite=true`, creates the parent folder, and returns the
absolute output path.

This is useful for Monte-Carlo station-keeping runs where each seed writes its
own kernel before the recursion loop starts.
"""
function prepare_spk_output!(output_spk::AbstractString; overwrite::Bool = true)
    output_spk_abs = abspath(output_spk)
    splitext(output_spk_abs)[2] == ".bsp" || error("`output_spk` must end in `.bsp`: $output_spk_abs")
    mkpath(dirname(output_spk_abs))

    if isfile(output_spk_abs)
        if overwrite
            try
                rm(output_spk_abs; force = true)
            catch err
                error("Could not remove existing SPK: $output_spk_abs. If it is furnished/loaded in SPICE or open elsewhere, close/unload it first. Original error: $err")
            end
        else
            error("Output SPK already exists and `overwrite=false`: $output_spk_abs")
        end
    end

    return output_spk_abs
end

"""
    write_solution_segment_states_for_spk!(sol, et0, parameters; kwargs...)

Write one MKSPK `STATES` file from one ODE/SCP solution segment. The solution
is assumed to use nondimensional time and nondimensional states, consistent
with `ode_sol_to_spk`.
"""
function write_solution_segment_states_for_spk!(
    sol,
    et0,
    parameters;
    segment_index::Integer,
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 0.0,
    outdir::AbstractString = "states_segmented",
)
    dt = Float64(dt_sec)
    gap = Float64(segment_gap_sec)
    dt > 0 || error("`dt_sec` must be positive.")
    gap >= 0 || error("`segment_gap_sec` must be nonnegative.")

    isdir(outdir) || mkpath(outdir)

    et_start = Float64(et0 + sol.t[1] * parameters.TU)
    et_end_true = Float64(et0 + sol.t[end] * parameters.TU)
    et_end = et_end_true - gap
    et_end > et_start || error("Bad segment span after applying segment_gap_sec=$(gap).")

    ts_et = build_segment_epochs(et_start, et_end; dt_sec = dt)

    cols = Vector{Vector{Float64}}()
    for et in ts_et
        t_nd = (et - et0) / parameters.TU
        x_nd = sol(t_nd)
        length(x_nd) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_nd)).")

        r_km = x_nd[1:3] .* parameters.DU
        v_kmps = x_nd[4:6] .* parameters.VU
        push!(cols, vcat(Float64.(r_km), Float64.(v_kmps)))
    end

    length(ts_et) > 1 || error("Segment $(segment_index) produced <2 points. Reduce dt_sec or check segment duration.")

    Y = reduce(hcat, cols)
    tag = lpad(string(segment_index), 3, '0')
    outpath = joinpath(outdir, "seg_$(tag)_states.txt")
    write_mkspk_states_file(outpath, ts_et, Y)

    return (
        state_file = outpath,
        epoch_range = (ts_et[1], ts_et[end]),
        point_count = length(ts_et),
    )
end

"""
    append_state_file_to_spk!(state_file; output_spk, segment_index, kwargs...)

Write the corresponding MKSPK setup file for `state_file` and create/append it
into `output_spk`. If `append=nothing`, the function appends when `output_spk`
already exists and creates it otherwise.
"""
function append_state_file_to_spk!(
    state_file::AbstractString;
    output_spk::AbstractString,
    segment_index::Integer,
    setup_dir::AbstractString = "setup_segmented",
    append::Union{Nothing,Bool} = nothing,
    mkspk_cmd::AbstractString = "mkspk",
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
    verbose::Bool = false,
    suppress_mkspk_output::Bool = true,
)
    isfile(state_file) || error("Missing state file: $state_file")
    isdir(setup_dir) || mkpath(setup_dir)

    output_spk_abs = abspath(output_spk)
    tag = lpad(string(segment_index), 3, '0')
    setup_path = joinpath(setup_dir, "seg_$(tag)_setup.txt")
    segid = segment_id_per_seg ? "$(segment_id)_$(tag)" : String(segment_id)

    write_full_mkspk_setup_exact(
        setup_path;
        segment_id = segid,
        states_file_for_epochs = state_file,
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
    )

    append_flag = append === nothing ? isfile(output_spk_abs) : Bool(append)

    wrap_mkspk(
        setup_path,
        state_file,
        output_spk_abs;
        mkspk_cmd = mkspk_cmd,
        append = append_flag,
        overwrite = false,
        verbose = verbose,
        suppress_output = suppress_mkspk_output,
    )

    return (
        output_spk = output_spk_abs,
        state_file = state_file,
        setup_file = setup_path,
        appended = append_flag,
        epoch_range = _epoch_range_from_states_file(state_file),
    )
end

"""
    append_solution_segment_to_spk!(sol, et0, parameters; output_spk, segment_index, kwargs...)

One-call helper for station-keeping recursion. It writes one `STATES` file from
`sol`, writes one MKSPK setup file, and creates/appends that segment into the
run-specific SPK kernel.

Recommended Monte-Carlo pattern:

```julia
output_spk = prepare_spk_output!("seed_001_recurse.bsp")

for idx_start in 1:N_recurse
    sol_recurse = solve(...)
    append_solution_segment_to_spk!(
        sol_recurse,
        et0,
        parameters;
        output_spk = output_spk,
        segment_index = idx_start,
        append = idx_start > 1,
    )
end
```
"""
function append_solution_segment_to_spk!(
    sol,
    et0,
    parameters;
    output_spk::AbstractString,
    segment_index::Integer,
    states_dir::AbstractString = "states_segmented",
    setup_dir::AbstractString = "setup_segmented",
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 0.0,
    kwargs...,
)
    state_result = write_solution_segment_states_for_spk!(
        sol,
        et0,
        parameters;
        segment_index = segment_index,
        dt_sec = dt_sec,
        segment_gap_sec = segment_gap_sec,
        outdir = states_dir,
    )

    append_result = append_state_file_to_spk!(
        state_result.state_file;
        output_spk = output_spk,
        segment_index = segment_index,
        setup_dir = setup_dir,
        kwargs...,
    )

    return merge(state_result, append_result)
end

function append_solution_segment_to_spk!(sol, parameters; kwargs...)
    et0 = _get_et0_from_parameters(parameters)
    et0 === nothing && error("No `et0` argument was provided and `parameters` has no recognized ET0 property. Tried: et0, ET0, epoch0, et_start, t0_et.")
    return append_solution_segment_to_spk!(sol, Float64(et0), parameters; kwargs...)
end

# =============================================================================
# Metadata JSON helpers
# =============================================================================

"""
    build_spk_metadata(; kwargs...) -> Dict

Build metadata for the generated SPK, including force-model information,
NAIF frame, coverage windows, time/scaling units, and maneuver-file metadata.
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

# =============================================================================
# Optional transformation helper copied from the old transformation.jl
# =============================================================================

"""
    cr3bp_to_j2000(μ, x_cr3bp, et, DU, TU)

Transform a CR3BP rotating-frame state to J2000.
Requires SPICE.jl to be loaded and the `EARTHMOONROTATINGMC` frame furnished.
"""
function cr3bp_to_j2000(μ::Float64, x_cr3bp::Vector, et::Float64, DU::Float64, TU::Float64)
    x_cr3bp_MC = [x_cr3bp[1] - (1 - μ); x_cr3bp[2:6]]
    x_cr3bp_MC_dim = [x_cr3bp_MC[1:3] * DU; x_cr3bp_MC[4:6] * DU / TU]
    T_EMrot_to_J2000 = sxform("EARTHMOONROTATINGMC", "J2000", et)
    return T_EMrot_to_J2000 * x_cr3bp_MC_dim
end