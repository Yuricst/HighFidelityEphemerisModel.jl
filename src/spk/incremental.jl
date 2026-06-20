"""Incremental SPK writing helpers"""

# These helpers support the station-keeping workflow where each propagated ODE
# arc is appended to the same BSP immediately. That avoids storing all
# `ODESolution` objects in memory until the end of a long Monte Carlo run.


"""
    prepare_spk_output!(output_spk; overwrite=true)

Prepare a final `.bsp` file path for a new SPK build. This removes an existing
file only when `overwrite=true`, creates the parent folder, and returns the
absolute output path.

For the high-level `ode_sol_to_spk` pipeline, the existing BSP is preserved until
a replacement kernel has been fully written and closed.
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

Write one text `STATES` file from one ODE solution segment. This helper is kept
for debugging and backwards compatibility. The main SPK pipeline now writes BSP
files directly with native SPICE type-13 routines.
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

    # Build one in-memory segment from this one ODE solution. This is the
    # canonical SK append workflow requested for long simulations.
    et_start = Float64(et0 + sol.t[1] * parameters.TU)
    et_end_true = Float64(et0 + sol.t[end] * parameters.TU)
    et_end = et_end_true - gap
    et_end > et_start || error("Bad segment span after applying segment_gap_sec=$(gap).")

    # Sample only this one propagated arc. The high-level append path sends the
    # sampled points directly to `spkw13`; the text file path is for debugging.
    ts_et = build_segment_epochs(et_start, et_end; dt_sec = dt)

    cols = Vector{Vector{Float64}}()
    for et in ts_et
        t_nd = (et - et0) / parameters.TU
        x_nd = sol(t_nd)
        length(x_nd) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_nd)).")

        # Convert canonical ODE states to the dimensional units required by SPK.
        r_km = x_nd[1:3] .* parameters.DU
        v_kmps = x_nd[4:6] .* parameters.VU
        push!(cols, vcat(Float64.(r_km), Float64.(v_kmps)))
    end

    length(ts_et) > 1 || error("Segment $(segment_index) produced <2 points. Reduce dt_sec or check segment duration.")

    Y = reduce(hcat, cols)
    tag = lpad(string(segment_index), 3, '0')
    outpath = joinpath(outdir, "seg_$(tag)_states.txt")
    write_spk_states_file(outpath, ts_et, Y)

    return (
        state_file = outpath,
        epoch_range = (ts_et[1], ts_et[end]),
        point_count = length(ts_et),
    )
end

"""
    append_state_file_to_spk!(state_file; output_spk, segment_index, kwargs...)

Create or append one native SPICE type-13 segment from an existing text `STATES`
file. This is a backwards-compatible debugging helper; the preferred path is to
sample ODE solutions in memory and call `append_solution_segment_to_spk!`.
"""
function append_state_file_to_spk!(
    state_file::AbstractString;
    output_spk::AbstractString,
    segment_index::Integer,
    setup_dir::AbstractString = "setup_segmented",
    append::Union{Nothing,Bool} = nothing,
    kwargs...,
)
    # Compatibility path: read a debug state file and append it through the same
    # native type-13 writer used by the in-memory path.
    segment = _read_spk_states_file_for_spkw13(state_file)
    append_flag = append === nothing ? isfile(abspath(output_spk)) : Bool(append)

    output_spk_abs = append_spkw13_segment_to_spk!(
        segment;
        output_spk = output_spk,
        segment_index = segment_index,
        append = append_flag,
        kwargs...,
    )

    return (
        output_spk = output_spk_abs,
        state_file = state_file,
        setup_file = nothing,
        appended = append_flag,
        epoch_range = (segment.epochs[1], segment.epochs[end]),
    )
end

"""
    append_solution_segment_to_spk!(sol, et0, parameters; output_spk, segment_index, kwargs...)

One-call helper for station-keeping recursion. It samples one ODE solution
segment and creates/appends it to a run-specific SPK kernel using native SPICE
type-13 routines.
"""
function append_solution_segment_to_spk!(
    sol,
    et0,
    parameters;
    output_spk::AbstractString,
    segment_index::Integer,
    append::Union{Nothing,Bool} = nothing,
    overwrite::Bool = true,
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 0.0,
    kwargs...,
)
    output_spk_abs = abspath(output_spk)
    append_flag = append === nothing ? isfile(output_spk_abs) : Bool(append)

    if append_flag
        isfile(output_spk_abs) || error("Cannot append: output SPK does not exist: $(output_spk_abs)")
    elseif isfile(output_spk_abs)
        if overwrite
            rm(output_spk_abs; force = true)
        else
            error("Output SPK already exists and `overwrite=false`: $(output_spk_abs)")
        end
    end

    state_result = sample_segmented_states_for_spk(
        [sol],
        [(1, 1)],
        et0,
        parameters;
        dt_sec = Float64(dt_sec),
        segment_gap_sec = Float64(segment_gap_sec),
        verbose = false,
        show_progress = false,
    )

    segment = state_result.segments[1]
    output_spk_abs = append_spkw13_segment_to_spk!(
        segment;
        output_spk = output_spk_abs,
        segment_index = segment_index,
        append = append_flag,
        kwargs...,
    )

    return (
        output_spk = output_spk_abs,
        state_file = nothing,
        setup_file = nothing,
        appended = append_flag,
        epoch_range = state_result.epoch_ranges[1],
        point_count = state_result.point_counts[1],
    )
end

function append_solution_segment_to_spk!(sol, parameters; kwargs...)
    et0 = _get_et0_from_parameters(parameters)
    et0 === nothing && error("No `et0` argument was provided and `parameters` has no recognized ET0 property. Tried: et0, ET0, epoch0, et_start, t0_et.")
    return append_solution_segment_to_spk!(sol, Float64(et0), parameters; kwargs...)
end
