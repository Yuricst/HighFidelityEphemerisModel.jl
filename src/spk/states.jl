"""State sampling helpers for SPK generation"""

# The high-level SPK path samples states in memory and writes the BSP directly
# with native SPICE routines. Text state-file writers are kept for debugging,
# inspection, and compatibility with earlier experiment scripts.


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

    # Build the grid manually instead of using a range so the final endpoint
    # can be forced to match `et_end` exactly.
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

"""
    write_spk_states_file(filepath, ts_et, Y)

Write one SPK `STATES` input file.

# Arguments
- `filepath`: output text file path
- `ts_et`: epochs in seconds past J2000
- `Y`: 6-by-N dimensional state matrix in km and km/s
"""
function write_spk_states_file(
    filepath::AbstractString,
    ts_et::Vector{Float64},
    Y::Matrix{Float64},
)
    @assert size(Y, 1) == 6
    @assert length(ts_et) == size(Y, 2)

    open(filepath, "w") do io
        # One row per epoch: ET seconds followed by Cartesian state in km/km/s.
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

Write one SPK `STATES` file per `coast_windows` entry and return all written
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

        # Convert the nondimensional ODE time span into ET seconds. The segment
        # may group one coast arc or several adjacent coast arcs.
        et_start_true = Float64(et0 + seg_sols[1].t[1]     * parameters.TU)
        et_end_true   = Float64(et0 + seg_sols[end].t[end] * parameters.TU)

        et_start = et_start_true

        # Avoid overlapping SPK coverage at impulsive segment boundaries. The
        # next segment still starts at the true boundary; only the previous
        # segment is trimmed by the small gap.
        et_end = seg_idx < nseg ? et_end_true - segment_gap_sec : et_end_true

        if et_end <= et_start + tol_et
            error("Segment $(seg_idx) arcs[$a,$b] has bad span after applying segment_gap_sec=$(segment_gap_sec).")
        end

        ts_et = build_segment_epochs(et_start, et_end; dt_sec = dt_sec)

        cols = Vector{Vector{Float64}}()
        sol_ptr = 1

        for et in ts_et
            t_nd = (et - et0) / parameters.TU

            # Move through grouped coast arcs until the current sample time is
            # covered by the active ODE solution.
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

            # SPK writers expect dimensional states.
            r_km   = x_nd[1:3] .* parameters.DU
            v_kmps = x_nd[4:6] .* parameters.VU

            push!(cols, vcat(Float64.(r_km), Float64.(v_kmps)))
        end

        @assert length(ts_et) > 1 "Segment $seg_idx produced <2 points. Reduce dt_sec or check segment duration."

        Y = reduce(hcat, cols)
        tag = lpad(string(seg_idx), 3, '0')
        outpath = joinpath(outdir, "seg_$(tag)_states.txt")
        write_spk_states_file(outpath, ts_et, Y)

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
    sample_segmented_states_for_spk(sols, coast_windows, et0, parameters; ...)

Sample ODE solution segments into in-memory SPK type-13 inputs.

Epochs are returned in seconds past J2000 TDB. States are returned as vectors
`[x, y, z, vx, vy, vz]` in km and km/s.
"""
function sample_segmented_states_for_spk(
    sols,
    coast_windows,
    et0,
    parameters;
    dt_sec::Float64 = 1800.0,
    segment_gap_sec::Float64 = 1e-7,
    verbose::Bool = true,
    show_progress::Bool = true,
)
    tol_nd = 1e-10
    tol_et = 1e-9
    nseg = length(coast_windows)
    progress_enabled = verbose && show_progress

    segments = Vector{NamedTuple}()
    epoch_ranges = Tuple{Float64,Float64}[]

    for (seg_idx, window) in enumerate(coast_windows)
        a, b = window
        @assert 1 <= a <= b <= length(sols) "Bad coast window $(window) for $(length(sols)) coast arcs."

        seg_sols = sols[a:b]

        # Convert the nondimensional ODE time span into ET seconds. The segment
        # may group one coast arc or several adjacent coast arcs.
        et_start_true = Float64(et0 + seg_sols[1].t[1] * parameters.TU)
        et_end_true = Float64(et0 + seg_sols[end].t[end] * parameters.TU)

        et_start = et_start_true
        et_end = seg_idx < nseg ? et_end_true - segment_gap_sec : et_end_true

        if et_end <= et_start + tol_et
            error("Segment $(seg_idx) arcs[$a,$b] has bad span after applying segment_gap_sec=$(segment_gap_sec).")
        end

        ts_et = build_segment_epochs(et_start, et_end; dt_sec = dt_sec)

        states = Vector{Vector{Float64}}()
        sol_ptr = 1

        for et in ts_et
            t_nd = (et - et0) / parameters.TU

            # Move through grouped coast arcs until the current sample time is
            # covered by the active ODE solution.
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

            # SPK type-13 states are dimensional, even though the ODE solution
            # is stored in canonical units.
            r_km = x_nd[1:3] .* parameters.DU
            v_kmps = x_nd[4:6] .* parameters.VU
            push!(states, vcat(Float64.(r_km), Float64.(v_kmps)))
        end

        length(ts_et) > 1 || error("Segment $(seg_idx) produced <2 points. Reduce dt_sec or check segment duration.")

        push!(segments, (
            epochs = ts_et,
            states = states,
            first = ts_et[1],
            last = ts_et[end],
            coast_window = (a, b),
        ))
        push!(epoch_ranges, (ts_et[1], ts_et[end]))

        _print_progress("sampling SPK states", seg_idx, nseg; enabled = progress_enabled)
    end

    verbose && !show_progress && println("All SPK states sampled in memory.")

    return (
        segments = segments,
        epoch_ranges = epoch_ranges,
        point_counts = [length(segment.epochs) for segment in segments],
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
