"""State-file writers for SPK generation"""


"""
    default_coast_windows(sols)

Default segmentation: one SPK segment per coast arc.
This avoids interpolating across impulsive velocity jumps.
"""
default_coast_windows(sols) = [(k, k) for k in 1:length(sols)]

function _validate_coast_window_continuity(seg_sols, window; continuity_tol::Float64 = 1e-8)
    length(seg_sols) <= 1 && return nothing

    for k in 1:(length(seg_sols) - 1)
        left = seg_sols[k]
        right = seg_sols[k + 1]

        t_left = left.t[end]
        t_right = right.t[1]
        if abs(t_left - t_right) > continuity_tol
            error("Coast window $(window) joins arcs with a time gap/overlap: $(t_left) vs $(t_right). Use separate windows at discontinuities.")
        end

        x_left = left(t_left)
        x_right = right(t_right)
        length(x_left) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_left)).")
        length(x_right) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_right)).")

        jump = maximum(abs.(Float64.(x_right[1:6]) .- Float64.(x_left[1:6])))
        if jump > continuity_tol
            error("Coast window $(window) crosses a discontinuous state jump of $(jump). Use separate windows around impulsive maneuvers.")
        end
    end

    return nothing
end

"""
    build_segment_epochs(et_start, et_end; dt_sec=1800.0)

Build a uniform epoch grid from `et_start` to `et_end`, always including the
endpoint exactly once.

# Arguments
- `et_start::Float64`: segment start epoch in seconds past J2000
- `et_end::Float64`: segment end epoch in seconds past J2000
- `dt_sec::Float64`: nominal sampling interval in seconds
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

"""
    write_mkspk_states_file(filepath, ts_et, Y)

Write one MKSPK `STATES` input file.

# Arguments
- `filepath`: output text file path
- `ts_et`: epochs in seconds past J2000
- `Y`: 6-by-N dimensional state matrix in km and km/s
"""
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

# Arguments
- `sols`: vector of coast-arc ODE solutions
- `coast_windows`: vector of `(start_index, end_index)` coast-arc windows
- `et0`: reference epoch in seconds past J2000
- `parameters`: object containing `TU`, `DU`, and `VU`
"""
function write_segmented_states_for_spk!(
    sols,
    coast_windows,
    et0,
    parameters;
    dt_sec::Float64 = 1800.0,
    segment_gap_sec::Float64 = 1e-7,
    continuity_tol::Float64 = 1e-8,
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
        _validate_coast_window_continuity(seg_sols, window; continuity_tol = continuity_tol)

        et_start_true = Float64(et0 + seg_sols[1].t[1]     * parameters.TU)
        et_end_true   = Float64(et0 + seg_sols[end].t[end] * parameters.TU)

        et_start = et_start_true
        # Avoid overlapping SPK coverage at impulsive segment boundaries.
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

            # MKSPK expects dimensional states.
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
        continuity_tol = 1e-8,
        outdir = outdir,
        verbose = true,
    )
    return outdir
end
