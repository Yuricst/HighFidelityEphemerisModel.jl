"""State sampling helpers for SPK generation"""

# The SPK writers sample states in memory and write the BSP directly with native
# SPICE routines.


"""
    default_coast_windows(sols)

Default segmentation: one SPK segment per coast arc.
This avoids interpolating across impulsive velocity jumps.
"""
default_coast_windows(sols) = [(k, k) for k in 1:length(sols)]


# Absolute tolerance used for the time-continuity check between merged arcs,
# in seconds. Arc boundaries produced by restarting an integration at
# `sol.t[end]` match to floating-point round-off, far below this value.
const _COAST_WINDOW_TIME_TOL_SEC = 1e-6


"""
    _validate_coast_window_continuity(seg_sols, window, parameters; ...)

Reject a custom coast window that groups arcs across a time gap/overlap or a
state discontinuity (for example an impulsive maneuver). Fitting one type-13
segment across such a jump silently corrupts interpolated SPK states.

Each state component must satisfy `abs(Δ) <= atol + continuity_rtol * scale`,
where `Δ` is the jump between `left(t_end)` and `right(t_start)` in physical
units, `scale = max(abs(x_left), abs(x_right))`, and `atol` is
`continuity_pos_tol_km` (position, km) or `continuity_vel_tol_kmps` (velocity,
km/s). Time gaps/overlaps larger than $(_COAST_WINDOW_TIME_TOL_SEC) s are always
rejected. Non-finite times or states are rejected.

`offset` is the global index of `seg_sols[1]` in the full arc list; it is only
used in error messages.
"""
function _validate_coast_window_continuity(
    seg_sols,
    window,
    parameters;
    continuity_pos_tol_km::Real = 1e-3,
    continuity_vel_tol_kmps::Real = 1e-6,
    continuity_rtol::Real = 0.0,
    offset::Integer = 1,
)
    length(seg_sols) <= 1 && return nothing

    DU = Float64(parameters.DU)
    VU = Float64(parameters.VU)
    TU = Float64(parameters.TU)
    pos_atol = Float64(continuity_pos_tol_km)
    vel_atol = Float64(continuity_vel_tol_kmps)
    rtol = Float64(continuity_rtol)

    for k in 1:(length(seg_sols) - 1)
        left = seg_sols[k]
        right = seg_sols[k + 1]
        gk = offset + k - 1
        loc = "Coast window $(window): boundary between arcs $(k) and $(k + 1) of the window " *
                "(global arcs $(gk) and $(gk + 1))"

        t_left = left.t[end]
        t_right = right.t[1]
        if !(isfinite(t_left) && isfinite(t_right))
            error("$(loc) has non-finite boundary times: t_end = $(t_left), t_start = $(t_right).")
        end
        dt_sec = abs(Float64(t_left) - Float64(t_right)) * TU
        if dt_sec > _COAST_WINDOW_TIME_TOL_SEC
            error(
                "$(loc) has a time gap/overlap of $(dt_sec) s " *
                "(t_end = $(t_left), t_start = $(t_right) in canonical time). " *
                "Use separate windows at discontinuities."
            )
        end

        x_left = left(t_left)
        x_right = right(t_right)
        length(x_left) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_left)).")
        length(x_right) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_right)).")

        scales = vcat(fill(DU, 3), fill(VU, 3))
        xl = Float64.(x_left[1:6]) .* scales
        xr = Float64.(x_right[1:6]) .* scales

        if !(all(isfinite, xl) && all(isfinite, xr))
            error(
                "$(loc) at t = $(t_left) has non-finite states: " *
                "left(t_end) = $(xl), right(t_start) = $(xr)."
            )
        end

        Δ = abs.(xr .- xl)
        bound = vcat(fill(pos_atol, 3), fill(vel_atol, 3)) .+ rtol .* max.(abs.(xl), abs.(xr))
        any(Δ .> bound) || continue

        pos_jump_km = maximum(Δ[1:3])
        vel_jump_kmps = maximum(Δ[4:6])

        msg = "$(loc) at t = $(t_left) (canonical) crosses a discontinuous state jump: " *
              "position jump $(pos_jump_km) km (tol $(pos_atol) km), " *
              "velocity jump $(vel_jump_kmps) km/s (tol $(vel_atol) km/s), " *
              "rtol = $(rtol). "

        # A `DiscreteCallback` firing at the final time leaves a duplicated
        # final time in `sol.t`; `sol(t_end)` then returns the pre-impulse
        # state while `sol.u[end]` is post-impulse. The jump is then inside
        # arc `k`, not between the two arcs.
        if length(left.t) >= 2 && left.t[end] == left.t[end - 1]
            msg *= "Arc $(k) of the window (global arc $(gk)) ends with a repeated time " *
                   "t = $(t_left), which usually means a callback applied an impulse at its " *
                   "final time: the discontinuity is inside that arc, because `sol(t_end)` " *
                   "returns the pre-impulse state. The SPK sampler also evaluates `sol(t)`, " *
                   "so this arc cannot be merged with the next one. "
        end

        msg *= "Use separate windows around impulsive maneuvers, or loosen " *
               "`continuity_pos_tol_km` / `continuity_vel_tol_kmps` / `continuity_rtol` " *
               "(or pass `validate_continuity = false`) if the defect is acceptable."
        error(msg)
    end

    return nothing
end


"""
    _validate_coast_windows(sols, windows, parameters; validate_continuity=true, kwargs...)

Check every coast window in one pass, before any output is written: each window
must be a pair `(a, b)` with `1 <= a <= b <= length(sols)`, and (unless
`validate_continuity = false`) merged arcs must be continuous in time and state
(see `_validate_coast_window_continuity` for the tolerance keywords).
"""
function _validate_coast_windows(
    sols,
    windows,
    parameters;
    validate_continuity::Bool = true,
    continuity_pos_tol_km::Real = 1e-3,
    continuity_vel_tol_kmps::Real = 1e-6,
    continuity_rtol::Real = 0.0,
)
    continuity_pos_tol_km >= 0 || error("`continuity_pos_tol_km` must be nonnegative.")
    continuity_vel_tol_kmps >= 0 || error("`continuity_vel_tol_kmps` must be nonnegative.")
    continuity_rtol >= 0 || error("`continuity_rtol` must be nonnegative.")

    n = length(sols)
    for window in windows
        length(window) == 2 || error("Bad coast window $(window): expected a pair `(a, b)`.")
        a, b = window
        (a isa Integer && b isa Integer) || error("Bad coast window $(window): indices must be integers.")
        1 <= a <= b <= n || error("Bad coast window $(window) for $(n) coast arcs.")

        validate_continuity || continue
        _validate_coast_window_continuity(
            sols[a:b],
            window,
            parameters;
            continuity_pos_tol_km = continuity_pos_tol_km,
            continuity_vel_tol_kmps = continuity_vel_tol_kmps,
            continuity_rtol = continuity_rtol,
            offset = a,
        )
    end

    return nothing
end


"""
    build_segment_epochs(et_start, et_end; dt_sec=1800.0)

Build a near-uniform epoch grid from `et_start` to `et_end`, always including
the endpoint exactly once. If the last uniform node would fall closer than
`dt_sec/2` to `et_end`, it is dropped so the final step lies in
`[dt_sec/2, 3dt_sec/2)`; this avoids nearly coincident nodes, which make
Hermite (type 13) interpolation ill-conditioned.
"""
function build_segment_epochs(et_start::Float64, et_end::Float64; dt_sec::Float64 = 1800.0)
    @assert et_end > et_start
    @assert dt_sec > 0

    # Build the grid manually instead of using a range so the final endpoint
    # matches `et_end` exactly. Nodes are computed as et_start + k*dt to avoid
    # accumulating round-off.
    ts = Float64[et_start]
    k = 1
    while true
        t = et_start + k * dt_sec
        et_end - t < 0.5 * dt_sec && break
        push!(ts, t)
        k += 1
    end
    push!(ts, et_end)

    return ts
end

"""
    sample_segmented_states_for_spk(sols, coast_windows, et0, parameters; ...)

Sample ODE solution segments into in-memory SPK type-13 inputs.

Epochs are returned in seconds past J2000 TDB. States are returned as vectors
`[x, y, z, vx, vy, vz]` in km and km/s.

`coast_windows` is a collection of arc-index pairs `(a, b)`. A window with
`a < b` merges arcs `a:b` into one SPK segment, so those arcs must be
continuous in time and state. All windows are validated before anything is
written or sampled; an invalid window raises an error. Continuity keywords:
- `validate_continuity=true`: set to `false` to skip the continuity check.
- `continuity_pos_tol_km=1e-3`: absolute position tolerance per component, km.
- `continuity_vel_tol_kmps=1e-6`: absolute velocity tolerance per component, km/s.
- `continuity_rtol=0.0`: relative tolerance; a component passes if
  `abs(Δ) <= atol + continuity_rtol * max(abs(x_left), abs(x_right))`.
Time gaps/overlaps above 1e-6 s and non-finite boundary states are always
rejected when the check is enabled.
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
    validate_continuity::Bool = true,
    continuity_pos_tol_km::Real = 1e-3,
    continuity_vel_tol_kmps::Real = 1e-6,
    continuity_rtol::Real = 0.0,
)
    # Validate every window up front so a bad window cannot leave partial output.
    _validate_coast_windows(
        sols,
        coast_windows,
        parameters;
        validate_continuity = validate_continuity,
        continuity_pos_tol_km = continuity_pos_tol_km,
        continuity_vel_tol_kmps = continuity_vel_tol_kmps,
        continuity_rtol = continuity_rtol,
    )

    tol_nd = 1e-10
    tol_et = 1e-9
    nseg = length(coast_windows)
    progress_enabled = verbose && show_progress

    segments = Vector{NamedTuple}()
    epoch_ranges = Tuple{Float64,Float64}[]

    for (seg_idx, window) in enumerate(coast_windows)
        a, b = window

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
