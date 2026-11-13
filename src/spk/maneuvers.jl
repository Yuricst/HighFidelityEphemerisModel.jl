"""Maneuver-file writers for SPK generation"""

# Two maneuver products are supported:
#   1. trajectory-jump diagnostics reconstructed from adjacent coast arcs,
#   2. OCP/executed controls passed explicitly through `ocp_control`.
# Station-keeping truth products usually use (2) as the primary maneuver file.


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

        # This is a diagnostic jump implied by the state history itself. It is
        # not necessarily the same as the commanded/executed control from an OCP.
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
    summary = Dict{String,Any}("count" => length(entries))
    _add_dv_stats!(summary, "delta_v", [Float64(e.dv_norm_mps) for e in entries]; rms = true)
    return summary
end


"""
    _add_dv_stats!(summary, name, values_mps; rms=false)

Add `total_<name>`, `max_<name>`, `mean_<name>` (and optionally `rms_<name>`)
statistics of `values_mps` to `summary`, each in m/s (`_mps` key suffix) and cm/s
(`_cmps`). All statistics are zero for an empty list.
"""
function _add_dv_stats!(summary::AbstractDict, name::AbstractString, values_mps::AbstractVector; rms::Bool = false)
    n = length(values_mps)
    stats = [
        "total" => (n == 0 ? 0.0 : sum(values_mps)),
        "max" => (n == 0 ? 0.0 : maximum(values_mps)),
        "mean" => (n == 0 ? 0.0 : sum(values_mps) / n),
    ]
    rms && push!(stats, "rms" => (n == 0 ? 0.0 : sqrt(sum(abs2, values_mps) / n)))
    for (stat, value) in stats
        summary["$(stat)_$(name)_mps"] = Float64(value)
        summary["$(stat)_$(name)_cmps"] = 100.0 * Float64(value)
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
        # Each row is tied to the control epoch used by the OCP/SK loop.
        t_nd = Float64(ocp_control_times[k])
        et = Float64(et0 + t_nd * parameters.TU)

        dvx_mps = Float64(ocp_control[1, k]) * vu_to_mps
        dvy_mps = Float64(ocp_control[2, k]) * vu_to_mps
        dvz_mps = Float64(ocp_control[3, k]) * vu_to_mps
        dv_norm_mps = sqrt(dvx_mps^2 + dvy_mps^2 + dvz_mps^2)

        # Row 4 is the scalar magnitude used by some OCP formulations.
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
        "cost_convention_vector_norm" => "sum(norm.(ocp_control[1:3,k])) * VU * 1000",
    )
    _add_dv_stats!(summary, "control_vector_norm", [Float64(e.dv_norm_from_vector_mps) for e in entries])

    # Row 4 (scalar magnitude/slack of the OCP objective) is optional.
    scalars = [Float64(e.dv_scalar_u4_mps) for e in entries if e.dv_scalar_u4_mps !== nothing]
    if !isempty(scalars)
        _add_dv_stats!(summary, "control_scalar", scalars)
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
