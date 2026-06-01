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
