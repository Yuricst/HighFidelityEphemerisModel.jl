"""Tests for the ode_sol_to_spk pipeline: OCP/maneuver products, metadata, verbose
output, incremental appends, and state-sampling error paths."""

using LinearAlgebra
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


# -----------------------------------------------------------------------------
# Helpers (prefixed `_spkC_` to avoid clashing with other test files)
# -----------------------------------------------------------------------------

"""
Constant-velocity coast arc in canonical units: position is linear in time and
the velocity is constant, so `d(r * DU)/d(t * TU) == v * VU` when `VU = DU / TU`.
Type-13 (Hermite) SPK segments reproduce such arcs exactly up to round-off.
"""
struct _SpkCArc
    t::Vector{Float64}
    r_t0::Vector{Float64}   # position at t[1]
    v::Vector{Float64}
end

(arc::_SpkCArc)(t) = vcat(arc.r_t0 .+ arc.v .* (t - arc.t[1]), arc.v)


# Position-continuous chain of arcs with impulsive velocity changes at the boundaries.
function _spkC_chain(tgrid, r0, vs)
    arcs = _SpkCArc[]
    r = copy(r0)
    for k in eachindex(vs)
        push!(arcs, _SpkCArc([tgrid[k], tgrid[k + 1]], copy(r), copy(vs[k])))
        r = r .+ vs[k] .* (tgrid[k + 1] - tgrid[k])
    end
    return arcs
end


const _SPKC_V1 = [0.01, -0.02, 0.03]
const _SPKC_V2 = _SPKC_V1 .+ [0.002, 0.0, -0.001]   # jump of [20, 0, -10] m/s with VU = 10 km/s
const _SPKC_V3 = _SPKC_V2 .+ [0.0, 0.005, 0.0]      # jump of [0, 50, 0] m/s

_spkC_arcs() = _spkC_chain([0.0, 8.0, 16.0, 20.0], [1.0, 2.0, 3.0], [_SPKC_V1, _SPKC_V2, _SPKC_V3])

_spkC_params(; et0 = 1000.0) = (
    TU = 100.0,   # s
    DU = 1000.0,  # km
    VU = 10.0,    # km/s  (= DU / TU)
    et0 = et0,
    naif_ids = ["399", "301"],
    GMs = [1.0, 0.0123],
    include_srp = true,
    srp_Cr = 1.2,
)

# OCP control matrix (canonical velocity). With VU = 10 km/s, 1 canonical unit = 1e4 m/s:
# column 1: vector (3, 4, 0) m/s -> norm 5 m/s, scalar u4 = 5.5 m/s
# column 2: vector (0, 6, 8) m/s -> norm 10 m/s, scalar u4 = 11 m/s
const _SPKC_OCP = [
    3e-4    0.0;
    4e-4    6e-4;
    0.0     8e-4;
    5.5e-4  1.1e-3;
]


function _spkC_check_spk(spk, arcs, params, t_queries; spice_id, center_id = 399)
    SPICE.furnsh(spk)
    try
        for t in t_queries
            et = params.et0 + t * params.TU
            st, _ = SPICE.spkezr(string(spice_id), et, "J2000", "NONE", string(center_id))
            idx = findfirst(a -> a.t[1] <= t <= a.t[end], arcs)
            x = arcs[idx](t)
            @test st[1:3] ≈ x[1:3] .* params.DU atol = 1e-7
            @test st[4:6] ≈ x[4:6] .* params.VU atol = 1e-10
        end
    finally
        SPICE.unload(spk)
    end
end


# Rows of a "# header" + CSV text file, parsed as Float64 vectors.
function _spkC_read_csv(path)
    lines = readlines(path)
    rows = [parse.(Float64, split(l, ",")) for l in lines[2:end] if !isempty(strip(l))]
    return lines[1], rows
end


# Run `f` with stdout redirected to a temporary file; return `(f(), captured_text)`.
function _spkC_capture(f)
    path, io = mktemp()
    result = try
        redirect_stdout(io) do
            r = f()
            flush(stdout)
            r
        end
    finally
        close(io)
    end
    out = read(path, String)
    rm(path; force = true)
    return result, out
end


function _spkC_error_message(f)
    try
        f()
    catch err
        return sprint(showerror, err)
    end
    return ""
end


# Minimal JSON reader for the files written by `write_spk_metadata_json`.
_spkC_skip_ws(cs, i) = (while i <= length(cs) && isspace(cs[i]); i += 1; end; i)

function _spkC_json(cs, i)
    i = _spkC_skip_ws(cs, i)
    c = cs[i]
    if c == '{'
        d = Dict{String,Any}()
        i = _spkC_skip_ws(cs, i + 1)
        cs[i] == '}' && return d, i + 1
        while true
            k, i = _spkC_json(cs, i)
            i = _spkC_skip_ws(cs, i)
            cs[i] == ':' || error("JSON: expected ':' at $i")
            v, i = _spkC_json(cs, i + 1)
            d[k] = v
            i = _spkC_skip_ws(cs, i)
            if cs[i] == ','
                i += 1
            elseif cs[i] == '}'
                return d, i + 1
            else
                error("JSON: expected ',' or '}' at $i")
            end
        end
    elseif c == '['
        a = Any[]
        i = _spkC_skip_ws(cs, i + 1)
        cs[i] == ']' && return a, i + 1
        while true
            v, i = _spkC_json(cs, i)
            push!(a, v)
            i = _spkC_skip_ws(cs, i)
            if cs[i] == ','
                i += 1
            elseif cs[i] == ']'
                return a, i + 1
            else
                error("JSON: expected ',' or ']' at $i")
            end
        end
    elseif c == '"'
        buf = IOBuffer()
        i += 1
        while cs[i] != '"'
            if cs[i] == '\\'
                e = cs[i + 1]
                print(buf, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e)
                i += 2
            else
                print(buf, cs[i])
                i += 1
            end
        end
        return String(take!(buf)), i + 1
    else
        for (lit, val) in (("true", true), ("false", false), ("null", nothing))
            n = length(lit)
            if i + n - 1 <= length(cs) && String(cs[i:i + n - 1]) == lit
                return val, i + n
            end
        end
        j = i
        while j <= length(cs) && cs[j] in "+-.eE0123456789"
            j += 1
        end
        s = String(cs[i:j - 1])
        v = tryparse(Int, s)
        return (v === nothing ? parse(Float64, s) : v), j
    end
end

_spkC_read_json(path) = first(_spkC_json(collect(read(path, String)), 1))


# -----------------------------------------------------------------------------
# Maneuver helpers
# -----------------------------------------------------------------------------

function test_spkC_maneuver_helpers()
    params = (TU = 100.0, DU = 1.0, VU = 0.01, et0 = 500.0)   # 1 canonical velocity = 10 m/s

    # summarize_ocp_control_mps with a 4-row matrix: vector norms and row-4 scalars.
    U = [
        0.3  0.0  0.0;
        0.4  0.6  0.0;
        0.0  0.8  0.0;
        0.55 1.1  0.0;
    ]
    s4 = HighFidelityEphemerisModel.summarize_ocp_control_mps(U, params)
    @test s4["count"] == 3
    @test s4["units"] == "m/s"
    @test s4["source"] == "ocp_control keyword"
    @test s4["total_control_vector_norm_mps"] ≈ 15.0       # 5 + 10 + 0
    @test s4["total_control_vector_norm_cmps"] ≈ 1500.0
    @test s4["max_control_vector_norm_mps"] ≈ 10.0
    @test s4["max_control_vector_norm_cmps"] ≈ 1000.0
    @test s4["mean_control_vector_norm_mps"] ≈ 5.0
    @test s4["mean_control_vector_norm_cmps"] ≈ 500.0
    @test s4["total_control_scalar_mps"] ≈ 16.5           # 5.5 + 11 + 0
    @test s4["total_control_scalar_cmps"] ≈ 1650.0
    @test s4["max_control_scalar_mps"] ≈ 11.0
    @test s4["max_control_scalar_cmps"] ≈ 1100.0
    @test s4["mean_control_scalar_mps"] ≈ 5.5
    @test s4["mean_control_scalar_cmps"] ≈ 550.0
    @test occursin("ocp_control[1:3,k]", s4["cost_convention_vector_norm"])
    @test occursin("ocp_control[4,:]", s4["cost_convention_scalar"])

    # 3 rows: vector diagnostics only, no scalar (row 4) keys.
    s3 = HighFidelityEphemerisModel.summarize_ocp_control_mps(U[1:3, :], params)
    @test s3["total_control_vector_norm_mps"] ≈ 15.0
    @test !haskey(s3, "total_control_scalar_mps")
    @test !haskey(s3, "cost_convention_scalar")

    # Fewer than 3 rows: only the bookkeeping fields.
    s2 = HighFidelityEphemerisModel.summarize_ocp_control_mps(U[1:2, :], params)
    @test Set(keys(s2)) == Set(["source", "count", "units"])
    @test s2["count"] == 3

    # No control columns: all statistics are zero instead of erroring on max/mean.
    s0 = HighFidelityEphemerisModel.summarize_ocp_control_mps(zeros(4, 0), params)
    @test s0["count"] == 0
    @test s0["total_control_vector_norm_mps"] == 0.0
    @test s0["max_control_vector_norm_mps"] == 0.0
    @test s0["mean_control_vector_norm_mps"] == 0.0
    @test s0["total_control_scalar_mps"] == 0.0
    @test s0["max_control_scalar_mps"] == 0.0
    @test s0["mean_control_scalar_mps"] == 0.0

    # Empty OCP entry list summary.
    se = HighFidelityEphemerisModel.summarize_ocp_control_entries_mps(Any[])
    @test se["count"] == 0
    for key in ("total_control_vector_norm_mps", "total_control_vector_norm_cmps",
                "max_control_vector_norm_mps", "max_control_vector_norm_cmps",
                "mean_control_vector_norm_mps", "mean_control_vector_norm_cmps")
        @test se[key] == 0.0
    end
    @test !haskey(se, "total_control_scalar_mps")

    # Entries from a 3-row control have no scalar; the text file writes NaN there.
    entries3 = HighFidelityEphemerisModel.collect_ocp_control_maneuvers_mps([1.0, 2.0, 3.0], U[1:3, :], params.et0, params)
    @test all(e -> e.dv_scalar_u4_mps === nothing && e.dv_scalar_u4_cmps === nothing, entries3)
    @test [e.et for e in entries3] ≈ [600.0, 700.0, 800.0]
    se3 = HighFidelityEphemerisModel.summarize_ocp_control_entries_mps(entries3)
    @test se3["total_control_vector_norm_mps"] ≈ 15.0
    @test !haskey(se3, "total_control_scalar_mps")

    # Input validation of the OCP collector.
    @test_throws ErrorException HighFidelityEphemerisModel.collect_ocp_control_maneuvers_mps([1.0, 2.0, 3.0], U[1:2, :], 0.0, params)
    @test_throws ErrorException HighFidelityEphemerisModel.collect_ocp_control_maneuvers_mps([1.0, 2.0], U, 0.0, params)

    # Node-to-node (trajectory-jump) maneuver files.
    arcs = _spkC_arcs()
    p = _spkC_params()
    mktempdir() do tmpdir
        ocp_path = joinpath(tmpdir, "ocp3.txt")
        HighFidelityEphemerisModel.write_ocp_control_entries_mps(ocp_path, entries3)
        _, ocp_rows = _spkC_read_csv(ocp_path)
        @test length(ocp_rows) == 3
        @test all(r -> isnan(r[6]), ocp_rows)
        @test ocp_rows[2][2:5] ≈ [0.0, 6.0, 8.0, 10.0]

        path = joinpath(tmpdir, "node_to_node.txt")
        ret = HighFidelityEphemerisModel.write_node_to_node_maneuvers_mps(path, arcs, p.et0, p)
        @test ret == path
        header, rows = _spkC_read_csv(path)
        @test header == "# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps"
        @test length(rows) == 2
        @test rows[1][1] ≈ p.et0 + 8.0 * p.TU
        @test rows[1][2:4] ≈ [20.0, 0.0, -10.0] atol = 1e-9
        @test rows[2][1] ≈ p.et0 + 16.0 * p.TU
        @test rows[2][2:4] ≈ [0.0, 50.0, 0.0] atol = 1e-9

        # An empty entry list writes the header only.
        empty_path = joinpath(tmpdir, "empty.txt")
        HighFidelityEphemerisModel.write_maneuver_entries_mps(empty_path, Any[])
        @test readlines(empty_path) == ["# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps"]
    end
end


# -----------------------------------------------------------------------------
# State sampling: error paths and the backward-compatible wrapper
# -----------------------------------------------------------------------------

function test_spkC_state_sampling_errors_and_wrapper()
    p = _spkC_params()
    arcs = _spkC_arcs()

    # Arcs of 800 s each; a 900 s right-end gap leaves segment 1 with a negative span.
    @test_throws ErrorException HighFidelityEphemerisModel.sample_segmented_states_for_spk(
        arcs[1:2], [(1, 1), (2, 2)], p.et0, p;
        dt_sec = 100.0, segment_gap_sec = 900.0, verbose = false,
    )
    @test occursin("bad span", _spkC_error_message(() ->
        HighFidelityEphemerisModel.sample_segmented_states_for_spk(
            arcs[1:2], [(1, 1), (2, 2)], p.et0, p;
            dt_sec = 100.0, segment_gap_sec = 900.0, verbose = false,
        )))

    # A merged window over a time gap, with the continuity check disabled, has
    # sample epochs that no arc covers (t = 9 lies in the gap (8, 10)).
    gapped = [
        _SpkCArc([0.0, 8.0], [1.0, 2.0, 3.0], _SPKC_V1),
        _SpkCArc([10.0, 16.0], [1.0, 2.0, 3.0] .+ 10.0 .* _SPKC_V1, _SPKC_V1),
    ]
    msg, out = _spkC_capture(() -> _spkC_error_message(() ->
        HighFidelityEphemerisModel.sample_segmented_states_for_spk(
            gapped, [(1, 2)], p.et0, p;
            dt_sec = 100.0, verbose = false, validate_continuity = false,
        )))
    @test occursin("Segment coverage failure", msg)
    @test occursin("epoch not covered by sols", out)

    mktempdir() do tmpdir
        outdir = joinpath(tmpdir, "bad_span")
        @test_throws ErrorException HighFidelityEphemerisModel.write_segmented_states_for_spk!(
            arcs[1:2], [(1, 1), (2, 2)], p.et0, p;
            dt_sec = 100.0, segment_gap_sec = 900.0, outdir = outdir, verbose = false,
        )
        @test isempty(readdir(outdir))

        outdir2 = joinpath(tmpdir, "gap")
        msg2, out2 = _spkC_capture(() -> _spkC_error_message(() ->
            HighFidelityEphemerisModel.write_segmented_states_for_spk!(
                gapped, [(1, 2)], p.et0, p;
                dt_sec = 100.0, outdir = outdir2, verbose = false, validate_continuity = false,
            )))
        @test occursin("Segment coverage failure", msg2)
        @test occursin("epoch not covered by sols", out2)
        @test isempty(readdir(outdir2))

        # Legacy wrapper: `eps_sec` is the right-end trim, the return value is the directory.
        outdir3 = joinpath(tmpdir, "legacy")
        ret, _ = _spkC_capture(() ->
            HighFidelityEphemerisModel.write_segmented_states_from_coast_windows_epsilon_trim!(
                arcs[1:2], [(1, 1), (2, 2)], p.et0, p;
                dt_sec = 100.0, eps_sec = 1e-3, outdir = outdir3,
            ))
        @test ret == outdir3
        @test sort(readdir(outdir3)) == ["seg_001_states.txt", "seg_002_states.txt"]

        for (seg, arc) in zip(1:2, arcs[1:2])
            header, rows = _spkC_read_csv(joinpath(outdir3, "seg_00$(seg)_states.txt"))
            @test header == "# ETSECONDS"
            ets = [r[1] for r in rows]
            @test ets[1] ≈ p.et0 + arc.t[1] * p.TU
            # only the first segment is trimmed by eps_sec
            @test ets[end] ≈ p.et0 + arc.t[end] * p.TU - (seg == 1 ? 1e-3 : 0.0) atol = 1e-9
            @test issorted(ets)
            for r in rows
                x = arc((r[1] - p.et0) / p.TU)
                @test r[2:4] ≈ x[1:3] .* p.DU rtol = 1e-12
                @test r[5:7] ≈ x[4:6] .* p.VU rtol = 1e-12
            end
        end
    end
end


# -----------------------------------------------------------------------------
# Batch ode_sol_to_spk: maneuver products, metadata, verbose output
# -----------------------------------------------------------------------------

function test_spkC_batch_verbose_trajectory_maneuvers()
    p = _spkC_params()
    arcs = _spkC_arcs()
    spice_id = -7770001

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "batch_a.bsp")
        maneuver_txt = joinpath(tmpdir, "sub", "jumps.txt")
        work_parent = joinpath(tmpdir, "work")

        result, out = _spkC_capture(() -> HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p.et0, p;
            output_spk = output_spk,
            spice_id = spice_id,
            center_id = 399,
            dt_sec = 100.0,
            keep_intermediates = true,
            intermediate_parent_dir = work_parent,
            write_maneuvers = true,
            maneuver_txt = maneuver_txt,
            merge_maneuvers_into_metadata = true,
            # OCP summary only: no times, no OCP file
            ocp_control = _SPKC_OCP,
            write_ocp_maneuvers = false,
            merge_ocp_maneuvers_into_metadata = false,
            dynamics_name = "Nbody",
            srp_Cr = 1.8,
            spherical_harmonics_body = "MOON",
            nmax = 10,
            spherical_harmonics_frame = "MOON_PA",
            extra_metadata = Dict("note" => "batch-a"),
            verbose = true,
            print_summary = false,
            show_progress = false,
        ))

        # Returned paths
        @test result.output_spk == abspath(output_spk)
        @test result.trajectory_maneuver_txt == abspath(maneuver_txt)
        @test result.maneuver_txt == abspath(maneuver_txt)
        @test result.ocp_maneuver_txt === nothing
        @test result.segment_count == 3
        @test result.coast_windows == [(1, 1), (2, 2), (3, 3)]
        @test result.point_counts == [9, 9, 5]   # 800 s / 100 s + 1, 400 s / 100 s + 1
        # every segment but the last is trimmed by the default 1e-7 s gap; the last ends exactly
        expected_ranges = [(p.et0, p.et0 + 800.0 - 1e-7), (p.et0 + 800.0, p.et0 + 1600.0 - 1e-7),
                           (p.et0 + 1600.0, p.et0 + 2000.0)]
        @test length(result.epoch_ranges) == 3
        for (got, want) in zip(result.epoch_ranges, expected_ranges)
            @test got[1] ≈ want[1] atol = 1e-9
            @test got[2] ≈ want[2] atol = 1e-9
        end
        @test result.intermediate_dir isa String
        @test isdir(result.intermediate_dir)
        @test dirname(result.intermediate_dir) == abspath(work_parent)
        @test startswith(basename(result.intermediate_dir), "ode_sol_to_spk_")

        # Verbose (non-summary) messages
        @test occursin("SPK generation pipeline", out)
        @test occursin("coast arcs / segments : 3 / 3", out)
        @test occursin("Wrote trajectory-jump diagnostic maneuver file", out)
        @test occursin("Wrote metadata JSON", out)
        @test occursin("Kept intermediate directory", out)
        @test !occursin("SPK generation complete", out)

        # Trajectory-jump file: velocity discontinuities between arcs, in m/s
        _, rows = _spkC_read_csv(maneuver_txt)
        @test length(rows) == 2
        @test rows[1][1] ≈ p.et0 + 800.0
        @test rows[1][2:4] ≈ [20.0, 0.0, -10.0] atol = 1e-9
        @test rows[2][1] ≈ p.et0 + 1600.0
        @test rows[2][2:4] ≈ [0.0, 50.0, 0.0] atol = 1e-9
        @test result.maneuver_summary["count"] == 2
        @test result.maneuver_summary["total_delta_v_mps"] ≈ sqrt(500.0) + 50.0

        # OCP summary from the matrix only (VU = 10 km/s)
        s = result.ocp_control_summary
        @test s["count"] == 2
        @test s["total_control_vector_norm_mps"] ≈ 15.0
        @test s["mean_control_vector_norm_mps"] ≈ 7.5
        @test s["total_control_scalar_mps"] ≈ 16.5
        @test s["max_control_scalar_mps"] ≈ 11.0

        # Metadata: inferred force model + keyword overrides, merged jump entries
        meta = _spkC_read_json(result.metadata_json)
        @test meta["schema"] == "HighFidelityEphemerisModel.SPKMetadata.v1"
        fm = meta["force_model"]
        @test fm["dynamics"] == "Nbody"
        @test fm["naif_ids"] == ["399", "301"]          # inferred from parameters
        @test fm["SRP"]["enabled"] == true              # inferred
        @test fm["SRP"]["Cr"] == 1.8                    # keyword overrides inferred 1.2
        @test fm["spherical_harmonics"]["body"] == "MOON"
        @test fm["spherical_harmonics"]["degree"] == 10  # from `nmax`
        @test fm["spherical_harmonics"]["body_fixed_frame"] == "MOON_PA"
        @test meta["extra"] == Dict("note" => "batch-a")
        m = meta["maneuvers"]
        @test m["type"] == "ocp_control"
        @test m["file"] === nothing
        @test m["primary_cost_key"] == "total_control_scalar_mps"
        @test m["total_control_scalar_mps"] ≈ 16.5
        @test m["merged_into_metadata"] == false
        @test !haskey(m, "entries")
        tj = m["trajectory_jumps"]
        @test tj["merged_into_metadata"] == true
        @test tj["file"] == abspath(maneuver_txt)
        @test length(tj["entries"]) == 2
        @test [tj["entries"][1]["dvx_mps"], tj["entries"][1]["dvy_mps"], tj["entries"][1]["dvz_mps"]] ≈
              [20.0, 0.0, -10.0] atol = 1e-9
        @test tj["entries"][2]["dv_norm_mps"] ≈ 50.0
        @test meta["products"]["trajectory_jump_maneuver_file"] == abspath(maneuver_txt)
        @test meta["products"]["ocp_maneuver_file"] === nothing
        @test meta["sampling"]["segment_count"] == 3
        @test meta["coverage"]["start_et"] ≈ p.et0
        @test meta["coverage"]["end_et"] ≈ p.et0 + 2000.0

        _spkC_check_spk(result.output_spk, arcs, p, (0.5, 4.0, 7.9, 8.1, 12.0, 15.5, 16.5, 19.9, 20.0); spice_id = spice_id)
    end
end


function test_spkC_batch_summary_and_ocp_file()
    p = _spkC_params()
    arcs = _spkC_arcs()
    spice_id = -7770002

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "batch_b.bsp")
        ocp_txt = joinpath(tmpdir, "ocp", "executed.txt")
        user_force = Dict{String,Any}(
            "custom_model" => "mine",
            "SRP" => Dict{String,Any}("Cr" => 1.0, "enabled" => false),
        )

        # Convenience method: et0 is taken from `parameters.et0`.
        result, out = _spkC_capture(() -> HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p;
            output_spk = output_spk,
            spice_id = spice_id,
            center_id = 399,
            dt_sec = 100.0,
            write_maneuvers = true,
            maneuver_txt = ocp_txt,    # with OCP output, `maneuver_txt` names the OCP file
            ocp_control = _SPKC_OCP,
            ocp_control_times = [8.0, 16.0],
            force_model_metadata = user_force,
            srp_Am = 0.02,
            srp_P0 = 4.56e-6,
            srp_enabled = true,
            verbose = true,
            print_summary = true,
            show_progress = false,
        ))

        traj_txt = joinpath(tmpdir, "batch_b_trajectory_jump_maneuvers.txt")
        @test result.ocp_maneuver_txt == abspath(ocp_txt)
        @test result.maneuver_txt == abspath(ocp_txt)
        @test result.trajectory_maneuver_txt == traj_txt
        @test result.metadata_json == joinpath(tmpdir, "batch_b_metadata.json")
        @test result.intermediate_dir === nothing
        @test isempty(filter(startswith("ode_sol_to_spk_"), readdir(tmpdir)))

        # Summary printout
        @test occursin("SPK generation complete", out)
        @test occursin("segments                    : 3", out)
        @test occursin("total control (OCP scalar)  : 1.650000e+01 m/s", out)
        @test !occursin("Wrote metadata JSON", out)     # suppressed when print_summary=true

        # OCP file: ET, DV vector, |DV|, u4 scalar (m/s)
        header, rows = _spkC_read_csv(ocp_txt)
        @test occursin("DV_scalar_u4_mps", header)
        @test length(rows) == 2
        @test rows[1] ≈ [p.et0 + 800.0, 3.0, 4.0, 0.0, 5.0, 5.5] atol = 1e-9
        @test rows[2] ≈ [p.et0 + 1600.0, 0.0, 6.0, 8.0, 10.0, 11.0] atol = 1e-9

        _, jrows = _spkC_read_csv(traj_txt)
        @test length(jrows) == 2

        meta = _spkC_read_json(result.metadata_json)
        fm = meta["force_model"]
        @test fm["custom_model"] == "mine"
        @test fm["SRP"]["Cr"] == 1.0           # from user dict
        @test fm["SRP"]["enabled"] == true     # keyword overrides user dict
        @test fm["SRP"]["A_over_m"] == 0.02
        @test fm["SRP"]["P0"] == 4.56e-6
        @test !haskey(fm, "naif_ids")          # user dict replaces inference
        # the caller's dictionary is not mutated by the merge
        @test user_force["SRP"] == Dict{String,Any}("Cr" => 1.0, "enabled" => false)
        @test meta["maneuvers"]["file"] == abspath(ocp_txt)
        @test meta["maneuvers"]["count"] == 2
        @test meta["products"]["maneuver_file"] == abspath(ocp_txt)

        _spkC_check_spk(result.output_spk, arcs, p, (1.0, 9.0, 17.0); spice_id = spice_id)
    end
end


function test_spkC_batch_errors()
    p = _spkC_params()
    arcs = _spkC_arcs()

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "err.bsp")
        common = (output_spk = output_spk, spice_id = -7770003, center_id = 399, dt_sec = 100.0,
                  verbose = false, show_progress = false)

        # OCP file requested (default when ocp_control is given) but no control times.
        msg = _spkC_error_message(() -> HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p.et0, p; common..., ocp_control = _SPKC_OCP))
        @test occursin("`ocp_control_times` must be provided", msg)
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p.et0, p; common..., ocp_control = _SPKC_OCP)
        # the failed run removed its work directory
        @test isempty(filter(startswith("ode_sol_to_spk_"), readdir(tmpdir)))

        # OCP products requested without any OCP control.
        @test occursin("`ocp_control` must be provided", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., write_ocp_maneuvers = true)))
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p.et0, p; common..., merge_ocp_maneuvers_into_metadata = true)

        # Convenience method without a recognizable et0 property.
        p_no_et0 = (TU = p.TU, DU = p.DU, VU = p.VU)
        other_spk = joinpath(tmpdir, "no_et0.bsp")
        @test occursin("No `et0`", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p_no_et0; common..., output_spk = other_spk)))
        @test !isfile(other_spk)

        # Batch-only / argument checks
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., output_spk_type = 9)
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., dt_sec = 0.0)
        @test occursin("`segment_gap_sec` must be nonnegative", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., segment_gap_sec = -1.0)))
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., append = true)
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., segment_index = 1)
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(_SpkCArc[], p.et0, p; common...)
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs[1], p.et0, p; common..., coast_windows = [(1, 1)])
    end
end


# -----------------------------------------------------------------------------
# Incremental (single-solution) ode_sol_to_spk
# -----------------------------------------------------------------------------

function test_spkC_incremental_products()
    p = _spkC_params()
    arcs = _spkC_arcs()
    spice_id = -7770004

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "inc.bsp")
        ocp_txt = joinpath(tmpdir, "inc_ocp.txt")
        meta1_path = joinpath(tmpdir, "meta", "first.json")

        # First segment: create the kernel, write all maneuver products, verbose log.
        r1, out1 = _spkC_capture(() -> HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs[1], p.et0, p;
            output_spk = output_spk,
            spice_id = spice_id,
            center_id = 399,
            dt_sec = 100.0,
            segment_index = 3,
            write_maneuvers = true,
            ocp_control = _SPKC_OCP[:, 1:1],
            ocp_control_times = [2.0],
            ocp_maneuver_txt = ocp_txt,
            merge_maneuvers_into_metadata = true,   # also merges OCP entries
            metadata_json = meta1_path,
            dynamics_name = "NbodySH",
            spherical_harmonics_enabled = true,
            nmax = 10,
            spherical_harmonics_nmax = 4,           # takes precedence over nmax
            extra_metadata = (note = "inc",),
            verbose = true,
            print_summary = false,
            show_progress = false,
        ))

        @test r1.appended == false
        @test r1.segment_index == 3
        @test r1.segment_count == 1
        @test r1.coast_windows == [(1, 1)]
        @test r1.ocp_maneuver_txt == ocp_txt
        @test r1.maneuver_txt == ocp_txt
        @test r1.trajectory_maneuver_txt == joinpath(tmpdir, "inc_trajectory_jump_maneuvers.txt")
        @test r1.metadata_json == meta1_path
        @test r1.epoch_ranges[1][1] ≈ p.et0
        @test r1.epoch_ranges[1][2] ≈ p.et0 + 800.0
        @test r1.point_counts == [9]

        @test occursin("SPK incremental generation pipeline", out1)
        @test occursin("create new kernel", out1)
        @test occursin("segment index         : 3", out1)
        @test occursin("Wrote empty trajectory-jump diagnostic maneuver file", out1)
        @test occursin("Wrote maneuver file from OCP controls", out1)
        @test occursin("Wrote metadata JSON", out1)

        # A single arc has no trajectory jump: header-only diagnostic file.
        @test readlines(r1.trajectory_maneuver_txt) == ["# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps"]
        _, orows = _spkC_read_csv(ocp_txt)
        @test length(orows) == 1
        @test orows[1] ≈ [p.et0 + 200.0, 3.0, 4.0, 0.0, 5.0, 5.5] atol = 1e-9

        meta = _spkC_read_json(meta1_path)
        inc = meta["extra"]["incremental_append"]
        @test inc["enabled"] == true
        @test inc["appended"] == false
        @test inc["segment_index"] == 3
        @test meta["extra"]["user_extra"] == Dict("note" => "inc")
        @test meta["coverage"]["start_et"] ≈ p.et0
        @test meta["coverage"]["end_et"] ≈ p.et0 + 800.0
        @test meta["force_model"]["dynamics"] == "NbodySH"
        @test meta["force_model"]["spherical_harmonics"]["enabled"] == true
        @test meta["force_model"]["spherical_harmonics"]["degree"] == 4
        @test meta["force_model"]["naif_ids"] == ["399", "301"]
        m = meta["maneuvers"]
        @test m["merged_into_metadata"] == true
        @test length(m["entries"]) == 1
        @test m["entries"][1]["et"] ≈ p.et0 + 200.0
        @test m["entries"][1]["dv_norm_from_vector_mps"] ≈ 5.0
        @test m["entries"][1]["dv_scalar_u4_mps"] ≈ 5.5
        @test m["trajectory_jumps"]["entries"] == Any[]
        @test m["trajectory_jumps"]["count"] == 0

        # Second segment: auto-append (file exists), OCP summary without times, summary printout.
        user_force = Dict("SRP" => Dict("Cr" => 1.3))
        r2, out2 = _spkC_capture(() -> HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs[2], p.et0, p;
            output_spk = output_spk,
            spice_id = spice_id,
            center_id = 399,
            dt_sec = 100.0,
            segment_index = 4,
            ocp_control = _SPKC_OCP[1:3, :],
            write_ocp_maneuvers = false,
            force_model_metadata = user_force,
            srp_P0 = 4.5e-6,
            verbose = true,
            print_summary = true,
            show_progress = false,
        ))

        @test r2.appended == true
        @test r2.segment_index == 4
        @test r2.ocp_maneuver_txt === nothing
        @test r2.maneuver_txt === nothing
        @test r2.metadata_json == joinpath(tmpdir, "inc_metadata.json")
        @test r2.ocp_control_summary["total_control_vector_norm_mps"] ≈ 15.0
        @test r2.ocp_control_summary["max_control_vector_norm_mps"] ≈ 10.0
        @test !haskey(r2.ocp_control_summary, "total_control_scalar_mps")
        @test occursin("append existing kernel", out2)
        @test occursin("SPK generation complete", out2)
        @test occursin("segments                    : 1", out2)
        @test !occursin("total control (OCP scalar)", out2)

        meta2 = _spkC_read_json(r2.metadata_json)
        @test meta2["extra"]["incremental_append"]["appended"] == true
        @test !haskey(meta2["extra"], "user_extra")
        @test meta2["force_model"]["SRP"] == Dict("Cr" => 1.3, "P0" => 4.5e-6)
        @test meta2["maneuvers"]["primary_cost_key"] == "total_control_vector_norm_mps"
        @test meta2["maneuvers"]["file"] === nothing

        # Both appended segments are in the kernel.
        _spkC_check_spk(output_spk, arcs[1:2], p, (0.5, 4.0, 7.5, 8.5, 12.0, 15.9); spice_id = spice_id)
    end
end


function test_spkC_incremental_errors()
    p = _spkC_params()
    arcs = _spkC_arcs()

    mktempdir() do tmpdir
        common = (spice_id = -7770005, center_id = 399, dt_sec = 100.0, verbose = false, show_progress = false)

        @test occursin("`ocp_control_times` must be provided", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs[1], p.et0, p; common...,
                output_spk = joinpath(tmpdir, "a.bsp"), ocp_control = _SPKC_OCP)))
        @test occursin("`ocp_control` must be provided", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs[1], p.et0, p; common...,
                output_spk = joinpath(tmpdir, "b.bsp"), write_ocp_maneuvers = true)))
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs[1], p.et0, p; common...,
            output_spk = joinpath(tmpdir, "c.bsp"), merge_ocp_maneuvers_into_metadata = true)

        # Argument checks happen before anything is written.
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs[1], p.et0, p; common...,
            output_spk = joinpath(tmpdir, "d.bsp"), segment_index = 0)
        @test !isfile(joinpath(tmpdir, "d.bsp"))
        @test occursin("must end in `.bsp`", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs[1], p.et0, p; common...,
                output_spk = joinpath(tmpdir, "e.txt"))))
        @test !isfile(joinpath(tmpdir, "e.txt"))
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs[1], p.et0, p; common...,
            output_spk = joinpath(tmpdir, "f.bsp"), output_spk_type = 9)
    end
end


@testset "SPK pipeline: maneuver helpers" begin
    test_spkC_maneuver_helpers()
end
@testset "SPK pipeline: state sampling errors" begin
    test_spkC_state_sampling_errors_and_wrapper()
end
function test_spkC_batch_uneven_sampling()
    p = _spkC_params()
    arcs = _spkC_arcs()
    spice_id = -7770006

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "uneven.bsp")
        # dt = 350 s on arcs of 800/800/400 s: a trailing node closer than dt/2 to the arc end
        # is dropped, giving nodes {0, 350, 800}, {0, 350, 800} and {0, 400}
        result = HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p.et0, p;
            output_spk = output_spk, spice_id = spice_id, center_id = 399,
            dt_sec = 350.0, polynom_degree = 3,
            write_maneuvers = false, verbose = false, print_summary = false, show_progress = false,
        )
        @test result.point_counts == [3, 3, 2]
        @test isfile(output_spk)
    end
end


@testset "SPK pipeline: batch products" begin
    test_spkC_batch_verbose_trajectory_maneuvers()
    test_spkC_batch_uneven_sampling()
    test_spkC_batch_summary_and_ocp_file()
    test_spkC_batch_errors()
end
@testset "SPK pipeline: incremental products" begin
    test_spkC_incremental_products()
    test_spkC_incremental_errors()
end
