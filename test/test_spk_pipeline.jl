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

    # 4-row control: vector norms and row-4 scalars.
    U = [
        0.3  0.0  0.0;
        0.4  0.6  0.0;
        0.0  0.8  0.0;
        0.55 1.1  0.0;
    ]
    s4 = HighFidelityEphemerisModel.summarize_ocp_control_entries_mps(
        HighFidelityEphemerisModel.collect_ocp_control_maneuvers_mps([1.0, 2.0, 3.0], U, params.et0, params))
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

    # Empty OCP entry list summary: zero statistics, no scalar keys.
    se = HighFidelityEphemerisModel.summarize_ocp_control_entries_mps(Any[])
    @test se["count"] == 0
    for key in ("total_control_vector_norm_mps", "total_control_vector_norm_cmps",
                "max_control_vector_norm_mps", "max_control_vector_norm_cmps",
                "mean_control_vector_norm_mps", "mean_control_vector_norm_cmps")
        @test se[key] == 0.0
    end
    @test !haskey(se, "total_control_scalar_mps")

    # Empty trajectory-jump summary.
    sj = HighFidelityEphemerisModel.summarize_maneuver_entries_mps(Any[])
    @test sj["count"] == 0
    @test sj["total_delta_v_mps"] == 0.0 && sj["rms_delta_v_cmps"] == 0.0

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
        entries = HighFidelityEphemerisModel.collect_node_to_node_maneuvers_mps(arcs, p.et0, p)
        @test HighFidelityEphemerisModel.write_maneuver_entries_mps(path, entries) == path
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
# State sampling: error paths
# -----------------------------------------------------------------------------

function test_spkC_state_sampling_errors()
    p = _spkC_params()
    arcs = _spkC_arcs()

    # Arcs of 800 s each; a 900 s right-end gap leaves segment 1 with a negative span.
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
end


# -----------------------------------------------------------------------------
# Batch ode_sol_to_spk: maneuver products, metadata, verbose output
# -----------------------------------------------------------------------------

function test_spkC_batch_verbose_trajectory_maneuvers()
    p = _spkC_params()
    arcs = _spkC_arcs()
    spice_id = -7770001

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "sub", "batch_a.bsp")

        result, out = _spkC_capture(() -> HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p.et0, p;
            output_spk = output_spk,
            spice_id = spice_id,
            dt_sec = 100.0,
            write_maneuvers = true,
            merge_maneuvers_into_metadata = true,
            force_model_metadata = Dict("dynamics" => "Nbody", "SRP" => Dict("Cr" => 1.8)),
            extra_metadata = Dict("note" => "batch-a"),
            verbose = true,
            show_progress = false,
        ))

        # Returned paths: sidecars are named after the SPK
        traj_txt = joinpath(tmpdir, "sub", "batch_a_trajectory_jump_maneuvers.txt")
        @test result.output_spk == abspath(output_spk)
        @test result.trajectory_maneuver_txt == traj_txt
        @test result.maneuver_txt === nothing
        @test result.ocp_control_summary === nothing
        @test result.metadata_json == joinpath(tmpdir, "sub", "batch_a_metadata.json")
        @test sort(readdir(dirname(output_spk))) == ["batch_a.bsp", "batch_a_metadata.json",
                                                    "batch_a_trajectory_jump_maneuvers.txt"]
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

        # Verbose output: header and summary, no progress bars
        @test occursin("SPK generation pipeline", out)
        @test occursin("coast arcs / segments : 3 / 3", out)
        @test occursin("SPK generation complete", out)
        @test occursin("segments                    : 3", out)
        @test occursin("trajectory-jump maneuvers   : ", out)
        @test !occursin("maneuvers (OCP controls)", out)
        @test !occursin("sampling SPK states", out)

        # Trajectory-jump file: velocity discontinuities between arcs, in m/s
        _, rows = _spkC_read_csv(traj_txt)
        @test length(rows) == 2
        @test rows[1][1] ≈ p.et0 + 800.0
        @test rows[1][2:4] ≈ [20.0, 0.0, -10.0] atol = 1e-9
        @test rows[2][1] ≈ p.et0 + 1600.0
        @test rows[2][2:4] ≈ [0.0, 50.0, 0.0] atol = 1e-9
        @test result.maneuver_summary["count"] == 2
        @test result.maneuver_summary["total_delta_v_mps"] ≈ sqrt(500.0) + 50.0

        # Metadata: inferred force model + user dictionary, merged jump entries
        meta = _spkC_read_json(result.metadata_json)
        @test meta["schema"] == "HighFidelityEphemerisModel.SPKMetadata.v2"
        @test meta["spk"]["center_id"] == 399                 # naif_ids[1]
        @test meta["spk"]["ref_frame_name"] == "J2000"
        @test meta["spk"]["polynom_degree"] == 7
        fm = meta["force_model"]
        @test fm["dynamics"] == "Nbody"
        @test fm["naif_ids"] == ["399", "301"]          # inferred from parameters
        @test fm["SRP"] == Dict("enabled" => true, "Cr" => 1.8)
        @test meta["extra"] == Dict("note" => "batch-a")
        @test collect(keys(meta["maneuvers"])) == ["trajectory_jumps"]
        tj = meta["maneuvers"]["trajectory_jumps"]
        @test tj["file"] == traj_txt
        @test length(tj["entries"]) == 2
        @test [tj["entries"][1]["dvx_mps"], tj["entries"][1]["dvy_mps"], tj["entries"][1]["dvz_mps"]] ≈
              [20.0, 0.0, -10.0] atol = 1e-9
        @test tj["entries"][2]["dv_norm_mps"] ≈ 50.0
        @test meta["products"]["trajectory_jump_maneuver_file"] == traj_txt
        @test meta["products"]["maneuver_file"] === nothing
        @test meta["sampling"] == Dict("dt_sec" => 100.0, "segment_gap_sec" => 1e-7, "segment_count" => 3)
        @test meta["coverage"]["start_et"] ≈ p.et0
        @test meta["coverage"]["end_et"] ≈ p.et0 + 2000.0
        windows = meta["coverage"]["windows"]
        @test [w["segment_id"] for w in windows] == ["HFEM_SPK_SEGMENT_001", "HFEM_SPK_SEGMENT_002", "HFEM_SPK_SEGMENT_003"]
        @test windows[2]["coast_window"] == [2, 2]
        @test windows[3]["point_count"] == 5

        _spkC_check_spk(result.output_spk, arcs, p, (0.5, 4.0, 7.9, 8.1, 12.0, 15.5, 16.5, 19.9, 20.0); spice_id = spice_id)
    end
end


function test_spkC_batch_summary_and_ocp_file()
    p = _spkC_params()
    arcs = _spkC_arcs()
    spice_id = -7770002

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "batch_b.bsp")
        user_force = Dict{String,Any}("SRP" => Dict{String,Any}("Cr" => 1.0))

        # Convenience method: et0 is taken from `parameters.et0`; center given by name.
        result, out = _spkC_capture(() -> HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs, p;
            output_spk = output_spk,
            spice_id = string(spice_id),
            center_id = "EARTH",
            dt_sec = 100.0,
            ocp_control = _SPKC_OCP,
            ocp_control_times = [8.0, 16.0],
            merge_maneuvers_into_metadata = true,
            force_model_metadata = user_force,
            show_progress = false,
        ))

        ocp_txt = joinpath(tmpdir, "batch_b_maneuvers.txt")
        @test result.maneuver_txt == ocp_txt
        @test result.trajectory_maneuver_txt === nothing
        @test result.metadata_json == joinpath(tmpdir, "batch_b_metadata.json")

        # Summary printout
        @test occursin("SPK generation complete", out)
        @test occursin("maneuvers (OCP controls)    : ", out)
        @test occursin("total control (OCP scalar)  : 1.650000e+01 m/s", out)

        # OCP file: ET, DV vector, |DV|, u4 scalar (m/s)
        header, rows = _spkC_read_csv(ocp_txt)
        @test occursin("DV_scalar_u4_mps", header)
        @test length(rows) == 2
        @test rows[1] ≈ [p.et0 + 800.0, 3.0, 4.0, 0.0, 5.0, 5.5] atol = 1e-9
        @test rows[2] ≈ [p.et0 + 1600.0, 0.0, 6.0, 8.0, 10.0, 11.0] atol = 1e-9

        s = result.ocp_control_summary
        @test s["count"] == 2
        @test s["total_control_vector_norm_mps"] ≈ 15.0
        @test s["mean_control_vector_norm_mps"] ≈ 7.5
        @test s["total_control_scalar_mps"] ≈ 16.5
        @test s["max_control_scalar_mps"] ≈ 11.0

        meta = _spkC_read_json(result.metadata_json)
        fm = meta["force_model"]
        @test fm["SRP"] == Dict("enabled" => true, "Cr" => 1.0)   # merged over the inferred entry
        @test fm["naif_ids"] == ["399", "301"]                    # inferred entries are kept
        # the caller's dictionary is not mutated by the merge
        @test user_force == Dict{String,Any}("SRP" => Dict{String,Any}("Cr" => 1.0))
        m = meta["maneuvers"]
        ocp = m["ocp_control"]
        @test ocp["file"] == ocp_txt
        @test ocp["count"] == 2
        @test ocp["primary_cost_key"] == "total_control_scalar_mps"
        @test length(ocp["entries"]) == 2
        @test ocp["entries"][2]["dv_scalar_u4_mps"] ≈ 11.0
        # trajectory jumps are summarized (and merged) even without the diagnostic file
        @test m["trajectory_jumps"]["file"] === nothing
        @test m["trajectory_jumps"]["count"] == 2
        @test length(m["trajectory_jumps"]["entries"]) == 2
        @test meta["products"]["maneuver_file"] == ocp_txt
        @test meta["spk"]["object_id"] == spice_id

        _spkC_check_spk(result.output_spk, arcs, p, (1.0, 9.0, 17.0); spice_id = spice_id)

        # A single solution is written as a one-segment kernel (overwriting the previous one).
        single = HighFidelityEphemerisModel.ode_sol_to_spk(
            arcs[1], p; output_spk = output_spk, spice_id = spice_id, dt_sec = 100.0,
            write_metadata = false, verbose = false)
        @test single.segment_count == 1
        @test single.maneuver_summary["count"] == 0
        @test length(HighFidelityEphemerisModel.read_spk_segments(output_spk)) == 1
    end
end


function test_spkC_batch_errors()
    p = _spkC_params()
    arcs = _spkC_arcs()

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "err.bsp")
        common = (output_spk = output_spk, spice_id = -7770003, dt_sec = 100.0, verbose = false)

        # OCP inputs are checked before anything is written.
        @test occursin("`ocp_control_times` must be provided", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., ocp_control = _SPKC_OCP)))
        @test occursin("without `ocp_control`", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., ocp_control_times = [1.0])))
        @test occursin("must match the number", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common...,
                ocp_control = _SPKC_OCP, ocp_control_times = [1.0])))
        @test occursin("at least 3 rows", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common...,
                ocp_control = _SPKC_OCP[1:2, :], ocp_control_times = [1.0, 2.0])))

        # Convenience method without an et0 property.
        p_no_et0 = (TU = p.TU, DU = p.DU, VU = p.VU, naif_ids = ["399"])
        @test occursin("no `et0` property", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p_no_et0; common...)))

        # No center given and none in `parameters`; frame without a SPICE equivalent.
        @test occursin("Pass `center_id`", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, (TU = p.TU, DU = p.DU, VU = p.VU); common...)))
        @test occursin("EME2000", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, (p..., naif_frame = "EME2000"); common...)))

        # Argument checks
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., dt_sec = 0.0)
        @test occursin("`segment_gap_sec` must be nonnegative", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., segment_gap_sec = -1.0)))
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(_SpkCArc[], p.et0, p; common...)
        @test occursin("must end in `.bsp`", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., output_spk = joinpath(tmpdir, "e.txt"))))
        @test isempty(readdir(tmpdir))

        # overwrite = false keeps an existing kernel and writes no sidecar
        write(output_spk, "existing")
        @test occursin("overwrite=false", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., overwrite = false)))
        @test readdir(tmpdir) == ["err.bsp"] && read(output_spk, String) == "existing"

        # a non-dictionary force_model_metadata is rejected before the kernel is replaced
        @test occursin("must be a dictionary or NamedTuple", _spkC_error_message(() ->
            HighFidelityEphemerisModel.ode_sol_to_spk(arcs, p.et0, p; common..., force_model_metadata = "eom_Nbody!")))
        @test readdir(tmpdir) == ["err.bsp"] && read(output_spk, String) == "existing"
    end
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
            output_spk = output_spk, spice_id = spice_id,
            dt_sec = 350.0, polynom_degree = 3, verbose = false,
        )
        @test result.point_counts == [3, 3, 2]
        @test [s.point_count for s in HighFidelityEphemerisModel.read_spk_segments(output_spk)] == [3, 3, 2]
        @test isfile(output_spk)
    end
end


# -----------------------------------------------------------------------------
# Incremental append + finalize
# -----------------------------------------------------------------------------

function test_spkC_incremental_products()
    p = _spkC_params()
    arcs = _spkC_arcs()
    spice_id = -7770004

    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "inc.bsp")

        for (k, arc) in enumerate(arcs)
            r = HighFidelityEphemerisModel.append_ode_sol_to_spk!(
                arc, p; output_spk = output_spk, spice_id = spice_id, append = k > 1,
                dt_sec = 100.0, polynom_degree = 5)
            @test r.segment_index == k
        end
        @test readdir(tmpdir) == ["inc.bsp"]
        # Arcs share boundary epochs: the segment appended last (post-maneuver arc) wins there.
        _spkC_check_spk(output_spk, arcs, p, (0.5, 4.0, 7.9, 8.1, 12.0, 15.9, 16.1, 20.0); spice_id = spice_id)
        SPICE.furnsh(output_spk)
        try
            st, _ = SPICE.spkezr(string(spice_id), p.et0 + 8.0 * p.TU, "J2000", "NONE", "399")
            @test st[4:6] ≈ arcs[2](8.0)[4:6] .* p.VU atol = 1e-10
        finally
            SPICE.unload(output_spk)
        end

        # finalize: OCP file + metadata describing every appended arc, verbose summary
        result, out = _spkC_capture(() -> HighFidelityEphemerisModel.finalize_spk(
            output_spk, p;
            ocp_control = _SPKC_OCP,
            ocp_control_times = [8.0, 16.0],
            merge_maneuvers_into_metadata = true,
            force_model_metadata = Dict("dynamics" => "NbodySH"),
            extra_metadata = (note = "inc",),
        ))
        @test result.segment_count == 3
        @test result.point_counts == [9, 9, 5]
        @test result.epoch_ranges == [(p.et0, p.et0 + 800.0), (p.et0 + 800.0, p.et0 + 1600.0), (p.et0 + 1600.0, p.et0 + 2000.0)]
        @test result.maneuver_txt == joinpath(tmpdir, "inc_maneuvers.txt")
        @test result.metadata_json == joinpath(tmpdir, "inc_metadata.json")
        @test result.ocp_control_summary["total_control_scalar_mps"] ≈ 16.5
        @test occursin("segments                    : 3", out)
        @test occursin("total control (OCP scalar)  : 1.650000e+01 m/s", out)

        _, orows = _spkC_read_csv(result.maneuver_txt)
        @test orows[1] ≈ [p.et0 + 800.0, 3.0, 4.0, 0.0, 5.0, 5.5] atol = 1e-9

        meta = _spkC_read_json(result.metadata_json)
        @test meta["spk"]["object_id"] == spice_id
        @test meta["spk"]["center_id"] == 399
        @test meta["spk"]["ref_frame_name"] == "J2000"
        @test meta["spk"]["polynom_degree"] == 5
        @test meta["sampling"] == Dict("dt_sec" => nothing, "segment_gap_sec" => nothing, "segment_count" => 3)
        @test meta["coverage"]["start_et"] ≈ p.et0
        @test meta["coverage"]["end_et"] ≈ p.et0 + 2000.0
        @test [w["segment_id"] for w in meta["coverage"]["windows"]] ==
              ["HFEM_SPK_SEGMENT_001", "HFEM_SPK_SEGMENT_002", "HFEM_SPK_SEGMENT_003"]
        @test meta["force_model"]["dynamics"] == "NbodySH"
        @test meta["force_model"]["naif_ids"] == ["399", "301"]
        @test meta["extra"] == Dict("note" => "inc")
        @test collect(keys(meta["maneuvers"])) == ["ocp_control"]
        @test length(meta["maneuvers"]["ocp_control"]["entries"]) == 2
        @test meta["products"]["trajectory_jump_maneuver_file"] === nothing

        # metadata only, no OCP file
        r2 = HighFidelityEphemerisModel.finalize_spk(output_spk, p.et0, p; verbose = false)
        @test r2.maneuver_txt === nothing && r2.ocp_control_summary === nothing
        @test _spkC_read_json(r2.metadata_json)["maneuvers"] == Dict{String,Any}()
    end
end


function test_spkC_incremental_errors()
    p = _spkC_params()
    arcs = _spkC_arcs()

    mktempdir() do tmpdir
        # finalize: missing file, bad OCP input, mixed objects
        @test_throws ErrorException HighFidelityEphemerisModel.finalize_spk(joinpath(tmpdir, "none.bsp"), p; verbose = false)

        mixed = joinpath(tmpdir, "mixed.bsp")
        HighFidelityEphemerisModel.append_ode_sol_to_spk!(arcs[1], p; output_spk = mixed, spice_id = -7770005, append = false, dt_sec = 100.0)
        @test occursin("`ocp_control_times` must be provided", _spkC_error_message(() ->
            HighFidelityEphemerisModel.finalize_spk(mixed, p; ocp_control = _SPKC_OCP, verbose = false)))
        HighFidelityEphemerisModel.append_ode_sol_to_spk!(arcs[2], p; output_spk = mixed, spice_id = -7770007, append = true, dt_sec = 100.0)
        @test occursin("share one `spice_id`", _spkC_error_message(() ->
            HighFidelityEphemerisModel.finalize_spk(mixed, p; verbose = false)))
        @test readdir(tmpdir) == ["mixed.bsp"]
    end
end


@testset "SPK pipeline: maneuver helpers" begin
    test_spkC_maneuver_helpers()
end
@testset "SPK pipeline: state sampling errors" begin
    test_spkC_state_sampling_errors()
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
