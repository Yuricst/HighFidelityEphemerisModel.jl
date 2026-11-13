"""Tests for SPK helper utilities, the native type-13 writers, incremental appends and metadata"""

using LinearAlgebra
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end

const _spkB_HFEM = HighFidelityEphemerisModel

# SPICE IDs used only by this file (do not reuse elsewhere in the test suite)
const _spkB_ID_A = -771001
const _spkB_ID_B = -771002
const _spkB_ID_C = -771003
const _spkB_ID_D = -771004
const _spkB_ID_E = -771005


"""Linear-motion coast arc with the minimal `ODESolution`-like interface (canonical units)."""
struct _SpkBArc
    t::Vector{Float64}
    x0::Vector{Float64}
    xdot::Vector{Float64}
end

(arc::_SpkBArc)(t) = arc.x0 .+ arc.xdot .* t


"""Exact linear-motion state in km, km/s (type-13 Hermite interpolation reproduces it exactly)."""
_spkB_linear_state(et; r0 = [7000.0, -1200.0, 300.0], v = [0.5, 7.2, -0.3], t0 = 0.0) =
    vcat(r0 .+ v .* (et - t0), v)

function _spkB_linear_segment(t0, t1, n; kwargs...)
    epochs = collect(range(t0, t1; length = n))
    return (epochs = epochs, states = [_spkB_linear_state(et; kwargs...) for et in epochs])
end

"""Query a written SPK (furnished only for the duration of the call)."""
function _spkB_query(spk, id, center, ets; frame = "J2000")
    SPICE.furnsh(spk)
    try
        return [SPICE.spkezr(string(id), et, frame, "NONE", string(center))[1] for et in ets]
    finally
        SPICE.unload(spk)
    end
end

_spkB_capture(f) = mktemp() do path, io
    redirect_stdout(f, io)
    flush(io)
    close(io)
    read(path, String)
end


@testset "SPK utils: parameter, ID and frame helpers" begin
    @test _spkB_HFEM._get_et0((et0 = 12, TU = 1.0)) === 12.0
    e = try _spkB_HFEM._get_et0((ET0 = 1.0,)); nothing; catch err; err; end
    @test e isa ErrorException && occursin("no `et0` property", e.msg)

    @test _spkB_HFEM._prop((a = :abc,), :a) === :abc
    @test _spkB_HFEM._prop((a = 1,), :b) === nothing

    rid = _spkB_HFEM._resolve_body_code
    @test rid(Int32(-5); label = "spice_id") === -5
    @test rid("EARTH"; label = "center_id") === 399
    @test rid(SubString("MOON", 1); label = "center_id") === 301
    @test rid("-123456"; label = "spice_id") === -123456
    e1 = try rid("NOT_A_REAL_BODY_XYZ"; label = "center_id"); nothing; catch e; e; end
    @test e1 isa ErrorException && occursin("Could not resolve `center_id`", e1.msg)
    e2 = try rid(1.5; label = "spice_id"); nothing; catch e; e; end
    @test e2 isa ErrorException && occursin("integer SPICE ID or a SPICE body name", e2.msg)

    # center: central body of the dynamics
    @test _spkB_HFEM._default_center((naif_ids = ["301", "399"],)) == "301"
    @test _spkB_HFEM._default_center((naif_ids = [399],)) == "399"
    @test_throws ErrorException _spkB_HFEM._default_center((naif_ids = String[],))
    e3 = try _spkB_HFEM._default_center((TU = 1.0,)); nothing; catch e; e; end
    @test e3 isa ErrorException && occursin("Pass `center_id`", e3.msg)

    # frame: integration frame, ICRF written as J2000, EME2000 rejected
    @test _spkB_HFEM._default_ref_frame((TU = 1.0,)) == "J2000"
    @test _spkB_HFEM._default_ref_frame((naif_frame = "ECLIPJ2000",)) == "ECLIPJ2000"
    @test _spkB_HFEM._default_ref_frame((naif_frame = "icrf",)) == "J2000"
    e4 = try _spkB_HFEM._default_ref_frame((naif_frame = "EME2000",)); nothing; catch e; e; end
    @test e4 isa ErrorException && occursin("not a SPICE frame", e4.msg)

    # all defaults resolved together; explicit values take precedence
    ids = _spkB_HFEM._resolve_spk_ids((naif_ids = ["301"], naif_frame = "ICRF"), -7, nothing, nothing)
    @test ids == (spice_id = -7, center_id = 301, ref_frame_name = "J2000")
    ids2 = _spkB_HFEM._resolve_spk_ids((TU = 1.0,), "-7", "EARTH", "ECLIPJ2000")
    @test ids2 == (spice_id = -7, center_id = 399, ref_frame_name = "ECLIPJ2000")

    @test _spkB_HFEM._body_name(399) == "EARTH"
    @test _spkB_HFEM._body_name(-987654) === nothing
end


@testset "SPK utils: path/summary helpers" begin
    # _display_path
    @test _spkB_HFEM._display_path(nothing) == ""
    @test _spkB_HFEM._display_path(joinpath(pwd(), "a", "b.bsp")) == joinpath("a", "b.bsp")
    # relpath throws on an empty path; the raw string is returned instead
    @test _spkB_HFEM._display_path("") == ""

    # _summary_get
    @test _spkB_HFEM._summary_get(nothing, "k", :dflt) == :dflt
    @test _spkB_HFEM._summary_get(Dict("k" => 3), "k") == 3
    @test _spkB_HFEM._summary_get(Dict("k" => 3), "other", -1) == -1
    # objects that do not support `get` with this key fall back to the default
    @test _spkB_HFEM._summary_get([1, 2, 3], "k", :dflt) == :dflt
    @test _spkB_HFEM._summary_get(5, "k", :dflt) == :dflt
end


@testset "SPK utils: printing helpers" begin
    out = _spkB_capture() do
        _spkB_HFEM._print_spk_pipeline_summary(
            output_spk = joinpath(pwd(), "run", "traj.bsp"),
            segment_count = 7,
            maneuver_txt = joinpath(pwd(), "run", "ocp.txt"),
            trajectory_maneuver_txt = joinpath(pwd(), "run", "jumps.txt"),
            metadata_json = joinpath(pwd(), "run", "meta.json"),
            ocp_control_summary = Dict("total_control_scalar_mps" => 1.25),
        )
    end
    @test occursin("SPK generation complete", out)
    @test occursin(r"segments\s+: 7", out)
    @test occursin("SPK                         : " * joinpath("run", "traj.bsp"), out)
    @test occursin("maneuvers (OCP controls)    : " * joinpath("run", "ocp.txt"), out)
    @test occursin("trajectory-jump maneuvers   : " * joinpath("run", "jumps.txt"), out)
    @test occursin("metadata JSON               : " * joinpath("run", "meta.json"), out)
    @test occursin("total control (OCP scalar)  : 1.250000e+00 m/s", out)

    # optional products omitted
    out2 = _spkB_capture() do
        _spkB_HFEM._print_spk_pipeline_summary(output_spk = joinpath(pwd(), "x.bsp"), segment_count = 1)
    end
    @test occursin("SPK                         : x.bsp", out2)
    @test !occursin("maneuvers", out2)
    @test !occursin("metadata JSON", out2)
    @test !occursin("total control", out2)

    # _print_progress
    @test _spkB_capture(() -> _spkB_HFEM._print_progress("lbl", 1, 2; enabled = false)) == ""
    @test _spkB_capture(() -> _spkB_HFEM._print_progress("lbl", 1, 0)) == ""
    half = _spkB_capture(() -> _spkB_HFEM._print_progress("lbl", 1, 2; width = 10))
    @test half == "\r" * rpad("lbl", 22) * " [=====     ] 1/2"
    full = _spkB_capture(() -> _spkB_HFEM._print_progress("lbl", 12, 12; width = 4))
    @test full == "\r" * rpad("lbl", 22) * " [====] 12/12\n"
    padded = _spkB_capture(() -> _spkB_HFEM._print_progress("lbl", 3, 12; width = 4))
    @test endswith(padded, "[=   ]  3/12")
end


@testset "SPK utils: JSON conversion and writer" begin
    js = _spkB_HFEM._json_safe
    @test js(nothing) === nothing
    @test js(missing) === nothing
    @test js(SubString("abc", 1, 2)) === "ab"
    @test js(:sym) == "sym"
    @test js(true) === true
    @test js(Int32(4)) === Int32(4)
    @test js(1.5f0) === 1.5
    @test js(NaN) == "NaN"
    @test js(-Inf) == "-Inf"
    @test js((a = 1, b = :x)) == Dict("a" => 1, "b" => "x")
    @test js(Dict(:k => (1, 2.0))) == Dict("k" => Any[1, 2.0])
    @test js((1, "two", :three)) == Any[1, "two", "three"]
    @test js([nothing, Inf]) == Any[nothing, "Inf"]
    @test js([1 2 3; 4 5 6]) == [[1, 2, 3], [4, 5, 6]]
    @test js(1 // 3) == "1//3"
    @test js(1.0 + 2.0im) == string(1.0 + 2.0im)

    render(x) = sprint(io -> _spkB_HFEM._write_json_value(io, x, 0))
    @test render(nothing) == "null"
    @test render(false) == "false"
    @test render(-3) == "-3"
    @test render(0.25) == "0.25"
    # non-finite floats are not valid JSON numbers: written as strings
    @test render(Inf) == "\"Inf\""
    @test render(NaN) == "\"NaN\""
    @test render(Any[]) == "[]"
    @test render(:sym) == "\"sym\""
    @test render("q\"b\\s\nr\rt\t") == "\"q\\\"b\\\\s\\nr\\rt\\t\""
    @test render([1, 2]) == "[\n  1,\n  2\n]"
    # dictionaries are written with sorted keys and nested indentation
    @test render(Dict("b" => [true], "a" => Dict("c" => nothing))) ==
        "{\n  \"a\": {\n    \"c\": null\n  },\n  \"b\": [\n    true\n  ]\n}"

    mktempdir() do tmpdir
        path = joinpath(tmpdir, "meta.json")
        @test _spkB_HFEM.write_spk_metadata_json(path, (name = "x", vals = (1, NaN), m = [1.0 2.0])) == path
        @test read(path, String) ==
            "{\n  \"m\": [\n    [\n      1.0,\n      2.0\n    ]\n  ],\n  \"name\": \"x\",\n  \"vals\": [\n    1,\n    \"NaN\"\n  ]\n}\n"
    end
end


@testset "SPK type 13: id/segment-id/path helpers" begin
    sid = _spkB_HFEM._spkw13_segment_id
    @test sid("ARC", 7) == "ARC_007"
    @test sid("ARC", 1234) == "ARC_1234"
    @test sid(repeat("X", 36), 1) == repeat("X", 36) * "_001"
    @test_throws ErrorException sid(repeat("X", 37), 1)        # 41 characters with the suffix
    @test_throws ErrorException sid("bad\tid", 1)
    @test_throws ErrorException sid("   ", 1)

    mktempdir() do tmpdir
        p = joinpath(tmpdir, "sub", "x.bsp")
        @test _spkB_HFEM._prepare_spk_final_path(p) == abspath(p)
        @test isdir(dirname(p))
        @test_throws ErrorException _spkB_HFEM._prepare_spk_final_path(joinpath(tmpdir, "x.txt"))
        write(p, "old")
        @test _spkB_HFEM._prepare_spk_final_path(p; overwrite = true) == abspath(p)
        e3 = try _spkB_HFEM._prepare_spk_final_path(p; overwrite = false); nothing; catch e; e; end
        @test e3 isa ErrorException && occursin("overwrite=false", e3.msg)
        @test read(p, String) == "old"   # never touched by the path check
    end
end


@testset "SPK type 13: segment validation" begin
    ok = _spkB_linear_segment(0.0, 100.0, 5)
    v(epochs, states; degree = 3, first = epochs[1], last = epochs[end], segid = "S") =
        _spkB_HFEM._validate_spkw13_segment(epochs, states; degree, first, last, segid)

    @test v(ok.epochs, ok.states) === nothing
    @test v(ok.epochs, ok.states; first = 10.0, last = 90.0) === nothing     # interior descriptor
    @test_throws ErrorException v(ok.epochs, ok.states[1:4])                 # length mismatch
    @test_throws ErrorException v(Float64[], Vector{Float64}[]; first = 0.0, last = 1.0)
    @test_throws ErrorException v(ok.epochs, ok.states; degree = 0)
    @test_throws ErrorException v(ok.epochs, ok.states; degree = 4)          # even
    @test_throws ErrorException v(ok.epochs, ok.states; degree = 11)         # needs 6 states
    @test_throws ErrorException v(ok.epochs, ok.states; first = 50.0, last = 50.0)
    @test_throws ErrorException v([0.0, 25.0, 25.0, 75.0, 100.0], ok.states)  # not increasing
    @test_throws ErrorException v(ok.epochs, ok.states; first = -1.0)         # before data
    @test_throws ErrorException v(ok.epochs, ok.states; last = 101.0)         # after data
    @test_throws ErrorException v(ok.epochs, ok.states; segid = repeat("Y", 41))
    @test_throws ErrorException v(ok.epochs, ok.states; segid = "  ")
    @test_throws ErrorException v(ok.epochs, [ok.states[1:4]; [zeros(5)]])
    @test_throws ErrorException v(ok.epochs, [ok.states[1:4]; [[NaN, 0, 0, 0, 0, 0]]])
end


@testset "SPK type 13: safe install" begin
    mktempdir() do tmpdir
        out = joinpath(tmpdir, "out.bsp")

        @test_throws ErrorException _spkB_HFEM._safe_install_spk!(joinpath(tmpdir, "nope.bsp"), out)

        # fresh install: plain move
        tmp1 = joinpath(tmpdir, "t1.bsp"); write(tmp1, "first")
        @test _spkB_HFEM._safe_install_spk!(tmp1, out) == out
        @test read(out, String) == "first" && !isfile(tmp1)

        # existing output, overwrite=false: refused, both files untouched
        tmp2 = joinpath(tmpdir, "t2.bsp"); write(tmp2, "second")
        e = try _spkB_HFEM._safe_install_spk!(tmp2, out; overwrite = false); nothing; catch err; err; end
        @test e isa ErrorException && occursin("overwrite=false", e.msg)
        @test read(out, String) == "first" && read(tmp2, String) == "second"

        # existing output, overwrite=true: replaced via a backup that is then removed
        @test _spkB_HFEM._safe_install_spk!(tmp2, out) == out
        @test read(out, String) == "second" && !isfile(tmp2)
        @test sort(readdir(tmpdir)) == ["out.bsp"]

        # failure after the old file was moved aside: the previous file is restored.
        # (tmp == output, so the second move fails because its source was moved away.)
        e2 = try _spkB_HFEM._safe_install_spk!(out, out); nothing; catch err; err; end
        @test e2 isa ErrorException
        @test occursin("Failed to replace output SPK safely", e2.msg)
        @test read(out, String) == "second"
        @test sort(readdir(tmpdir)) == ["out.bsp"]
    end
end


@testset "SPK type 13: write_spkw13_spk!" begin
    mktempdir() do tmpdir
        out = joinpath(tmpdir, "w13.bsp")
        segs = [_spkB_linear_segment(0.0, 400.0, 9), _spkB_linear_segment(400.0, 800.0, 9)]

        @test_throws ErrorException _spkB_HFEM.write_spkw13_spk!([], out; spice_id = _spkB_ID_A, center_id = 399)

        printed = _spkB_capture() do
            _spkB_HFEM.write_spkw13_spk!(
                segs, out; spice_id = _spkB_ID_A, center_id = 399,
                segment_id = "SPKB", polynom_degree = 5, verbose = true, show_progress = true,
            )
        end
        @test isfile(out)
        @test occursin("writing SPK segments", printed)
        @test sort(readdir(tmpdir)) == ["w13.bsp"]   # temporary BSP was installed, not left behind
        ets = [0.0, 123.0, 400.0, 555.5, 800.0]
        for (et, x) in zip(ets, _spkB_query(out, _spkB_ID_A, 399, ets))
            @test x ≈ _spkB_linear_state(et) rtol = 1e-12
        end

        # segment descriptors read back from the file
        segments = _spkB_HFEM.read_spk_segments(out)
        @test length(segments) == 2
        @test segments[2] == (spice_id = _spkB_ID_A, center_id = 399, ref_frame_name = "J2000", spk_type = 13,
            first = 400.0, last = 800.0, segment_id = "SPKB_002", polynom_degree = 5, point_count = 9)
        @test_throws ErrorException _spkB_HFEM.read_spk_segments(joinpath(tmpdir, "none.bsp"))

        # overwrite an existing kernel with different content (safe-install backup path)
        v2 = [-1.0, 2.0, 0.25]
        segs2 = [_spkB_linear_segment(0.0, 800.0, 11; v = v2)]
        _spkB_HFEM.write_spkw13_spk!(segs2, out; spice_id = _spkB_ID_A, center_id = 399, verbose = false)
        @test sort(readdir(tmpdir)) == ["w13.bsp"]
        for (et, x) in zip(ets, _spkB_query(out, _spkB_ID_A, 399, ets))
            @test x ≈ _spkB_linear_state(et; v = v2) rtol = 1e-12
        end

        # overwrite=false is refused before anything is written
        bytes = read(out)
        @test_throws ErrorException _spkB_HFEM.write_spkw13_spk!(segs, out; spice_id = _spkB_ID_A,
            center_id = 399, overwrite = false, verbose = false)
        @test read(out) == bytes

        # invalid even degree is rejected during validation, before any temporary
        # BSP is opened; the previous output is preserved, the error propagates
        e = try
            _spkB_HFEM.write_spkw13_spk!(segs, out; spice_id = _spkB_ID_A, center_id = 399,
                polynom_degree = 6, verbose = false)
            nothing
        catch err
            err
        end
        @test e isa ErrorException && occursin("must be odd", e.msg)
        @test sort(readdir(tmpdir)) == ["w13.bsp"]
        @test read(out) == bytes

        # failure inside SPICE after `spkopn` with zero segments written (unknown
        # frame): `spkcls` refuses to close an empty SPK, so the handle must be
        # released via `dafcls` and the staged file removed (EBUSY on Windows otherwise)
        e = try
            _spkB_HFEM.write_spkw13_spk!(segs, out; spice_id = _spkB_ID_A, center_id = 399,
                ref_frame_name = "SPKB_NOT_A_FRAME", verbose = false)
            nothing
        catch err
            err
        end
        @test e isa SPICE.SpiceError
        @test sort(readdir(tmpdir)) == ["w13.bsp"]
        @test read(out) == bytes
    end
end


@testset "SPK type 13: custom frame kernel" begin
    mktempdir() do tmpdir
        # Identity TK frame relative to J2000 that SPICE only knows after the kernel is loaded
        fk = joinpath(tmpdir, "spkb_test_frame.tf")
        write(fk, """
        KPL/FK

        \\begindata
        FRAME_SPKB_TEST_FRAME      = 1771001
        FRAME_1771001_NAME         = 'SPKB_TEST_FRAME'
        FRAME_1771001_CLASS        = 4
        FRAME_1771001_CLASS_ID     = 1771001
        FRAME_1771001_CENTER       = 399
        TKFRAME_1771001_RELATIVE   = 'J2000'
        TKFRAME_1771001_SPEC       = 'MATRIX'
        TKFRAME_1771001_MATRIX     = ( 1 0 0 0 1 0 0 0 1 )
        \\begintext
        """)
        seg = _spkB_linear_segment(0.0, 300.0, 7)
        out = joinpath(tmpdir, "frame.bsp")
        out_app = joinpath(tmpdir, "frame_append.bsp")
        try
            _spkB_HFEM.write_spkw13_spk!([seg], out; spice_id = _spkB_ID_C, center_id = 399,
                ref_frame_name = "SPKB_TEST_FRAME", frame_def_file = fk, verbose = false)
            # the appended segment uses the custom frame; the first one does not
            _spkB_HFEM.write_spkw13_spk!([_spkB_linear_segment(-300.0, 0.0, 7)], out_app;
                spice_id = _spkB_ID_D, center_id = 399, verbose = false)
            _spkB_HFEM.append_spkw13_segment_to_spk!(seg, out_app; segment_index = 2,
                spice_id = _spkB_ID_D, center_id = 399, ref_frame_name = "SPKB_TEST_FRAME",
                frame_def_file = fk)
            # the frame is now known to SPICE
            @test SPICE.namfrm("SPKB_TEST_FRAME") == 1771001
            @test _spkB_HFEM.read_spk_segments(out_app)[2].ref_frame_name == "SPKB_TEST_FRAME"
            for (spk, id) in ((out, _spkB_ID_C), (out_app, _spkB_ID_D))
                x_frame = _spkB_query(spk, id, 399, [150.0]; frame = "SPKB_TEST_FRAME")[1]
                x_j2000 = _spkB_query(spk, id, 399, [150.0]; frame = "J2000")[1]
                @test x_frame ≈ _spkB_linear_state(150.0) rtol = 1e-12
                @test x_j2000 ≈ x_frame rtol = 1e-12
            end
        finally
            SPICE.unload(fk)
        end
    end
end


@testset "SPK type 13: append_spkw13_segment_to_spk!" begin
    mktempdir() do tmpdir
        out = joinpath(tmpdir, "app.bsp")
        s1 = _spkB_linear_segment(0.0, 200.0, 5)
        s2 = _spkB_linear_segment(200.0, 400.0, 5)
        kw = (spice_id = _spkB_ID_E, center_id = 399, segment_id = "APP")

        e = try
            _spkB_HFEM.append_spkw13_segment_to_spk!(s2, out; segment_index = 2, kw...)
            nothing
        catch err
            err
        end
        @test e isa ErrorException && occursin("Cannot append", e.msg)
        @test !isfile(out)

        _spkB_HFEM.write_spkw13_spk!([s1], out; kw..., verbose = false)
        @test _spkB_HFEM.append_spkw13_segment_to_spk!(s2, out; segment_index = 2, kw...) == abspath(out)
        ets = [10.0, 199.0, 201.0, 390.0]
        for (et, x) in zip(ets, _spkB_query(out, _spkB_ID_E, 399, ets))
            @test x ≈ _spkB_linear_state(et) rtol = 1e-12
        end
        @test [s.segment_id for s in _spkB_HFEM.read_spk_segments(out)] == ["APP_001", "APP_002"]

        # SPICE rejects an unknown frame inside spkw13: the handle is closed and the
        # SPICE error propagates; the existing segments remain readable
        @test_throws SPICE.SpiceError _spkB_HFEM.append_spkw13_segment_to_spk!(s2, out;
            segment_index = 3, kw..., ref_frame_name = "SPKB_NOT_A_FRAME")
        @test _spkB_query(out, _spkB_ID_E, 399, [100.0])[1] ≈ _spkB_linear_state(100.0) rtol = 1e-12
        # the file handle was released: the file can be appended to again
        s3 = _spkB_linear_segment(400.0, 600.0, 5)
        _spkB_HFEM.append_spkw13_segment_to_spk!(s3, out; segment_index = 3, kw...)
        @test _spkB_query(out, _spkB_ID_E, 399, [500.0])[1] ≈ _spkB_linear_state(500.0) rtol = 1e-12
    end
end


@testset "SPK incremental: append_ode_sol_to_spk!" begin
    mktempdir() do tmpdir
        params = (TU = 100.0, DU = 1000.0, VU = 10.0, et0 = 500.0, naif_ids = ["399"])
        # dynamically consistent arc: d(position)/dt = velocity (DU/TU == VU here)
        arc1 = _SpkBArc([0.0, 4.0], [1.0, 2.0, 3.0, 0.1, -0.2, 0.3], [0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
        arc2 = _SpkBArc([4.0, 8.0], arc1.x0, arc1.xdot)
        ref(t_nd) = vcat(arc1(t_nd)[1:3] .* params.DU, arc1(t_nd)[4:6] .* params.VU)

        sol_spk = joinpath(tmpdir, "sol.bsp")
        write(sol_spk, "not an spk")

        # append=false replaces an existing file; et0 is taken from `parameters`
        r1 = _spkB_HFEM.append_ode_sol_to_spk!(arc1, params; output_spk = sol_spk,
            append = false, dt_sec = 50.0, spice_id = _spkB_ID_B)
        @test r1.appended == false
        @test r1.segment_index == 1
        @test r1.epoch_range == (500.0, 900.0)
        @test r1.point_count == 9
        # explicit et0 and a center given by name
        r2 = _spkB_HFEM.append_ode_sol_to_spk!(arc2, 500.0, (TU = 100.0, DU = 1000.0, VU = 10.0);
            output_spk = sol_spk, append = true, dt_sec = 50.0, spice_id = _spkB_ID_B, center_id = "EARTH")
        @test r2.appended == true
        @test r2.segment_index == 2
        @test r2.epoch_range == (900.0, 1300.0)
        @test sort(readdir(tmpdir)) == ["sol.bsp"]
        ts_nd = [0.5, 3.9, 4.1, 8.0]
        xs = _spkB_query(sol_spk, _spkB_ID_B, 399, params.et0 .+ ts_nd .* params.TU)
        for (t_nd, x) in zip(ts_nd, xs)
            @test x[1:3] ≈ ref(t_nd)[1:3] atol = 1e-7
            @test x[4:6] ≈ ref(t_nd)[4:6] atol = 1e-10
        end

        # argument checks happen before anything is written
        bad = (output_spk = joinpath(tmpdir, "bad.bsp"), spice_id = _spkB_ID_B, append = false)
        e = try _spkB_HFEM.append_ode_sol_to_spk!(arc1, (TU = 100.0, DU = 1000.0, VU = 10.0, naif_ids = ["399"]); bad...); nothing; catch err; err; end
        @test e isa ErrorException && occursin("no `et0` property", e.msg)
        e = try _spkB_HFEM.append_ode_sol_to_spk!(arc1, 500.0, (TU = 100.0, DU = 1000.0, VU = 10.0); bad...); nothing; catch err; err; end
        @test e isa ErrorException && occursin("Pass `center_id`", e.msg)
        @test_throws ErrorException _spkB_HFEM.append_ode_sol_to_spk!(arc1, params; bad..., dt_sec = 0.0)
        e = try _spkB_HFEM.append_ode_sol_to_spk!(arc1, params; bad..., output_spk = joinpath(tmpdir, "x.txt")); nothing; catch err; err; end
        @test e isa ErrorException && occursin("must end in `.bsp`", e.msg)
        @test sort(readdir(tmpdir)) == ["sol.bsp"]
    end
end


@testset "SPK metadata: build_spk_metadata branches" begin
    mktempdir() do tmpdir
        jumps = joinpath(tmpdir, "jumps.txt")
        entries = [(et = 1.0, dv = [1.0, 0.0, 0.0])]

        # trajectory jumps only: entries merged on request, coverage per window, extra metadata
        meta = _spkB_HFEM.build_spk_metadata(
            output_spk = joinpath(tmpdir, "a.bsp"),
            spice_id = -123456,
            center_id = 301,
            trajectory_maneuver_file = jumps,
            windows = [(start_et = 10.0, end_et = 20.0, coast_window = (1, 2)), (start_et = 20.0, end_et = 35.0)],
            maneuver_summary = Dict("count" => 1, "stats" => Dict("max" => 2.0)),
            maneuver_entries = entries,
            merge_maneuvers_into_metadata = true,
            extra_metadata = (run = :mc01, seeds = (1, 2)),
        )
        @test meta["schema"] == "HighFidelityEphemerisModel.SPKMetadata.v2"
        @test meta["spk"]["object_id"] == -123456 && meta["spk"]["object_name"] === nothing
        @test meta["spk"]["center_id"] == 301 && meta["spk"]["center_name"] == "MOON"
        @test meta["spk"]["spk_type"] == 13
        @test collect(keys(meta["maneuvers"])) == ["trajectory_jumps"]
        m = meta["maneuvers"]["trajectory_jumps"]
        @test m["file"] == abspath(jumps)
        @test m["count"] == 1 && m["stats"] == Dict("max" => 2.0)
        @test m["entries"] == [Dict("et" => 1.0, "dv" => [1.0, 0.0, 0.0])]
        @test meta["products"]["trajectory_jump_maneuver_file"] == abspath(jumps)
        @test meta["products"]["maneuver_file"] === nothing
        @test meta["extra"] == Dict("run" => "mc01", "seeds" => Any[1, 2])
        @test meta["coverage"]["start_et"] == 10.0 && meta["coverage"]["end_et"] == 35.0
        w = meta["coverage"]["windows"]
        @test w[1] == Dict("segment_index" => 1, "start_et" => 10.0, "end_et" => 20.0, "coast_window" => [1, 2])
        @test !haskey(w[2], "coast_window")
        @test meta["sampling"] == Dict("dt_sec" => nothing, "segment_gap_sec" => nothing, "segment_count" => 2)
        @test meta["force_model"] == Dict{String,Any}()   # no parameters

        # coverage is the min/max over the windows, not first/last (ranges given out of order)
        meta_u = _spkB_HFEM.build_spk_metadata(output_spk = "u.bsp",
            windows = [(start_et = 20.0, end_et = 35.0), (start_et = 10.0, end_et = 20.0)])
        @test meta_u["coverage"]["start_et"] == 10.0 && meta_u["coverage"]["end_et"] == 35.0

        # nothing given: no maneuver sections, no coverage, no extra
        meta_e = _spkB_HFEM.build_spk_metadata(output_spk = "b.bsp", merge_maneuvers_into_metadata = true)
        @test meta_e["maneuvers"] == Dict{String,Any}()
        @test meta_e["coverage"]["start_et"] === nothing
        @test !haskey(meta_e, "extra")

        # OCP controls with only the vector-norm cost; merged without entries -> empty list;
        # force-model dictionary merged over the inferred one
        ocp = joinpath(tmpdir, "ocp.txt")
        meta2 = _spkB_HFEM.build_spk_metadata(
            output_spk = "c.bsp",
            ocp_control_file = ocp,
            ocp_control_summary = Dict{String,Any}("total_control_vector_norm_mps" => 0.5),
            merge_maneuvers_into_metadata = true,
            parameters = (TU = 2.0, DU = 3, VU = 1.5, include_srp = false),
            et0 = 7,
            force_model_metadata = (dynamics = :Nbody, SRP = Dict("Cr" => 1.2)),
        )
        m2 = meta2["maneuvers"]["ocp_control"]
        @test m2["primary_cost_key"] == "total_control_vector_norm_mps"
        @test m2["file"] == abspath(ocp)
        @test m2["entries"] == Any[]
        @test !haskey(meta2["maneuvers"], "trajectory_jumps")
        @test meta2["products"]["maneuver_file"] == abspath(ocp)
        @test meta2["time_base"] == Dict("et0" => 7.0, "TU_seconds" => 2.0, "DU_km" => 3, "VU_kmps" => 1.5)
        @test meta2["force_model"] == Dict("dynamics" => "Nbody", "SRP" => Dict("enabled" => false, "Cr" => 1.2))

        meta3 = _spkB_HFEM.build_spk_metadata(output_spk = "d.bsp",
            ocp_control_summary = Dict{String,Any}("total_control_scalar_mps" => 1.0),
            ocp_control_entries = [(et = 2.0,)], merge_maneuvers_into_metadata = true)
        @test meta3["maneuvers"]["ocp_control"]["primary_cost_key"] == "total_control_scalar_mps"
        @test meta3["maneuvers"]["ocp_control"]["entries"] == [Dict("et" => 2.0)]
        @test meta3["maneuvers"]["ocp_control"]["file"] === nothing
    end
end


@testset "SPK metadata: force-model inference and dict merging" begin
    @test _spkB_HFEM.infer_force_model_metadata(nothing) == Dict{String,Any}()

    # harmonics, SRP and drag on: central body, degree, GM and radius from the data dictionary
    p_on = (naif_ids = ["301", "399"], GMs = (1.0, 0.0123), naif_frame = "J2000",
        spherical_harmonics_data = Dict("nmax" => 8, "GM" => 4902.8, "REFERENCE RADIUS" => 1738.0, "Cnm" => [1.0]),
        frame_PCPF = "MOON_PA", include_drag = true, k_drag = 0.1, omega_atm = [0.0, 0.0, 7.29e-5],
        include_srp = true, k_srp_cannonball = 1e-9)
    fm = _spkB_HFEM.infer_force_model_metadata(p_on)
    @test fm["naif_ids"] == ["301", "399"]
    @test fm["GMs"] == Any[1.0, 0.0123]
    @test fm["frame"] == "J2000"
    @test fm["spherical_harmonics"] == Dict("enabled" => true, "body" => "301", "degree" => 8, "GM" => 4902.8,
        "reference_radius" => 1738.0, "body_fixed_frame" => "MOON_PA")
    @test fm["drag"] == Dict("enabled" => true, "k_drag_canonical" => 0.1, "omega_atm_radps" => [0.0, 0.0, 7.29e-5])
    @test fm["SRP"] == Dict("enabled" => true, "k_cannonball_canonical" => 1e-9)
    @test !haskey(fm, "parameters_type")   # duck-typed parameters

    # everything off: coefficients computed by the constructors are not reported
    p_off = (spherical_harmonics_data = nothing, frame_PCPF = "ITRF93", include_drag = false, k_drag = 0.1,
        include_srp = false, k_srp_cannonball = 47.0)
    @test _spkB_HFEM.infer_force_model_metadata(p_off) == Dict(
        "spherical_harmonics" => Dict("enabled" => false),
        "drag" => Dict("enabled" => false),
        "SRP" => Dict("enabled" => false))

    # nothing recognizable -> no entries
    @test _spkB_HFEM.infer_force_model_metadata((foo = 1,)) == Dict{String,Any}()

    # the package's parameter struct
    p_spice = _spkB_HFEM.SpiceParameters(
        0.0, 1737.4, [4902.8, 398600.4, 1.327e11], ["301", "399", "10"], "J2000", "NONE";
        filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab"),
        nmax = 8, frame_PCPF = "MOON_PA", include_srp = false,
    )
    fm_spice = _spkB_HFEM.infer_force_model_metadata(p_spice)
    @test fm_spice["parameters_type"] == "SpiceParameters"
    @test fm_spice["naif_ids"] == ["301", "399", "10"]
    sh = fm_spice["spherical_harmonics"]
    @test sh["enabled"] == true && sh["body"] == "301" && sh["degree"] == 8 && sh["body_fixed_frame"] == "MOON_PA"
    @test haskey(sh, "GM") && haskey(sh, "reference_radius")
    # Sun in `naif_ids` gives a nonzero k_srp_cannonball even with SRP off: not reported
    @test p_spice.k_srp_cannonball != 0.0
    @test fm_spice["SRP"] == Dict("enabled" => false)

    base = Dict{String,Any}("a" => 1, "n" => Dict{String,Any}("x" => 1, "y" => Dict{String,Any}("z" => 1)), "s" => "keep")
    over = Dict{String,Any}("a" => 2, "n" => Dict{String,Any}("y" => Dict{String,Any}("w" => 2), "v" => 3), "s" => Dict("now" => "dict"))
    merged = _spkB_HFEM._deep_merge_dicts!(base, over)
    @test merged === base
    @test base == Dict("a" => 2, "n" => Dict("x" => 1, "v" => 3, "y" => Dict("z" => 1, "w" => 2)),
        "s" => Dict("now" => "dict"))
end
