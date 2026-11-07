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


"""Object whose `et0` property access throws; `ET0` works."""
struct _SpkBThrowingProps end
Base.propertynames(::_SpkBThrowingProps, private::Bool = false) = (:et0, :ET0)
function Base.getproperty(::_SpkBThrowingProps, name::Symbol)
    name === :et0 && error("broken property")
    name === :ET0 && return 42.0
    error("no property $(name)")
end


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


@testset "SPK utils: property/path/summary helpers" begin
    # _get_et0_from_parameters picks the first available alias
    @test _spkB_HFEM._get_et0_from_parameters((ET0 = 12.5, TU = 1.0)) == 12.5
    @test _spkB_HFEM._get_et0_from_parameters((t0_et = 3.0,)) == 3.0
    @test _spkB_HFEM._get_et0_from_parameters((et0 = 1.0, ET0 = 2.0)) == 1.0
    @test _spkB_HFEM._get_et0_from_parameters((TU = 1.0,)) === nothing
    @test _spkB_HFEM._get_et0_from_parameters(nothing) === nothing
    # an alias whose property access throws is skipped, the next one is used
    @test _spkB_HFEM._get_et0_from_parameters(_SpkBThrowingProps()) == 42.0

    # _add_if_found!: adds a JSON-safe value only if a property was found
    d = Dict{String,Any}()
    _spkB_HFEM._add_if_found!(d, "sym", (a = :abc,), (:a,))
    _spkB_HFEM._add_if_found!(d, "missing", (a = 1,), (:b, :c))
    @test d == Dict{String,Any}("sym" => "abc")

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

    # _tool_path
    @test _spkB_HFEM._tool_path("C:\\work\\out\\file.bsp") == "C:/work/out/file.bsp"
    @test _spkB_HFEM._tool_path(SubString("a/b\\c", 1)) == "a/b/c"
end


@testset "SPK utils: printing helpers" begin
    out = _spkB_capture() do
        _spkB_HFEM._print_spk_pipeline_summary(
            output_spk = joinpath(pwd(), "run", "traj.bsp"),
            maneuver_txt = joinpath(pwd(), "run", "jumps.txt"),
            ocp_maneuver_txt = joinpath(pwd(), "run", "ocp.txt"),
            metadata_json = joinpath(pwd(), "run", "meta.json"),
            segment_count = 7,
            intermediate_dir = joinpath(pwd(), "run", "tmp"),
            maneuver_summary = nothing,
            ocp_control_summary = Dict("total_control_scalar_mps" => 1.25),
        )
    end
    @test occursin("SPK generation complete", out)
    @test occursin(r"segments\s+: 7", out)
    @test occursin("SPK                         : " * joinpath("run", "traj.bsp"), out)
    @test occursin("maneuvers                   : " * joinpath("run", "jumps.txt"), out)
    @test occursin("OCP maneuvers               : " * joinpath("run", "ocp.txt"), out)
    @test occursin("metadata JSON               : " * joinpath("run", "meta.json"), out)
    @test occursin("intermediates               : " * joinpath("run", "tmp"), out)
    @test occursin("total control (OCP scalar)  : 1.250000e+00 m/s", out)

    # optional products omitted; identical OCP/maneuver paths are listed once
    same = joinpath(pwd(), "m.txt")
    out2 = _spkB_capture() do
        _spkB_HFEM._print_spk_pipeline_summary(
            output_spk = joinpath(pwd(), "x.bsp"),
            maneuver_txt = same,
            ocp_maneuver_txt = same,
            metadata_json = nothing,
            segment_count = 1,
            intermediate_dir = nothing,
            maneuver_summary = nothing,
            ocp_control_summary = nothing,
        )
    end
    @test occursin("maneuvers                   : m.txt", out2)
    @test !occursin("OCP maneuvers", out2)
    @test !occursin("metadata JSON", out2)
    @test !occursin("intermediates", out2)
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


@testset "SPK utils: segment state/setup file discovery" begin
    mktempdir() do tmpdir
        Y1 = reduce(hcat, [_spkB_linear_state(et) for et in (100.0, 150.0, 200.0)])
        Y2 = reduce(hcat, [_spkB_linear_state(et) for et in (200.0, 260.0)])
        Y10 = reduce(hcat, [_spkB_linear_state(et) for et in (900.0, 950.0)])
        _spkB_HFEM.write_spk_states_file(joinpath(tmpdir, "seg_002_states.txt"), [200.0, 260.0], Y2)
        _spkB_HFEM.write_spk_states_file(joinpath(tmpdir, "seg_001_states.txt"), [100.0, 150.0, 200.0], Y1)
        # not matching the 3-digit pattern: ignored by the directory scan
        _spkB_HFEM.write_spk_states_file(joinpath(tmpdir, "seg_10_states.txt"), [900.0, 950.0], Y10)
        # header-only file: contributes no epochs
        write(joinpath(tmpdir, "seg_003_states.txt"), "# ETSECONDS\n")
        for name in ("seg_002_setup.txt", "seg_001_setup.txt", "seg_1_setup.txt", "notes.txt")
            write(joinpath(tmpdir, name), "x\n")
        end

        @test _spkB_HFEM.list_segment_state_files(tmpdir) ==
            [joinpath(tmpdir, "seg_00$(k)_states.txt") for k in 1:3]
        @test _spkB_HFEM.list_segment_setup_files(tmpdir) ==
            [joinpath(tmpdir, "seg_00$(k)_setup.txt") for k in 1:2]

        @test _spkB_HFEM.read_segment_boundary_epochs(tmpdir) == [100.0, 200.0, 260.0]
        @test _spkB_HFEM.read_segment_boundary_epochs(tmpdir, 1) == [100.0, 200.0]
        @test _spkB_HFEM.read_segment_boundary_epochs(tmpdir, 3) == [100.0, 200.0, 260.0]
        err = try
            _spkB_HFEM.read_segment_boundary_epochs(tmpdir, 4)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("Missing states file", err.msg) && occursin("seg_004_states.txt", err.msg)
    end
end


@testset "SPK type 13: id/segment-id/path helpers" begin
    rid = _spkB_HFEM._resolve_spice_integer_id
    @test rid(-5, "EARTH"; label = "spice") === -5          # ID takes precedence over name
    @test rid(nothing, "EARTH"; label = "center") === 399
    @test rid(nothing, SubString("MOON", 1); label = "center") === 301
    e1 = try rid(nothing, "NOT_A_REAL_BODY_XYZ"; label = "center"); nothing; catch e; e; end
    @test e1 isa ErrorException && occursin("Could not resolve center name", e1.msg)
    e2 = try rid(nothing, nothing; label = "spice"); nothing; catch e; e; end
    @test e2 isa ErrorException && occursin("`spice_id` or `spice_name`", e2.msg)

    sid = _spkB_HFEM._spkw13_segment_id
    @test sid("ARC", 7) == "ARC"
    @test sid("ARC", 7; segment_id_per_seg = true) == "ARC_007"
    @test_throws ErrorException sid(repeat("X", 38), 1; segment_id_per_seg = true)
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
                segs, out; spice_id = _spkB_ID_A, center_name = "EARTH",
                segment_id = "SPKB", segment_id_per_seg = true, polynom_degree = 5,
                internal_file_name = repeat("I", 80), verbose = true, show_progress = false,
            )
        end
        @test isfile(out)
        @test occursin("SPK complete: ", printed)
        @test sort(readdir(tmpdir)) == ["w13.bsp"]   # temporary BSP was installed, not left behind
        ets = [0.0, 123.0, 400.0, 555.5, 800.0]
        for (et, x) in zip(ets, _spkB_query(out, _spkB_ID_A, 399, ets))
            @test x ≈ _spkB_linear_state(et) rtol = 1e-12
        end

        # overwrite an existing kernel with different content (safe-install backup path)
        v2 = [-1.0, 2.0, 0.25]
        segs2 = [_spkB_linear_segment(0.0, 800.0, 11; v = v2)]
        _spkB_HFEM.write_spkw13_spk!(segs2, out; spice_id = _spkB_ID_A, center_id = 399,
            verbose = false, show_progress = false)
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
        tmp_spk = joinpath(tmpdir, "staging", "stage.bsp")
        e = try
            _spkB_HFEM.write_spkw13_spk!(segs, out; spice_id = _spkB_ID_A, center_id = 399,
                polynom_degree = 6, tmp_spk = tmp_spk, verbose = false)
            nothing
        catch err
            err
        end
        @test e isa ErrorException && occursin("must be odd", e.msg)
        @test !isfile(tmp_spk)
        @test read(out) == bytes

        # failure inside SPICE after `spkopn` with zero segments written (unknown
        # frame): `spkcls` refuses to close an empty SPK, so the handle must be
        # released via `dafcls` and the staged file removed (EBUSY on Windows otherwise)
        e = try
            _spkB_HFEM.write_spkw13_spk!(segs, out; spice_id = _spkB_ID_A, center_id = 399,
                ref_frame_name = "SPKB_NOT_A_FRAME", tmp_spk = tmp_spk, verbose = false)
            nothing
        catch err
            err
        end
        @test e isa SPICE.SpiceError
        @test !isfile(tmp_spk)
        @test read(out) == bytes

        # a stale file at the explicit temporary path is replaced, not appended to
        write(tmp_spk, "stale")
        out_b = joinpath(tmpdir, "w13b.bsp")
        _spkB_HFEM.write_spkw13_spk!(segs, out_b; spice_id = _spkB_ID_B, center_id = 399,
            tmp_spk = tmp_spk, verbose = false)
        @test !isfile(tmp_spk) && isfile(out_b)
        @test _spkB_query(out_b, _spkB_ID_B, 399, [200.0])[1] ≈ _spkB_linear_state(200.0) rtol = 1e-12
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
            _spkB_HFEM.append_spkw13_segment_to_spk!(seg; output_spk = out_app, segment_index = 1,
                spice_id = _spkB_ID_D, center_id = 399, ref_frame_name = "SPKB_TEST_FRAME",
                frame_def_file = fk)
            # the frame is now known to SPICE
            @test SPICE.namfrm("SPKB_TEST_FRAME") == 1771001
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
        kw = (spice_id = _spkB_ID_E, center_id = 399, segment_id = "APP", segment_id_per_seg = true)

        @test_throws ErrorException _spkB_HFEM.append_spkw13_segment_to_spk!(s1; output_spk = joinpath(tmpdir, "a.txt"),
            segment_index = 1, kw...)
        e = try
            _spkB_HFEM.append_spkw13_segment_to_spk!(s1; output_spk = out, segment_index = 1, append = true, kw...)
            nothing
        catch err
            err
        end
        @test e isa ErrorException && occursin("Cannot append", e.msg)
        @test !isfile(out)

        # create (append inferred as false), then append (inferred as true)
        @test _spkB_HFEM.append_spkw13_segment_to_spk!(s1; output_spk = out, segment_index = 1, kw...) == abspath(out)
        @test _spkB_HFEM.append_spkw13_segment_to_spk!(s2; output_spk = out, segment_index = 2,
            internal_file_name = repeat("J", 70), kw...) == abspath(out)
        ets = [10.0, 199.0, 201.0, 390.0]
        for (et, x) in zip(ets, _spkB_query(out, _spkB_ID_E, 399, ets))
            @test x ≈ _spkB_linear_state(et) rtol = 1e-12
        end

        # explicit append=false on an existing file is refused
        bytes = read(out)
        e2 = try
            _spkB_HFEM.append_spkw13_segment_to_spk!(s2; output_spk = out, segment_index = 3, append = false, kw...)
            nothing
        catch err
            err
        end
        @test e2 isa ErrorException && occursin("append=false", e2.msg)
        @test read(out) == bytes

        # SPICE rejects an unknown frame inside spkw13: the handle is closed and the
        # SPICE error propagates; the existing segments remain readable
        @test_throws SPICE.SpiceError _spkB_HFEM.append_spkw13_segment_to_spk!(s2; output_spk = out,
            segment_index = 3, ref_frame_name = "SPKB_NOT_A_FRAME", kw...)
        @test _spkB_query(out, _spkB_ID_E, 399, [100.0])[1] ≈ _spkB_linear_state(100.0) rtol = 1e-12
        # the file handle was released: the file can be appended to again
        s3 = _spkB_linear_segment(400.0, 600.0, 5)
        _spkB_HFEM.append_spkw13_segment_to_spk!(s3; output_spk = out, segment_index = 3, kw...)
        @test _spkB_query(out, _spkB_ID_E, 399, [500.0])[1] ≈ _spkB_linear_state(500.0) rtol = 1e-12
    end
end


@testset "SPK incremental: state files and solution segments" begin
    mktempdir() do tmpdir
        # --- _read_spk_states_file_for_spkw13 / append_state_file_to_spk!
        @test_throws ErrorException _spkB_HFEM._read_spk_states_file_for_spkw13(joinpath(tmpdir, "none.txt"))

        bad = joinpath(tmpdir, "bad.txt")
        write(bad, "# ETSECONDS\n0.0,1,2,3,4,5\n")
        e = try _spkB_HFEM._read_spk_states_file_for_spkw13(bad); nothing; catch err; err; end
        @test e isa ErrorException && occursin("Expected 7 comma-separated fields", e.msg)

        one = joinpath(tmpdir, "one.txt")
        write(one, "# ETSECONDS\n0.0,1,2,3,4,5,6\n\n")
        e = try _spkB_HFEM._read_spk_states_file_for_spkw13(one); nothing; catch err; err; end
        @test e isa ErrorException && occursin("at least two states", e.msg)

        ts1 = collect(0.0:50.0:300.0)
        ts2 = collect(300.0:50.0:600.0)
        f1 = joinpath(tmpdir, "seg_001_states.txt")
        f2 = joinpath(tmpdir, "seg_002_states.txt")
        _spkB_HFEM.write_spk_states_file(f1, ts1, reduce(hcat, _spkB_linear_state.(ts1)))
        _spkB_HFEM.write_spk_states_file(f2, ts2, reduce(hcat, _spkB_linear_state.(ts2)))
        # blank lines (and whitespace around fields) are tolerated
        open(f2, "a") do io
            println(io)
            println(io, "   ")
        end
        seg = _spkB_HFEM._read_spk_states_file_for_spkw13(f2)
        @test seg.epochs == ts2
        @test seg.first == 300.0 && seg.last == 600.0
        @test seg.states[end] ≈ _spkB_linear_state(600.0) rtol = 1e-14

        out = joinpath(tmpdir, "from_states.bsp")
        r1 = _spkB_HFEM.append_state_file_to_spk!(f1; output_spk = out, segment_index = 1,
            spice_id = _spkB_ID_A, center_id = 399)
        r2 = _spkB_HFEM.append_state_file_to_spk!(f2; output_spk = out, segment_index = 2,
            spice_id = _spkB_ID_A, center_id = 399)
        @test r1.appended == false && r2.appended == true
        @test r1.output_spk == abspath(out) && r2.state_file == f2 && r2.setup_file === nothing
        @test r1.epoch_range == (0.0, 300.0) && r2.epoch_range == (300.0, 600.0)
        ets = [25.0, 299.0, 333.0, 600.0]
        for (et, x) in zip(ets, _spkB_query(out, _spkB_ID_A, 399, ets))
            @test x ≈ _spkB_linear_state(et) rtol = 1e-12
        end

        # --- append_solution_segment_to_spk!
        params = (TU = 100.0, DU = 1000.0, VU = 10.0, et0 = 500.0)
        # dynamically consistent arc: d(position)/dt = velocity (DU/TU == VU here)
        arc1 = _SpkBArc([0.0, 4.0], [1.0, 2.0, 3.0, 0.1, -0.2, 0.3], [0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
        arc2 = _SpkBArc([4.0, 8.0], arc1.x0, arc1.xdot)
        ref(t_nd) = vcat(arc1(t_nd)[1:3] .* params.DU, arc1(t_nd)[4:6] .* params.VU)

        sol_spk = joinpath(tmpdir, "sol.bsp")
        write(sol_spk, "not an spk")
        # existing file and overwrite=false: refused, file untouched
        e = try
            _spkB_HFEM.append_solution_segment_to_spk!(arc1, params; output_spk = sol_spk,
                segment_index = 1, append = false, overwrite = false, spice_id = _spkB_ID_B, center_id = 399)
            nothing
        catch err
            err
        end
        @test e isa ErrorException && occursin("overwrite=false", e.msg)
        @test read(sol_spk, String) == "not an spk"

        # existing file and overwrite=true: removed and recreated; et0 taken from `parameters`
        r1 = _spkB_HFEM.append_solution_segment_to_spk!(arc1, params; output_spk = sol_spk,
            segment_index = 1, append = false, dt_sec = 50.0, spice_id = _spkB_ID_B, center_id = 399)
        @test r1.appended == false
        @test r1.epoch_range == (500.0, 900.0)
        @test r1.point_count == 9
        @test r1.state_file === nothing
        # et0 alias `ET0` is also accepted by the two-argument method
        r2 = _spkB_HFEM.append_solution_segment_to_spk!(arc2, (TU = 100.0, DU = 1000.0, VU = 10.0, ET0 = 500.0);
            output_spk = sol_spk, segment_index = 2, dt_sec = 50.0, spice_id = _spkB_ID_B, center_id = 399)
        @test r2.appended == true
        @test r2.epoch_range == (900.0, 1300.0)
        ts_nd = [0.5, 3.9, 4.1, 8.0]
        xs = _spkB_query(sol_spk, _spkB_ID_B, 399, params.et0 .+ ts_nd .* params.TU)
        for (t_nd, x) in zip(ts_nd, xs)
            @test x[1:3] ≈ ref(t_nd)[1:3] atol = 1e-7
            @test x[4:6] ≈ ref(t_nd)[4:6] atol = 1e-10
        end

        # no recognizable et0 in `parameters`
        e = try
            _spkB_HFEM.append_solution_segment_to_spk!(arc1, (TU = 100.0, DU = 1000.0, VU = 10.0);
                output_spk = joinpath(tmpdir, "no_et0.bsp"), segment_index = 1, spice_id = _spkB_ID_B, center_id = 399)
            nothing
        catch err
            err
        end
        @test e isa ErrorException && occursin("No `et0` argument", e.msg)
        @test !isfile(joinpath(tmpdir, "no_et0.bsp"))
    end
end


@testset "SPK incremental: prepare_spk_output!" begin
    mktempdir() do tmpdir
        out = joinpath(tmpdir, "nested", "p.bsp")
        mkpath(dirname(out))
        write(out, "old")
        e = try _spkB_HFEM.prepare_spk_output!(out; overwrite = false); nothing; catch err; err; end
        @test e isa ErrorException && occursin("overwrite=false", e.msg)
        @test read(out, String) == "old"
        @test _spkB_HFEM.prepare_spk_output!(out; overwrite = true) == abspath(out)
        @test !isfile(out) && isdir(dirname(out))
    end
end


@testset "SPK metadata: build_spk_metadata branches" begin
    mktempdir() do tmpdir
        jumps = joinpath(tmpdir, "jumps.txt")
        entries = [(et = 1.0, dv = [1.0, 0.0, 0.0])]

        # no OCP summary: the trajectory-jump diagnostic becomes the primary product,
        # entries merged on request, maneuver_summary deep-merged, extra metadata stored
        meta = _spkB_HFEM.build_spk_metadata(
            output_spk = joinpath(tmpdir, "a.bsp"),
            maneuver_file = jumps,
            epoch_ranges = [(10.0, 20.0), (20.0, 35.0)],
            coast_windows = [(1, 2)],
            maneuver_summary = Dict("count" => 1, "stats" => Dict("max" => 2.0)),
            maneuver_entries = entries,
            merge_maneuvers_into_metadata = true,
            extra_metadata = (run = :mc01, seeds = (1, 2)),
        )
        m = meta["maneuvers"]
        @test m["is_primary"] == true
        @test m["type"] == "trajectory_jump_diagnostic"
        @test m["file"] == abspath(jumps)
        @test m["count"] == 1 && m["stats"] == Dict("max" => 2.0)
        @test m["merged_into_metadata"] == true
        @test m["entries"] == [Dict("et" => 1.0, "dv" => [1.0, 0.0, 0.0])]
        @test meta["products"]["maneuver_file"] == abspath(jumps)
        @test meta["products"]["ocp_maneuver_file"] === nothing
        @test meta["extra"] == Dict("run" => "mc01", "seeds" => Any[1, 2])
        @test meta["coverage"]["start_et"] == 10.0 && meta["coverage"]["end_et"] == 35.0
        w = meta["coverage"]["windows"]
        @test w[1]["coast_window"] == [1, 2] && !haskey(w[2], "coast_window")
        @test meta["force_model"] == Dict{String,Any}()   # no parameters

        # coverage is the min/max over the windows, not first/last (ranges given out of order)
        meta_u = _spkB_HFEM.build_spk_metadata(output_spk = "u.bsp", epoch_ranges = [(20.0, 35.0), (10.0, 20.0)])
        @test meta_u["coverage"]["start_et"] == 10.0 && meta_u["coverage"]["end_et"] == 35.0

        # merge requested but no entries supplied -> empty list
        meta_e = _spkB_HFEM.build_spk_metadata(output_spk = "b.bsp", merge_maneuvers_into_metadata = true)
        @test meta_e["maneuvers"]["entries"] == Any[]
        @test meta_e["coverage"]["start_et"] === nothing
        @test !haskey(meta_e, "extra")

        # OCP summary with only the vector-norm cost, entries merged
        ocp = joinpath(tmpdir, "ocp.txt")
        meta2 = _spkB_HFEM.build_spk_metadata(
            output_spk = "c.bsp",
            maneuver_file = jumps,
            ocp_control_file = ocp,
            ocp_control_summary = Dict{String,Any}("total_control_vector_norm_mps" => 0.5,
                "cost_convention_vector_norm" => "sum(norm.(u_i))"),
            ocp_control_entries = [(et = 2.0,)],
            merge_ocp_maneuvers_into_metadata = true,
            force_model_metadata = (dynamics = :Nbody,),
        )
        m2 = meta2["maneuvers"]
        @test m2["type"] == "ocp_control" && m2["is_primary"] == true
        @test m2["primary_cost_key"] == "total_control_vector_norm_mps"
        @test m2["cost_convention"] == "sum(norm.(u_i))"
        @test m2["entries"] == [Dict("et" => 2.0)]
        @test m2["trajectory_jumps"]["is_primary"] == false
        @test meta2["products"]["maneuver_file"] == abspath(ocp)
        @test meta2["products"]["trajectory_jump_maneuver_file"] == abspath(jumps)
        @test meta2["force_model"] == Dict("dynamics" => "Nbody")

        meta3 = _spkB_HFEM.build_spk_metadata(output_spk = "d.bsp",
            ocp_control_summary = Dict{String,Any}("total_control_scalar_mps" => 1.0, "units" => "m/s"), merge_ocp_maneuvers_into_metadata = true)
        @test meta3["maneuvers"]["entries"] == Any[]
        @test meta3["maneuvers"]["cost_convention"] === nothing
        @test meta3["maneuvers"]["primary_cost_key"] == "total_control_scalar_mps"
    end
end


@testset "SPK metadata: force-model inference and dict merging" begin
    @test _spkB_HFEM.infer_force_model_metadata(nothing) == Dict{String,Any}()

    # harmonics given as a data dictionary: enabled, degree from "nmax", frame recorded
    p_sh = (naif_ids = [399, 301], GMs = (1.0, 0.0123), spherical_harmonics_data = Dict("nmax" => 8),
        frame_PCPF = "ITRF93", sh_file = "egm.txt", include_drag = true, k_drag = 0.1, omega_atm = 7.29e-5,
        include_srp = true, srp_Cr = 1.3, srp_Am = 0.02, srp_P0 = 4.56e-6, k_srp_cannonball = 1e-9)
    fm = _spkB_HFEM.infer_force_model_metadata(p_sh)
    @test fm["naif_ids"] == [399, 301]
    @test fm["mu_values"] == Any[1.0, 0.0123]
    @test fm["spherical_harmonics"] == Dict("enabled" => true, "degree" => 8, "file" => "egm.txt",
        "body_fixed_frame" => "ITRF93")
    @test fm["drag"] == Dict("enabled" => true, "k_drag_canonical" => 0.1, "omega_atm_radps" => 7.29e-5)
    @test fm["SRP"] == Dict("enabled" => true, "Cr" => 1.3, "A_over_m" => 0.02, "P0" => 4.56e-6,
        "k_cannonball_canonical" => 1e-9)

    # explicit degree overrides the data dictionary's nmax
    @test _spkB_HFEM.infer_force_model_metadata((spherical_harmonics_data = Dict("nmax" => 8), nmax = 4))["spherical_harmonics"]["degree"] == 4
    # harmonics off: the body-fixed frame is not reported as a gravity frame
    fm_off = _spkB_HFEM.infer_force_model_metadata((use_spherical_harmonics = false, frame_PCPF = "ITRF93"))
    @test fm_off["spherical_harmonics"] == Dict("enabled" => false)
    fm_on = _spkB_HFEM.infer_force_model_metadata((use_spherical_harmonics = true, frame_PCPF = "MOON_PA"))
    @test fm_on["spherical_harmonics"] == Dict("enabled" => true, "body_fixed_frame" => "MOON_PA")
    # nothing recognizable -> no sub-dictionaries
    @test _spkB_HFEM.infer_force_model_metadata((foo = 1,)) == Dict{String,Any}()

    kw = _spkB_HFEM._build_force_model_keyword_metadata(dynamics_name = :NbodySH, srp_enabled = true, srp_Cr = 1,
        srp_Am = 2, srp_P0 = 3, spherical_harmonics_enabled = true, spherical_harmonics_body = :EARTH,
        spherical_harmonics_file = "f.txt", spherical_harmonics_nmax = 4.0, spherical_harmonics_frame = :ITRF93)
    @test kw == Dict("dynamics" => "NbodySH",
        "SRP" => Dict("enabled" => true, "Cr" => 1.0, "A_over_m" => 2.0, "P0" => 3.0),
        "spherical_harmonics" => Dict("enabled" => true, "body" => "EARTH", "file" => "f.txt",
            "degree" => 4, "body_fixed_frame" => "ITRF93"))
    @test _spkB_HFEM._build_force_model_keyword_metadata() == Dict{String,Any}()

    base = Dict{String,Any}("a" => 1, "n" => Dict{String,Any}("x" => 1, "y" => Dict{String,Any}("z" => 1)), "s" => "keep")
    over = Dict{String,Any}("a" => 2, "n" => Dict{String,Any}("y" => Dict{String,Any}("w" => 2), "v" => 3), "s" => Dict("now" => "dict"))
    merged = _spkB_HFEM._deep_merge_dicts!(base, over)
    @test merged === base
    @test base == Dict("a" => 2, "n" => Dict("x" => 1, "v" => 3, "y" => Dict("z" => 1, "w" => 2)),
        "s" => Dict("now" => "dict"))
end
