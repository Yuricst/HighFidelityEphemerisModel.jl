"""Tests for SPK generation from ODE solution segments"""

using LinearAlgebra
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


"""
Mock coast-arc solution with the minimal `ODESolution`-like interface needed by
`ode_sol_to_spk`.

The real workflow passes an `OrdinaryDiffEq.ODESolution`; this small object keeps
the SPK tests fast and deterministic.
"""
struct MockCoastArc
    t::Vector{Float64}
    x0::Vector{Float64}
    xdot::Vector{Float64}
end


(arc::MockCoastArc)(t) = arc.x0 .+ arc.xdot .* t


function _mock_parameters(; et0 = 1000.0)
    return (
        TU = 100.0,   # seconds per nondimensional time unit
        DU = 1000.0,  # km per nondimensional distance unit
        VU = 10.0,    # km/s per nondimensional velocity unit
        et0 = et0,
        naif_ids = ["399"],
        GMs = [1.0],
        include_srp = false,
    )
end


function _mock_sols()
    r0 = [1.0, 2.0, 3.0]
    v0 = [0.01, -0.02, 0.03]
    x0 = vcat(r0, v0)
    xdot = vcat(v0, zeros(3))

    return [
        MockCoastArc([0.0, 8.0], x0, xdot),
        MockCoastArc([8.0, 16.0], x0, xdot),
    ]
end


function _reference_state(sol, t_nd, parameters)
    x_ref = sol(t_nd)[1:6]
    return vcat(
        x_ref[1:3] .* parameters.DU,
        x_ref[4:6] .* parameters.VU,
    )
end


function _solution_for_time(sols, t_nd)
    for sol in sols
        if sol.t[1] <= t_nd <= sol.t[end]
            return sol
        end
    end
    error("No mock solution covers t_nd = $t_nd.")
end


function _validate_spk_against_sols(
    output_spk,
    sols,
    t_queries_nd,
    parameters;
    spice_id,
    center_id,
    frame = "J2000",
)
    @test isfile(output_spk)

    SPICE.furnsh(output_spk)
    try
        for t_nd in t_queries_nd
            et = parameters.et0 + t_nd * parameters.TU

            # Query the generated BSP and compare against the source ODE arc.
            state_spk, _ = SPICE.spkezr(
                string(spice_id),
                et,
                frame,
                "NONE",
                string(center_id),
            )

            sol = _solution_for_time(sols, t_nd)
            state_ref = _reference_state(sol, t_nd, parameters)

            @test state_spk[1:3] ≈ state_ref[1:3] atol = 1e-7
            @test state_spk[4:6] ≈ state_ref[4:6] atol = 1e-10
        end
    finally
        SPICE.unload(output_spk)
    end
end


function test_spk_helper_utilities()
    ts = HighFidelityEphemerisModel.build_segment_epochs(
        0.0,
        3600.0;
        dt_sec = 1800.0,
    )

    @test ts == [0.0, 1800.0, 3600.0]

    windows = HighFidelityEphemerisModel.default_coast_windows([:a, :b, :c])
    @test windows == [(1, 1), (2, 2), (3, 3)]
end


function test_spk_state_sampling()
    parameters = (
        TU = 100.0,
        DU = 10.0,
        VU = 0.01,
    )

    sols = [
        MockCoastArc(
            [0.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ),
    ]

    et0 = 1000.0

    sampled = HighFidelityEphemerisModel.sample_segmented_states_for_spk(
        sols,
        [(1, 1)],
        et0,
        parameters;
        dt_sec = 50.0,
        segment_gap_sec = 0.0,
        verbose = false,
        show_progress = false,
    )

    @test length(sampled.segments) == 1
    @test sampled.epoch_ranges[1] == (1000.0, 1100.0)
    @test sampled.point_counts == [3]
    @test sampled.segments[1].epochs == [1000.0, 1050.0, 1100.0]
    @test sampled.segments[1].states[1] ≈ [10.0, 20.0, 30.0, 0.04, 0.05, 0.06]
    @test sampled.segments[1].states[end] ≈ [11.0, 22.0, 33.0, 0.044, 0.055, 0.066]
end


function test_spk_rejects_discontinuous_coast_windows()
    parameters = (
        TU = 100.0,
        DU = 10.0,
        VU = 0.01,
    )
    et0 = 1000.0

    # Contiguous times, but an impulsive Δv between arcs.
    jumping_sols = [
        MockCoastArc(
            [0.0, 1.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            zeros(6),
        ),
        MockCoastArc(
            [1.0, 2.0],
            [1.0, 2.0, 3.0, 0.2, 0.2, 0.3],
            zeros(6),
        ),
    ]

    # Time gap between arcs is also invalid for a merged segment.
    gapped_sols = [
        MockCoastArc(
            [0.0, 1.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            zeros(6),
        ),
        MockCoastArc(
            [1.1, 2.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            zeros(6),
        ),
    ]

    # Continuous adjacent arcs may still be grouped into one SPK segment.
    continuous_sols = [
        MockCoastArc(
            [0.0, 1.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            zeros(6),
        ),
        MockCoastArc(
            [1.0, 2.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            zeros(6),
        ),
    ]

    mktempdir() do tmpdir
        @test_throws ErrorException HighFidelityEphemerisModel.sample_segmented_states_for_spk(
            jumping_sols,
            [(1, 2)],
            et0,
            parameters;
            dt_sec = 50.0,
            segment_gap_sec = 0.0,
            verbose = false,
            show_progress = false,
        )

        @test_throws ErrorException HighFidelityEphemerisModel.sample_segmented_states_for_spk(
            gapped_sols,
            [(1, 2)],
            et0,
            parameters;
            dt_sec = 50.0,
            segment_gap_sec = 0.0,
            verbose = false,
            show_progress = false,
        )

        continuous = HighFidelityEphemerisModel.sample_segmented_states_for_spk(
            continuous_sols,
            [(1, 2)],
            et0,
            parameters;
            dt_sec = 50.0,
            segment_gap_sec = 0.0,
            verbose = false,
            show_progress = false,
        )
        @test length(continuous.segments) == 1
        @test continuous.epoch_ranges[1] == (1000.0, 1200.0)
    end
end


"""Mock arc whose evaluation always fails; used to trigger errors mid-pipeline."""
struct ThrowingCoastArc
    t::Vector{Float64}
end

(arc::ThrowingCoastArc)(t) = error("ThrowingCoastArc evaluated at t = $(t)")


function _constant_arc(t, x)
    return MockCoastArc(collect(Float64, t), collect(Float64, x), zeros(6))
end


function test_spk_coast_window_continuity_tolerances()
    # DU = 10 km, VU = 0.01 km/s.
    parameters = (TU = 100.0, DU = 10.0, VU = 0.01)
    et0 = 1000.0
    x = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    kw = (dt_sec = 50.0, segment_gap_sec = 0.0, verbose = false, show_progress = false)

    sample(sols; extra...) = HighFidelityEphemerisModel.sample_segmented_states_for_spk(
        sols, [(1, 2)], et0, parameters; kw..., extra...,
    )

    # Position defect of 5e-5 DU = 5e-4 km: below the default 1e-3 km.
    small = [_constant_arc([0.0, 1.0], x), _constant_arc([1.0, 2.0], x .+ [5e-5, 0, 0, 0, 0, 0])]
    @test length(sample(small).segments) == 1

    # Position defect of 1e-3 DU = 1e-2 km: rejected by default ...
    large = [_constant_arc([0.0, 1.0], x), _constant_arc([1.0, 2.0], x .+ [1e-3, 0, 0, 0, 0, 0])]
    @test_throws ErrorException sample(large)
    # ... accepted with a looser absolute tolerance, a relative tolerance,
    # or with the check disabled.
    @test length(sample(large; continuity_pos_tol_km = 2e-2).segments) == 1
    @test length(sample(large; continuity_rtol = 1e-2).segments) == 1
    @test length(sample(large; validate_continuity = false).segments) == 1
    # A tighter tolerance rejects the small defect.
    @test_throws ErrorException sample(small; continuity_pos_tol_km = 1e-4)

    # Velocity defect of 1e-5 VU = 1e-7 km/s: below the default 1e-6 km/s.
    small_v = [_constant_arc([0.0, 1.0], x), _constant_arc([1.0, 2.0], x .+ [0, 0, 0, 1e-5, 0, 0])]
    @test length(sample(small_v).segments) == 1
    @test_throws ErrorException sample(small_v; continuity_vel_tol_kmps = 1e-8)

    # Negative tolerances are rejected.
    @test_throws ErrorException sample(small; continuity_pos_tol_km = -1.0)

    # The error message names the arcs, the boundary time and both jumps.
    err = try
        sample(large)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("global arcs 1 and 2", err.msg)
    @test occursin("position jump", err.msg)
    @test occursin("velocity jump", err.msg)
    @test occursin("continuity_pos_tol_km", err.msg)
end


function test_spk_coast_window_nonfinite_and_callback_messages()
    parameters = (TU = 100.0, DU = 10.0, VU = 0.01)
    et0 = 1000.0
    x = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    kw = (dt_sec = 50.0, segment_gap_sec = 0.0, verbose = false, show_progress = false)

    sample(sols, windows = [(1, 2)]) = HighFidelityEphemerisModel.sample_segmented_states_for_spk(
        sols, windows, et0, parameters; kw...,
    )

    # NaN endpoint state.
    nan_state = [_constant_arc([0.0, 1.0], x), _constant_arc([1.0, 2.0], [NaN, 2.0, 3.0, 0.1, 0.2, 0.3])]
    @test_throws ErrorException sample(nan_state)

    # NaN endpoint time.
    nan_time = [_constant_arc([0.0, NaN], x), _constant_arc([1.0, 2.0], x)]
    @test_throws ErrorException sample(nan_time)

    # Impulse applied by a callback at the final time of arc 2 (of 3): the
    # final time is repeated and `sol(t_f)` is the pre-impulse state.
    x_post = x .+ [0, 0, 0, 0.1, 0, 0]
    callback_sols = [
        _constant_arc([0.0, 1.0], x),
        _constant_arc([1.0, 2.0, 2.0], x),
        _constant_arc([2.0, 3.0], x_post),
    ]
    err = try
        sample(callback_sols, [(1, 1), (2, 3)])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("global arcs 2 and 3", err.msg)
    @test occursin("repeated time", err.msg)
    @test occursin("inside that arc", err.msg)

    # Bad window bounds are still reported.
    @test_throws ErrorException sample(callback_sols, [(1, 4)])
    @test_throws ErrorException sample(callback_sols, [(2, 1)])
end


function test_spk_rejected_window_leaves_no_output()
    parameters = (TU = 100.0, DU = 10.0, VU = 0.01)
    et0 = 1000.0
    x = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]

    # Arcs 2 and 3 are separated by an impulsive Δv.
    sols = [
        _constant_arc([0.0, 1.0], x),
        _constant_arc([1.0, 2.0], x),
        _constant_arc([2.0, 3.0], x .+ [0, 0, 0, 0.1, 0, 0]),
    ]

    spk_parameters = _mock_parameters(et0 = et0)
    spk_kw = (
        spice_id = -123456,
        center_id = 399,
        dt_sec = 50.0,
        write_metadata = true,
        verbose = false,
    )

    # Rejected window: no BSP, metadata or temporary file is created.
    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "out", "rejected.bsp")
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(
            sols, et0, spk_parameters;
            output_spk = output_spk, coast_windows = [(1, 1), (2, 3)], spk_kw...,
        )
        @test !isdir(dirname(output_spk)) || isempty(readdir(dirname(output_spk)))
    end

    # Failure while sampling: nothing is left in the output directory.
    mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "failing.bsp")
        @test_throws ErrorException HighFidelityEphemerisModel.ode_sol_to_spk(
            [ThrowingCoastArc([0.0, 1.0])], et0, spk_parameters;
            output_spk = output_spk, spk_kw...,
        )
        @test isempty(readdir(tmpdir))
    end
end

function test_spk_maneuver_and_metadata_helpers()
    parameters = (
        TU = 100.0,
        DU = 10.0,
        VU = 0.01,
        et0 = 1000.0,
        naif_ids = ["399", "301", "10"],
        GMs = [1.0, 0.01, 0.001],
        include_srp = false,
    )

    sols = [
        MockCoastArc(
            [0.0, 1.0],
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        MockCoastArc(
            [1.0, 2.0],
            [1.0, 2.0, 3.0, 0.2, 0.2, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]

    entries = HighFidelityEphemerisModel.collect_node_to_node_maneuvers_mps(
        sols,
        parameters.et0,
        parameters,
    )
    summary = HighFidelityEphemerisModel.summarize_maneuver_entries_mps(entries)

    @test length(entries) == 1
    @test summary["count"] == 1
    @test summary["total_delta_v_mps"] ≈ 1.0

    # Row 4 is the scalar OCP cost convention used by station-keeping.
    ocp_control = [
        0.01 0.0;
        0.0  0.02;
        0.0  0.0;
        0.01 0.02;
    ]
    ocp_times = [0.0, 1.0]

    ocp_entries = HighFidelityEphemerisModel.collect_ocp_control_maneuvers_mps(
        ocp_times,
        ocp_control,
        parameters.et0,
        parameters,
    )
    ocp_summary = HighFidelityEphemerisModel.summarize_ocp_control_entries_mps(ocp_entries)

    @test length(ocp_entries) == 2
    @test ocp_summary["total_control_scalar_mps"] ≈ 0.3
    @test ocp_summary["total_control_vector_norm_mps"] ≈ 0.3

    mktempdir() do tmpdir
        metadata = HighFidelityEphemerisModel.build_spk_metadata(
            output_spk = joinpath(tmpdir, "test.bsp"),
            metadata_json = joinpath(tmpdir, "metadata.json"),
            spice_id = -123456,
            center_id = 399,
            windows = [(start_et = 1000.0, end_et = 1100.0)],
            parameters = parameters,
            et0 = parameters.et0,
            maneuver_summary = summary,
            ocp_control_summary = ocp_summary,
            ocp_control_file = joinpath(tmpdir, "executed_maneuvers.txt"),
        )

        @test metadata["spk"]["object_id"] == -123456
        @test metadata["spk"]["center_id"] == 399
        @test metadata["coverage"]["start_et"] == 1000.0
        @test metadata["coverage"]["end_et"] == 1100.0
        @test metadata["sampling"]["segment_count"] == 1
        @test metadata["spk"]["center_name"] == "EARTH"
        @test metadata["maneuvers"]["ocp_control"]["primary_cost_key"] == "total_control_scalar_mps"
        @test metadata["maneuvers"]["ocp_control"]["total_control_scalar_mps"] ≈ 0.3
        @test haskey(metadata["maneuvers"], "trajectory_jumps")

        json_path = joinpath(tmpdir, "metadata.json")
        HighFidelityEphemerisModel.write_spk_metadata_json(json_path, metadata)
        @test isfile(json_path)
        @test occursin("HighFidelityEphemerisModel.SPKMetadata.v2", read(json_path, String))
    end
end


function test_ode_sol_to_native_spk()
    spice_id = -123456
    center_id = 399
    et0 = 1000.0
    parameters = _mock_parameters(et0 = et0)
    sols = _mock_sols()

    tmpdir_removed = mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "test_ode_sol_to_spk.bsp")

        # Treat this as the primary maneuver product, matching OCP/executed-control use.
        ocp_control = [
            0.01 0.0;
            0.0  0.02;
            0.0  0.0;
            0.01 0.02;
        ]
        ocp_times = [sols[1].t[1], sols[2].t[1]]

        result = HighFidelityEphemerisModel.ode_sol_to_spk(
            sols,
            et0,
            parameters;
            # center (naif_ids[1] = 399) and frame (J2000) default from `parameters`
            output_spk = output_spk,
            spice_id = spice_id,
            polynom_degree = 7,
            segment_id = "TEST_ODE_SOL_TO_SPK",
            dt_sec = 100.0,
            segment_gap_sec = 1e-7,
            write_maneuvers = false,
            ocp_control = ocp_control,
            ocp_control_times = ocp_times,
            write_metadata = true,
            verbose = false,
        )

        @test isfile(result.output_spk)
        @test result.maneuver_txt == joinpath(tmpdir, "test_ode_sol_to_spk_maneuvers.txt")
        @test isfile(result.maneuver_txt)
        @test result.trajectory_maneuver_txt === nothing
        @test isfile(result.metadata_json)
        @test result.segment_count == 2
        @test sort(readdir(tmpdir)) == ["test_ode_sol_to_spk.bsp", "test_ode_sol_to_spk_maneuvers.txt",
                                        "test_ode_sol_to_spk_metadata.json"]
        @test [s.segment_id for s in HighFidelityEphemerisModel.read_spk_segments(result.output_spk)] ==
              ["TEST_ODE_SOL_TO_SPK_001", "TEST_ODE_SOL_TO_SPK_002"]
        @test result.ocp_control_summary["total_control_scalar_mps"] ≈ 300.0

        maneuver_text = read(result.maneuver_txt, String)
        @test occursin("DV_scalar_u4_mps", maneuver_text)
        @test occursin("1.000000000000000e+02", maneuver_text)
        @test occursin("2.000000000000000e+02", maneuver_text)

        metadata_text = read(result.metadata_json, String)
        @test occursin("\"ocp_control\": {", metadata_text)
        @test occursin("\"primary_cost_key\": \"total_control_scalar_mps\"", metadata_text)
        @test occursin("\"trajectory_jumps\"", metadata_text)

        # Validate that the BSP can be furnished and reproduces the source arcs.
        _validate_spk_against_sols(
            result.output_spk,
            sols,
            (1.25, 6.5, 9.25, 14.75),
            parameters;
            spice_id = spice_id,
            center_id = center_id,
        )

        tmpdir
    end

    @test !isdir(tmpdir_removed)
end


function test_incremental_native_spk_append()
    spice_id = -123457
    center_id = 399
    et0 = 1000.0
    parameters = _mock_parameters(et0 = et0)
    sols = _mock_sols()

    tmpdir_removed = mktempdir() do tmpdir
        output_spk = joinpath(tmpdir, "test_append_ode_sol_to_spk.bsp")
        write(output_spk, "stale kernel from a previous run")

        # The first call replaces any existing file, the next ones append.
        results = [
            HighFidelityEphemerisModel.append_ode_sol_to_spk!(
                sol, et0, parameters;
                output_spk = output_spk,
                spice_id = spice_id,
                append = k > 1,
                segment_id = "TEST_APPEND",
                dt_sec = 100.0,
            ) for (k, sol) in enumerate(sols)
        ]

        @test isfile(output_spk)
        @test [r.segment_index for r in results] == [1, 2]
        @test [r.appended for r in results] == [false, true]
        @test results[2].output_spk == abspath(output_spk)
        @test results[2].epoch_range == (et0 + 800.0, et0 + 1600.0)
        @test results[2].point_count == 9
        @test sort(readdir(tmpdir)) == ["test_append_ode_sol_to_spk.bsp"]   # no sidecars yet

        segments = HighFidelityEphemerisModel.read_spk_segments(output_spk)
        @test [s.segment_id for s in segments] == ["TEST_APPEND_001", "TEST_APPEND_002"]
        @test all(s -> s.spice_id == spice_id && s.center_id == center_id && s.ref_frame_name == "J2000", segments)
        @test all(s -> s.spk_type == 13 && s.polynom_degree == 7 && s.point_count == 9, segments)

        # This is the key station-keeping use case: append separate arcs to the
        # same BSP and then query the combined product.
        _validate_spk_against_sols(
            output_spk,
            sols,
            (1.25, 6.5, 8.0, 9.25, 14.75, 16.0),
            parameters;
            spice_id = spice_id,
            center_id = center_id,
        )

        # Sidecars of the whole kernel, written once at the end.
        final = HighFidelityEphemerisModel.finalize_spk(
            output_spk, parameters;
            ocp_control = [0.01 0.0; 0.0 0.02; 0.0 0.0],
            ocp_control_times = [0.0, 8.0],
            verbose = false,
        )
        @test final.segment_count == 2
        @test final.epoch_ranges == [(et0, et0 + 800.0), (et0 + 800.0, et0 + 1600.0)]
        @test final.point_counts == [9, 9]
        @test final.maneuver_txt == joinpath(tmpdir, "test_append_ode_sol_to_spk_maneuvers.txt")
        @test final.metadata_json == joinpath(tmpdir, "test_append_ode_sol_to_spk_metadata.json")
        @test final.ocp_control_summary["total_control_vector_norm_mps"] ≈ 300.0
        metadata_text = read(final.metadata_json, String)
        @test occursin("\"segment_id\": \"TEST_APPEND_002\"", metadata_text)

        # Appending requires an existing kernel.
        @test_throws ErrorException HighFidelityEphemerisModel.append_ode_sol_to_spk!(
            sols[1], et0, parameters;
            output_spk = joinpath(tmpdir, "missing.bsp"), spice_id = spice_id, append = true,
        )
        return tmpdir
    end

    @test !isdir(tmpdir_removed)
end

test_spk_helper_utilities()
test_spk_state_sampling()
test_spk_rejects_discontinuous_coast_windows()
test_spk_coast_window_continuity_tolerances()
test_spk_coast_window_nonfinite_and_callback_messages()
test_spk_rejected_window_leaves_no_output()
test_spk_maneuver_and_metadata_helpers()
test_ode_sol_to_native_spk()
test_incremental_native_spk_append()
