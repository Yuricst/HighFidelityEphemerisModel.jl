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

    tempdir_removed = mktempdir() do tmpdir
        out = joinpath(tmpdir, "test.bsp")
        prepared = HighFidelityEphemerisModel.prepare_spk_output!(out)

        @test endswith(prepared, ".bsp")
        @test isdir(dirname(prepared))
        @test !isfile(prepared)
        @test_throws ErrorException HighFidelityEphemerisModel.prepare_spk_output!(joinpath(tmpdir, "test.txt"))

        tmpdir
    end

    @test !isdir(tempdir_removed)
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

    mktempdir() do tmpdir
        states_dir = joinpath(tmpdir, "states")

        state_result = HighFidelityEphemerisModel.write_segmented_states_for_spk!(
            sols,
            [(1, 1)],
            et0,
            parameters;
            dt_sec = 50.0,
            segment_gap_sec = 0.0,
            outdir = states_dir,
            verbose = false,
            show_progress = false,
        )

        @test length(state_result.state_files) == 1
        @test length(state_result.epoch_ranges) == 1
        @test state_result.epoch_ranges[1] == (1000.0, 1100.0)

        state_file = state_result.state_files[1]
        @test isfile(state_file)

        lines = readlines(state_file)
        @test lines[1] == "# ETSECONDS"
        @test length(lines) == 4

        first_row = parse.(Float64, split(lines[2], ","))
        last_row = parse.(Float64, split(lines[end], ","))

        @test first_row[1] ≈ 1000.0
        @test first_row[2:4] ≈ [10.0, 20.0, 30.0]
        @test first_row[5:7] ≈ [0.04, 0.05, 0.06]

        @test last_row[1] ≈ 1100.0
        @test last_row[2:4] ≈ [11.0, 22.0, 33.0]
        @test last_row[5:7] ≈ [0.044, 0.055, 0.066]

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

        single_segment = HighFidelityEphemerisModel.write_solution_segment_states_for_spk!(
            sols[1],
            et0,
            parameters;
            segment_index = 1,
            dt_sec = 50.0,
            outdir = joinpath(tmpdir, "incremental_states"),
        )

        @test isfile(single_segment.state_file)
        @test single_segment.epoch_range == (1000.0, 1100.0)
        @test single_segment.point_count == 3
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
            maneuver_file = nothing,
            metadata_json = joinpath(tmpdir, "metadata.json"),
            spice_id = -123456,
            center_id = 399,
            epoch_ranges = [(1000.0, 1100.0)],
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
        @test metadata["maneuvers"]["is_primary"] == true
        @test metadata["maneuvers"]["type"] == "ocp_control"
        @test metadata["maneuvers"]["primary_cost_key"] == "total_control_scalar_mps"
        @test metadata["maneuvers"]["total_control_scalar_mps"] ≈ 0.3
        @test haskey(metadata["maneuvers"], "trajectory_jumps")

        json_path = joinpath(tmpdir, "metadata.json")
        HighFidelityEphemerisModel.write_spk_metadata_json(json_path, metadata)
        @test isfile(json_path)
        @test occursin("HighFidelityEphemerisModel.SPKMetadata.v1", read(json_path, String))
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
            output_spk = output_spk,
            spice_id = spice_id,
            center_id = center_id,
            ref_frame_name = "J2000",
            output_spk_type = 13,
            polynom_degree = 7,
            segment_id = "TEST_ODE_SOL_TO_SPK",
            dt_sec = 100.0,
            segment_gap_sec = 1e-7,
            keep_intermediates = false,
            write_maneuvers = false,
            ocp_control = ocp_control,
            ocp_control_times = ocp_times,
            write_metadata = true,
            verbose = false,
            show_progress = false,
        )

        @test isfile(result.output_spk)
        @test isfile(result.maneuver_txt)
        @test isfile(result.ocp_maneuver_txt)
        @test result.maneuver_txt == result.ocp_maneuver_txt
        @test result.trajectory_maneuver_txt === nothing
        @test isfile(result.metadata_json)
        @test result.segment_count == 2
        @test result.intermediate_dir === nothing
        @test isempty(result.state_files)
        @test isempty(result.setup_files)
        @test result.ocp_control_summary["total_control_scalar_mps"] ≈ 300.0

        maneuver_text = read(result.maneuver_txt, String)
        @test occursin("DV_scalar_u4_mps", maneuver_text)
        @test occursin("1.000000000000000e+02", maneuver_text)
        @test occursin("2.000000000000000e+02", maneuver_text)

        metadata_text = read(result.metadata_json, String)
        @test occursin("\"type\": \"ocp_control\"", metadata_text)
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

        result1 = HighFidelityEphemerisModel.ode_sol_to_spk(
            sols[1],
            et0,
            parameters;
            output_spk = output_spk,
            spice_id = spice_id,
            center_id = center_id,
            ref_frame_name = "J2000",
            output_spk_type = 13,
            polynom_degree = 7,
            segment_id = "TEST_APPEND",
            segment_id_per_seg = true,
            append = false,
            overwrite = true,
            segment_index = 1,
            dt_sec = 100.0,
            segment_gap_sec = 1e-7,
            write_maneuvers = false,
            write_ocp_maneuvers = false,
            write_metadata = false,
            verbose = false,
            show_progress = false,
        )

        result2 = HighFidelityEphemerisModel.ode_sol_to_spk(
            sols[2],
            et0,
            parameters;
            output_spk = output_spk,
            spice_id = spice_id,
            center_id = center_id,
            ref_frame_name = "J2000",
            output_spk_type = 13,
            polynom_degree = 7,
            segment_id = "TEST_APPEND",
            segment_id_per_seg = true,
            append = true,
            segment_index = 2,
            dt_sec = 100.0,
            segment_gap_sec = 0.0,
            write_maneuvers = false,
            write_ocp_maneuvers = false,
            write_metadata = false,
            verbose = false,
            show_progress = false,
        )

        @test isfile(output_spk)
        @test result1.segment_count == 1
        @test result1.segment_index == 1
        @test result1.appended == false
        @test result2.segment_count == 1
        @test result2.segment_index == 2
        @test result2.appended == true
        @test result2.output_spk == abspath(output_spk)

        # This is the key station-keeping use case: append separate arcs to the
        # same BSP and then query the combined product.
        _validate_spk_against_sols(
            output_spk,
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


test_spk_helper_utilities()
test_spk_state_sampling()
test_spk_maneuver_and_metadata_helpers()
test_ode_sol_to_native_spk()
test_incremental_native_spk_append()
