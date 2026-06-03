using Test
using SPICE

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end

struct MockCoastArc
    t::Vector{Float64}
    x0::Vector{Float64}
    xdot::Vector{Float64}
end

(arc::MockCoastArc)(t) = arc.x0 .+ arc.xdot .* t

function _mkspk_available(cmd::AbstractString)
    return isfile(cmd) || Sys.which(cmd) !== nothing
end

function _find_test_lsk()
    for key in ("LSK_PATH", "LEAPSECONDS_FILE")
        path = get(ENV, key, "")
        if !isempty(path) && isfile(path)
            return path
        end
    end

    spice_dir = get(ENV, "SPICE", "")
    if !isempty(spice_dir)
        candidates = [
            joinpath(spice_dir, "lsk", "naif0012.tls"),
            joinpath(spice_dir, "kernels", "lsk", "naif0012.tls"),
            joinpath(spice_dir, "naif0012.tls"),
        ]
        for path in candidates
            isfile(path) && return path
        end
    end

    return nothing
end

@testset "SPK helper utilities" begin
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

@testset "SPK state and setup file writers" begin
    parameters = (
        TU = 100.0,   # seconds per nondimensional time unit
        DU = 10.0,    # km per nondimensional distance unit
        VU = 0.01,    # km/s per nondimensional velocity unit
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
        setup_dir = joinpath(tmpdir, "setup")

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

        setup_files = HighFidelityEphemerisModel.write_full_setups_for_state_files_exact!(
            state_result.state_files;
            outdir = setup_dir,
            output_spk_type = 13,
            object_id = -123456,
            center_id = 399,
            ref_frame_name = "J2000",
            producer_id = "HighFidelityEphemerisModel.jl",
            leapseconds_file = "naif0012.tls",
            polynom_degree = 7,
            segment_id = "TEST_SEGMENT",
            verbose = false,
            show_progress = false,
        )

        @test length(setup_files) == 1
        @test isfile(setup_files[1])

        setup_text = read(setup_files[1], String)

        @test occursin("INPUT_DATA_TYPE", setup_text)
        @test occursin("OUTPUT_SPK_TYPE   = 13", setup_text)
        @test occursin("OBJECT_ID", setup_text)
        @test occursin("-123456", setup_text)
        @test occursin("CENTER_ID", setup_text)
        @test occursin("399", setup_text)
        @test occursin("REF_FRAME_NAME", setup_text)
        @test occursin("J2000", setup_text)
        @test occursin("SEGMENT_ID", setup_text)
        @test occursin("TEST_SEGMENT", setup_text)
        @test occursin("EARLIEST_EPOCH", setup_text)
        @test occursin("LATEST_EPOCH", setup_text)

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

        @test_throws ErrorException HighFidelityEphemerisModel.write_segmented_states_for_spk!(
            jumping_sols,
            [(1, 2)],
            et0,
            parameters;
            dt_sec = 50.0,
            segment_gap_sec = 0.0,
            outdir = joinpath(tmpdir, "jumping_states"),
            verbose = false,
            show_progress = false,
        )

        existing_spk = joinpath(tmpdir, "existing.bsp")
        write(existing_spk, "previous kernel")

        @test_throws Exception HighFidelityEphemerisModel.ode_sol_to_spk(
            sols,
            et0,
            parameters;
            output_spk = existing_spk,
            spice_id = -123456,
            center_id = 399,
            leapseconds_file = "missing.tls",
            mkspk_cmd = joinpath(tmpdir, "missing_mkspk"),
            dt_sec = 50.0,
            segment_gap_sec = 0.0,
            write_maneuvers = false,
            write_metadata = false,
            verbose = false,
            show_progress = false,
        )
        @test read(existing_spk, String) == "previous kernel"
    end
end

@testset "SPK maneuver and metadata helpers" begin
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

    mktempdir() do tmpdir
        metadata = HighFidelityEphemerisModel.build_spk_metadata(
            output_spk = joinpath(tmpdir, "test.bsp"),
            maneuver_file = joinpath(tmpdir, "maneuvers.txt"),
            metadata_json = joinpath(tmpdir, "metadata.json"),
            spice_id = -123456,
            center_id = 399,
            epoch_ranges = [(1000.0, 1100.0)],
            parameters = parameters,
            et0 = parameters.et0,
            maneuver_summary = summary,
            ocp_control_summary = ocp_summary,
        )

        @test metadata["spk"]["object_id"] == -123456
        @test metadata["spk"]["center_id"] == 399
        @test metadata["coverage"]["start_et"] == 1000.0
        @test metadata["coverage"]["end_et"] == 1100.0
        @test metadata["sampling"]["segment_count"] == 1
        @test metadata["maneuvers"]["total_delta_v_mps"] ≈ 1.0
        @test haskey(metadata["maneuvers"], "ocp_control")

        json_path = joinpath(tmpdir, "metadata.json")
        HighFidelityEphemerisModel.write_spk_metadata_json(json_path, metadata)
        @test isfile(json_path)
        @test occursin("HighFidelityEphemerisModel.SPKMetadata.v1", read(json_path, String))
    end
end

@testset "ODE solution to SPK end-to-end" begin
    mkspk_cmd = get(ENV, "MKSPK_CMD", "mkspk")
    lsk_path = _find_test_lsk()

    if !_mkspk_available(mkspk_cmd) || lsk_path === nothing
        @info "Skipping end-to-end SPK test because mkspk or the leapseconds kernel is unavailable." mkspk_cmd lsk_path
        @test_skip _mkspk_available(mkspk_cmd) && lsk_path !== nothing
    else
        spice_id = -123456
        center_id = 399
        et0 = 1000.0

        parameters = (
            TU = 100.0,
            DU = 1000.0,
            VU = 10.0,
            et0 = et0,
            naif_ids = ["399"],
            GMs = [1.0],
            include_srp = false,
        )

        r0 = [1.0, 2.0, 3.0]
        v0 = [0.01, -0.02, 0.03]
        x0 = vcat(r0, v0)
        xdot = vcat(v0, zeros(3))

        sols = [
            MockCoastArc([0.0, 8.0], x0, xdot),
            MockCoastArc([8.0, 16.0], x0, xdot),
        ]

        tmpdir_removed = mktempdir() do tmpdir
            output_spk = joinpath(tmpdir, "test_ode_sol_to_spk.bsp")

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
                leapseconds_file = lsk_path,
                mkspk_cmd = mkspk_cmd,
                dt_sec = 100.0,
                segment_gap_sec = 1e-7,
                keep_intermediates = false,
                write_maneuvers = true,
                write_metadata = true,
                verbose = false,
                show_progress = false,
            )

            @test isfile(result.output_spk)
            @test isfile(result.maneuver_txt)
            @test isfile(result.metadata_json)
            @test result.segment_count == 2
            @test result.intermediate_dir === nothing

            SPICE.furnsh(result.output_spk)
            try
                for t_nd in (1.25, 6.5, 9.25, 14.75)
                    et = et0 + t_nd * parameters.TU
                    state_spk, _ = SPICE.spkezr(
                        string(spice_id),
                        et,
                        "J2000",
                        "NONE",
                        string(center_id),
                    )

                    x_ref = sols[t_nd <= 8.0 ? 1 : 2](t_nd)
                    state_ref = vcat(
                        x_ref[1:3] .* parameters.DU,
                        x_ref[4:6] .* parameters.VU,
                    )

                    @test state_spk[1:3] ≈ state_ref[1:3] atol = 1e-7
                    @test state_spk[4:6] ≈ state_ref[4:6] atol = 1e-10
                end
            finally
                SPICE.unload(result.output_spk)
            end

            tmpdir
        end

        @test !isdir(tmpdir_removed)
    end
end
