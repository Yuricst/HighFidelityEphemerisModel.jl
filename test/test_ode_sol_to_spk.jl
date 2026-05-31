using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end

struct MockCoastArc
    t::Vector{Float64}
    x0::Vector{Float64}
    xdot::Vector{Float64}
end

(arc::MockCoastArc)(t) = arc.x0 .+ arc.xdot .* t

@testset "SPK helper utilities" begin
    # -------------------------------------------------------------------------
    # Epoch grid helper
    # -------------------------------------------------------------------------
    ts = HighFidelityEphemerisModel.build_segment_epochs(
        0.0,
        3600.0;
        dt_sec = 1800.0,
    )

    @test ts == [0.0, 1800.0, 3600.0]

    # -------------------------------------------------------------------------
    # Default segmentation helper
    # -------------------------------------------------------------------------
    windows = HighFidelityEphemerisModel.default_coast_windows([:a, :b, :c])
    @test windows == [(1, 1), (2, 2), (3, 3)]

    # -------------------------------------------------------------------------
    # Output path preparation
    # -------------------------------------------------------------------------
    out = tempname() * ".bsp"
    prepared = HighFidelityEphemerisModel.prepare_spk_output!(out)

    @test endswith(prepared, ".bsp")
    @test isdir(dirname(prepared))
    @test !isfile(prepared)

    @test_throws ErrorException HighFidelityEphemerisModel.prepare_spk_output!(tempname() * ".txt")
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
    states_dir = mktempdir()
    setup_dir = mktempdir()

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

    # First sample: t_nd = 0.0
    @test first_row[1] ≈ 1000.0
    @test first_row[2:4] ≈ [10.0, 20.0, 30.0]       # position scaled by DU
    @test first_row[5:7] ≈ [0.04, 0.05, 0.06]       # velocity scaled by VU

    # Last sample: t_nd = 1.0
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
end