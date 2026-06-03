"""Incremental SPK writing helpers"""


function _check_spk_output_target(output_spk::AbstractString; overwrite::Bool = true)
    output_spk_abs = abspath(output_spk)
    splitext(output_spk_abs)[2] == ".bsp" || error("`output_spk` must end in `.bsp`: $output_spk_abs")
    mkpath(dirname(output_spk_abs))

    if isfile(output_spk_abs) && !overwrite
        error("Output SPK already exists and `overwrite=false`: $output_spk_abs")
    end

    return output_spk_abs
end

"""
    prepare_spk_output!(output_spk; overwrite=true)

Prepare a final `.bsp` file path for a new SPK build. This removes an existing
file only when `overwrite=true`, creates the parent folder, and returns the
absolute output path.

This is useful for Monte-Carlo station-keeping runs where each seed writes its
own kernel before the recursion loop starts.

# Arguments
- `output_spk`: output `.bsp` file path
- `overwrite::Bool`: if true, remove an existing file before writing
"""
function prepare_spk_output!(output_spk::AbstractString; overwrite::Bool = true)
    output_spk_abs = _check_spk_output_target(output_spk; overwrite = overwrite)

    if isfile(output_spk_abs)
        try
            rm(output_spk_abs; force = true)
        catch err
            error("Could not remove existing SPK: $output_spk_abs. If it is furnished/loaded in SPICE or open elsewhere, close/unload it first. Original error: $err")
        end
    end

    return output_spk_abs
end

"""
    write_solution_segment_states_for_spk!(sol, et0, parameters; kwargs...)

Write one MKSPK `STATES` file from one ODE solution segment. The solution
is assumed to use nondimensional time and nondimensional states, consistent
with `ode_sol_to_spk`.

# Arguments
- `sol`: ODE solution segment
- `et0`: reference epoch in seconds past J2000
- `parameters`: object containing `TU`, `DU`, and `VU`
"""
function write_solution_segment_states_for_spk!(
    sol,
    et0,
    parameters;
    segment_index::Integer,
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 0.0,
    outdir::AbstractString = "states_segmented",
)
    dt = Float64(dt_sec)
    gap = Float64(segment_gap_sec)
    dt > 0 || error("`dt_sec` must be positive.")
    gap >= 0 || error("`segment_gap_sec` must be nonnegative.")

    isdir(outdir) || mkpath(outdir)

    et_start = Float64(et0 + sol.t[1] * parameters.TU)
    et_end_true = Float64(et0 + sol.t[end] * parameters.TU)
    et_end = et_end_true - gap
    et_end > et_start || error("Bad segment span after applying segment_gap_sec=$(gap).")

    ts_et = build_segment_epochs(et_start, et_end; dt_sec = dt)

    cols = Vector{Vector{Float64}}()
    for et in ts_et
        t_nd = (et - et0) / parameters.TU
        x_nd = sol(t_nd)
        length(x_nd) >= 6 || error("Expected a state with at least 6 components, got length $(length(x_nd)).")

        # MKSPK expects dimensional states.
        r_km = x_nd[1:3] .* parameters.DU
        v_kmps = x_nd[4:6] .* parameters.VU
        push!(cols, vcat(Float64.(r_km), Float64.(v_kmps)))
    end

    length(ts_et) > 1 || error("Segment $(segment_index) produced <2 points. Reduce dt_sec or check segment duration.")

    Y = reduce(hcat, cols)
    tag = lpad(string(segment_index), 3, '0')
    outpath = joinpath(outdir, "seg_$(tag)_states.txt")
    write_mkspk_states_file(outpath, ts_et, Y)

    return (
        state_file = outpath,
        epoch_range = (ts_et[1], ts_et[end]),
        point_count = length(ts_et),
    )
end

"""
    append_state_file_to_spk!(state_file; output_spk, segment_index, kwargs...)

Write the corresponding MKSPK setup file for `state_file` and create/append it
into `output_spk`. If `append=nothing`, the function appends when `output_spk`
already exists and creates it otherwise.
"""
function append_state_file_to_spk!(
    state_file::AbstractString;
    output_spk::AbstractString,
    segment_index::Integer,
    setup_dir::AbstractString = "setup_segmented",
    append::Union{Nothing,Bool} = nothing,
    mkspk_cmd::AbstractString = "mkspk",
    spice_id::Union{Nothing,Integer} = nothing,
    object_name::Union{Nothing,AbstractString} = nothing,
    center_id::Union{Nothing,Integer} = nothing,
    center_name::Union{Nothing,AbstractString} = nothing,
    ref_frame_name::AbstractString = "J2000",
    producer_id::AbstractString = "HighFidelityEphemerisModel.jl",
    output_spk_type::Integer = 13,
    polynom_degree::Integer = 7,
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    leapseconds_file::AbstractString = "naif0012.tls",
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    verbose::Bool = false,
    suppress_mkspk_output::Bool = true,
)
    isfile(state_file) || error("Missing state file: $state_file")
    isdir(setup_dir) || mkpath(setup_dir)

    output_spk_abs = abspath(output_spk)
    tag = lpad(string(segment_index), 3, '0')
    setup_path = joinpath(setup_dir, "seg_$(tag)_setup.txt")
    segid = segment_id_per_seg ? "$(segment_id)_$(tag)" : String(segment_id)

    write_full_mkspk_setup_exact(
        setup_path;
        segment_id = segid,
        states_file_for_epochs = state_file,
        output_spk_type = output_spk_type,
        object_id = spice_id,
        object_name = object_name,
        center_id = center_id,
        center_name = center_name,
        ref_frame_name = ref_frame_name,
        producer_id = producer_id,
        data_delimiter = ",",
        lines_per_record = 1,
        time_wrapper = "# ETSECONDS",
        ignore_first_line = 1,
        leapseconds_file = leapseconds_file,
        frame_def_file = frame_def_file,
        polynom_degree = polynom_degree,
    )

    # Default behavior: create the BSP if absent, append if it already exists.
    append_flag = append === nothing ? isfile(output_spk_abs) : Bool(append)

    wrap_mkspk(
        setup_path,
        state_file,
        output_spk_abs;
        mkspk_cmd = mkspk_cmd,
        append = append_flag,
        overwrite = false,
        verbose = verbose,
        suppress_output = suppress_mkspk_output,
    )

    return (
        output_spk = output_spk_abs,
        state_file = state_file,
        setup_file = setup_path,
        appended = append_flag,
        epoch_range = _epoch_range_from_states_file(state_file),
    )
end

"""
    append_solution_segment_to_spk!(sol, et0, parameters; output_spk, segment_index, kwargs...)

One-call helper for station-keeping recursion. It writes one `STATES` file from
`sol`, writes one MKSPK setup file, and creates/appends that segment into the
run-specific SPK kernel.

Recommended Monte-Carlo pattern:

```julia
output_spk = prepare_spk_output!("seed_001_recurse.bsp")

for idx_start in 1:N_recurse
    sol_recurse = solve(...)
    append_solution_segment_to_spk!(
        sol_recurse,
        et0,
        parameters;
        output_spk = output_spk,
        segment_index = idx_start,
        append = idx_start > 1,
    )
end
```
"""
function append_solution_segment_to_spk!(
    sol,
    et0,
    parameters;
    output_spk::AbstractString,
    segment_index::Integer,
    states_dir::AbstractString = "states_segmented",
    setup_dir::AbstractString = "setup_segmented",
    dt_sec::Real = 1800.0,
    segment_gap_sec::Real = 0.0,
    kwargs...,
)
    state_result = write_solution_segment_states_for_spk!(
        sol,
        et0,
        parameters;
        segment_index = segment_index,
        dt_sec = dt_sec,
        segment_gap_sec = segment_gap_sec,
        outdir = states_dir,
    )

    append_result = append_state_file_to_spk!(
        state_result.state_file;
        output_spk = output_spk,
        segment_index = segment_index,
        setup_dir = setup_dir,
        kwargs...,
    )

    return merge(state_result, append_result)
end

function append_solution_segment_to_spk!(sol, parameters; kwargs...)
    et0 = _get_et0_from_parameters(parameters)
    et0 === nothing && error("No `et0` argument was provided and `parameters` has no recognized ET0 property. Tried: et0, ET0, epoch0, et_start, t0_et.")
    return append_solution_segment_to_spk!(sol, Float64(et0), parameters; kwargs...)
end
