"""MKSPK setup generation and execution helpers"""


"""
    _epoch_range_from_states_file(states_file)

Read the first and last epochs from an MKSPK `STATES` file.
"""
function _epoch_range_from_states_file(states_file::AbstractString)
    first_epoch = nothing
    last_epoch = nothing

    open(states_file, "r") do io
        first_line_skipped = false
        for line in eachline(io)
            if !first_line_skipped
                first_line_skipped = true
                continue
            end

            s = strip(line)
            isempty(s) && continue

            idx = findfirst(==(','), s)
            idx === nothing && continue

            epoch = parse(Float64, s[1:idx-1])
            if first_epoch === nothing
                first_epoch = epoch
            end
            last_epoch = epoch
        end
    end

    @assert first_epoch !== nothing && last_epoch !== nothing "No epochs found in $states_file"
    return Float64(first_epoch), Float64(last_epoch)
end

"""
    write_full_mkspk_setup_exact(setup_path; states_file_for_epochs, ...)

Write a full MKSPK setup file with exact `EARLIEST_EPOCH` and `LATEST_EPOCH`
computed directly from the corresponding segment states file.

# Arguments
- `setup_path::AbstractString`: output setup file path
- `segment_id::String`: SPK segment identifier
- `states_file_for_epochs::AbstractString`: MKSPK `STATES` file used for coverage
"""
function write_full_mkspk_setup_exact(
    setup_path::AbstractString;
    segment_id::String,
    states_file_for_epochs::AbstractString,
    output_spk_type::Integer = 13,
    object_id::Union{Nothing,Integer} = nothing,
    object_name::Union{Nothing,AbstractString} = nothing,
    center_id::Union{Nothing,Integer} = nothing,
    center_name::Union{Nothing,AbstractString} = nothing,
    ref_frame_name::AbstractString = "J2000",
    producer_id::AbstractString = "HighFidelityEphemerisModel.jl",
    data_delimiter::AbstractString = ",",
    lines_per_record::Integer = 1,
    time_wrapper::AbstractString = "# ETSECONDS",
    ignore_first_line::Integer = 1,
    leapseconds_file::AbstractString = "naif0012.tls",
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    polynom_degree::Integer = 7,
)
    earliest, latest = _epoch_range_from_states_file(states_file_for_epochs)

    open(setup_path, "w") do io
        @printf(io, "\\begindata\n")
        @printf(io, "   INPUT_DATA_TYPE   = 'STATES'\n")
        @printf(io, "   OUTPUT_SPK_TYPE   = %d\n", output_spk_type)

        if object_id !== nothing
            @printf(io, "   OBJECT_ID         = %d\n", object_id)
        elseif object_name !== nothing
            @printf(io, "   OBJECT_NAME       = '%s'\n", object_name)
        else
            error("Either object_id or object_name must be provided.")
        end

        if center_id !== nothing
            @printf(io, "   CENTER_ID         = %d\n", center_id)
        elseif center_name !== nothing
            @printf(io, "   CENTER_NAME       = '%s'\n", center_name)
        else
            error("Either center_id or center_name must be provided.")
        end

        @printf(io, "   REF_FRAME_NAME    = '%s'\n", ref_frame_name)
        @printf(io, "   PRODUCER_ID       = '%s'\n", producer_id)
        @printf(io, "   DATA_ORDER        = 'EPOCH X Y Z VX VY VZ'\n")
        @printf(io, "   INPUT_DATA_UNITS  = ('ANGLES=DEGREES' 'DISTANCES=km')\n")
        @printf(io, "   DATA_DELIMITER    = '%s'\n", data_delimiter)
        @printf(io, "   LINES_PER_RECORD  = %d\n", lines_per_record)
        @printf(io, "   TIME_WRAPPER      = '%s'\n", time_wrapper)
        @printf(io, "   IGNORE_FIRST_LINE = %d\n", ignore_first_line)
        @printf(io, "   LEAPSECONDS_FILE  = '%s'\n", _mkspk_path(leapseconds_file))

        if frame_def_file !== nothing
            @printf(io, "   FRAME_DEF_FILE    = '%s'\n", _mkspk_path(frame_def_file))
        end

        @printf(io, "   POLYNOM_DEGREE    = %d\n", polynom_degree)
        @printf(io, "   SEGMENT_ID        = '%s'\n", segment_id)
        @printf(io, "   EARLIEST_EPOCH    = %.15f\n", earliest)
        @printf(io, "   LATEST_EPOCH      = %.15f\n", latest)
        @printf(io, "\\begintext\n")
    end

    return setup_path
end

"""
    write_full_setups_for_state_files_exact!(state_files; kwargs...) -> Vector{String}

Write setup files for an already known vector of states files.
"""
function write_full_setups_for_state_files_exact!(
    state_files::Vector{String};
    outdir::AbstractString = "setup_segmented",
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
)
    isdir(outdir) || mkpath(outdir)
    progress_enabled = verbose && show_progress

    setup_files = String[]
    for (s, states_path) in enumerate(state_files)
        tag = lpad(string(s), 3, '0')
        setup_path = joinpath(outdir, "seg_$(tag)_setup.txt")
        @assert isfile(states_path) "Missing states file: $states_path"

        segid = segment_id_per_seg ? "$(segment_id)_$(tag)" : String(segment_id)
        write_full_mkspk_setup_exact(
            setup_path;
            segment_id = segid,
            states_file_for_epochs = states_path,
            kwargs...,
        )

        push!(setup_files, setup_path)
        _print_progress("writing setup files", s, length(state_files); enabled = progress_enabled)
    end

    return setup_files
end

"""
    write_full_setups_for_segments_exact!(nseg; ...)

Backward-compatible wrapper for the older manual-count workflow. New code should
prefer `write_full_setups_for_state_files_exact!` or `ode_sol_to_spk`.
"""
function write_full_setups_for_segments_exact!(
    nseg::Int;
    outdir::String = "setup_segmented",
    state_dir::String = "states_segmented",
    segment_id::String = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    kwargs...,
)
    state_files = [joinpath(state_dir, "seg_$(lpad(string(s), 3, '0'))_states.txt") for s in 1:nseg]
    write_full_setups_for_state_files_exact!(
        state_files;
        outdir = outdir,
        segment_id = segment_id,
        segment_id_per_seg = segment_id_per_seg,
        kwargs...,
    )
    return outdir
end

"""
    wrap_mkspk(filepath_set, filepath_in, filepath_out; mkspk_cmd="mkspk", append=false, overwrite=false)

Non-interactive wrapper around the NAIF `mkspk` executable. It never prompts
the user.

# Arguments
- `filepath_set`: MKSPK setup file path
- `filepath_in`: MKSPK input states file path
- `filepath_out`: output BSP file path
"""
function wrap_mkspk(
    filepath_set,
    filepath_in,
    filepath_out;
    mkspk_cmd = "mkspk",
    append::Bool = false,
    overwrite::Bool = false,
    verbose::Bool = true,
    suppress_output::Bool = true,
)
    @assert splitext(filepath_out)[2] == ".bsp"

    if !append && isfile(filepath_out)
        if overwrite
            rm(filepath_out; force = true)
        else
            error("Output BSP already exists: $filepath_out. Use overwrite=true or remove it first.")
        end
    end

    cmd = if append
        @assert isfile(filepath_out) "Cannot append: output BSP does not exist: $filepath_out"
        `$mkspk_cmd -append -setup $filepath_set -input $filepath_in -output $filepath_out`
    else
        `$mkspk_cmd -setup $filepath_set -input $filepath_in -output $filepath_out`
    end

    if suppress_output
        run(pipeline(cmd; stdout = devnull, stderr = devnull))
    else
        run(cmd)
    end

    if isfile(filepath_out)
        if append
            verbose && println("Successfully appended to $(filepath_out)!")
        else
            verbose && println("Successfully generated $(filepath_out)!")
        end
    else
        error("mkspk did not create expected output: $filepath_out")
    end

    return filepath_out
end

"""
    run_mkspk_for_segments!(setup_files, state_files, output_spk; kwargs...)

Run MKSPK sequentially for one setup/state-file pair per segment. The first
segment creates `output_spk`, and later segments append to the same BSP.
"""
function run_mkspk_for_segments!(
    setup_files::Vector{String},
    state_files::Vector{String},
    output_spk::AbstractString;
    mkspk_cmd::AbstractString = "mkspk",
    verbose::Bool = true,
    show_progress::Bool = true,
    suppress_mkspk_output::Bool = true,
)
    length(setup_files) == length(state_files) || error("setup_files and state_files must have the same length.")
    length(setup_files) > 0 || error("No setup/state files provided.")
    progress_enabled = verbose && show_progress

    for s in eachindex(setup_files)
        setup_path = setup_files[s]
        state_path = state_files[s]
        @assert isfile(setup_path) "Missing setup file: $setup_path"
        @assert isfile(state_path) "Missing states file: $state_path"

        wrap_mkspk(
            setup_path,
            state_path,
            output_spk;
            mkspk_cmd = mkspk_cmd,
            append = s > 1,
            overwrite = false,
            verbose = false,
            suppress_output = suppress_mkspk_output,
        )
        _print_progress("running mkspk", s, length(setup_files); enabled = progress_enabled)
    end

    verbose && !show_progress && println("SPK complete: ", _display_path(output_spk))
    return output_spk
end
