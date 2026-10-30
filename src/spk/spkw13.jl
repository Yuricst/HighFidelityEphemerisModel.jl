"""Native SPICE type-13 SPK writers"""


"""
    _resolve_spice_integer_id(id_value, name_value; label)

Resolve either an integer SPICE ID or a registered SPICE object name to the
integer code required by `spkw13`.

The public API accepts both IDs and names, but the low-level SPICE writer always
needs integer body and center codes.
"""
function _resolve_spice_integer_id(id_value, name_value; label::AbstractString)
    if id_value !== nothing
        return Int(id_value)
    elseif name_value !== nothing
        code = SPICE.bods2c(String(name_value))
        code === nothing && error("Could not resolve $(label) name to a SPICE ID: $(name_value). Pass an integer ID instead.")
        return Int(code)
    else
        error("Either `$(label)_id` or `$(label)_name` must be provided.")
    end
end

"""
    _spkw13_segment_id(segment_id, segment_index; segment_id_per_seg=false)

Build a valid SPK segment identifier. NAIF segment identifiers are limited to
40 printable characters, so this helper checks the limit before calling SPICE.
"""
function _spkw13_segment_id(segment_id::AbstractString, segment_index::Integer; segment_id_per_seg::Bool = false)
    tag = lpad(string(segment_index), 3, '0')
    segid = segment_id_per_seg ? "$(segment_id)_$(tag)" : String(segment_id)
    length(segid) <= 40 || error("SPK segment ID is too long ($(length(segid)) characters). Type-13 segment IDs must be at most 40 characters: $(segid)")
    all(c -> isprint(c), segid) || error("SPK segment ID contains non-printable characters: $(segid)")
    isempty(strip(segid)) && error("SPK segment ID must not be empty.")
    return segid
end

"""
    _validate_spkw13_segment(epochs, states; degree, first, last, segid)

Check the requirements that are easiest to diagnose before calling `spkw13`.
Failing here gives a Julia error with segment context instead of a lower-level
SPICE error.
"""
function _validate_spkw13_segment(epochs, states; degree::Integer, first::Real, last::Real, segid::AbstractString)
    n = length(epochs)
    n == length(states) || error("epochs and states must have the same length. Got $(n) epochs and $(length(states)) states.")
    n > 0 || error("A type-13 SPK segment needs at least one state.")

    degree >= 1 || error("Type-13 SPK polynomial degree must be at least 1.")
    isodd(degree) || error("Type-13 SPK polynomial degree must be odd. Got degree=$(degree).")
    n >= div(degree + 1, 2) || error("Type-13 SPK degree $(degree) requires at least $((degree + 1) ÷ 2) states, but segment $(segid) has only $(n).")

    first < last || error("Bad SPK segment descriptor times for $(segid): first=$(first), last=$(last). Expected first < last.")
    all(diff(Float64.(epochs)) .> 0.0) || error("SPK epochs for $(segid) must be strictly increasing.")
    first < Float64(epochs[1]) && error("First state epoch for $(segid) is later than segment start: epoch[1]=$(epochs[1]), first=$(first).")
    last > Float64(epochs[end]) && error("Last state epoch for $(segid) is earlier than segment end: epoch[end]=$(epochs[end]), last=$(last).")

    length(segid) <= 40 || error("SPK segment ID is too long: $(segid)")
    isempty(strip(segid)) && error("SPK segment ID must not be empty.")

    for (idx, state) in enumerate(states)
        length(state) == 6 || error("State $(idx) in segment $(segid) has length $(length(state)); expected length 6.")
        all(isfinite, state) || error("State $(idx) in segment $(segid) contains non-finite values.")
    end

    return nothing
end

"""
    _prepare_spk_final_path(output_spk; overwrite=true)

Normalize and validate the final `.bsp` path for a full new-kernel write.
"""
function _prepare_spk_final_path(output_spk::AbstractString; overwrite::Bool = true)
    output_spk_abs = abspath(output_spk)
    splitext(output_spk_abs)[2] == ".bsp" || error("`output_spk` must end in `.bsp`: $(output_spk_abs)")
    mkpath(dirname(output_spk_abs))

    if isfile(output_spk_abs) && !overwrite
        error("Output SPK already exists and `overwrite=false`: $(output_spk_abs)")
    end

    return output_spk_abs
end

"""
    _safe_install_spk!(tmp_spk, output_spk; overwrite=true)

Move a fully-written temporary BSP into the requested output path. The previous
file is moved aside first so a failed replacement is less likely to destroy an
existing product.
"""
function _safe_install_spk!(tmp_spk::AbstractString, output_spk::AbstractString; overwrite::Bool = true)
    isfile(tmp_spk) || error("Temporary SPK was not created: $(tmp_spk)")

    if !isfile(output_spk)
        mv(tmp_spk, output_spk; force = false)
        return output_spk
    end

    overwrite || error("Output SPK already exists and `overwrite=false`: $(output_spk)")

    backup_spk = joinpath(dirname(output_spk), ".$(basename(output_spk)).backup_$(replace(string(time_ns()), '-' => '_'))")
    moved_old = false

    try
        mv(output_spk, backup_spk; force = false)
        moved_old = true
        mv(tmp_spk, output_spk; force = false)
        rm(backup_spk; force = true)
    catch err
        if moved_old && !isfile(output_spk) && isfile(backup_spk)
            try
                mv(backup_spk, output_spk; force = false)
            catch restore_err
                error("Failed to install new SPK and also failed to restore previous SPK. Install error: $(err). Restore error: $(restore_err). Backup remains at $(backup_spk)")
            end
        end
        error("Failed to replace output SPK safely. Previous SPK was preserved when possible. Original error: $(err)")
    end

    return output_spk
end

"""
    write_spkw13_spk!(segments, output_spk; kwargs...)

Write one native SPICE type-13 SPK file from sampled segment data.

Each element of `segments` must contain `epochs` and `states`, where epochs are
seconds past J2000 TDB and states are six-component vectors in km and km/s.
"""
function write_spkw13_spk!(
    segments,
    output_spk::AbstractString;
    spice_id::Union{Nothing,Integer} = nothing,
    object_name::Union{Nothing,AbstractString} = nothing,
    center_id::Union{Nothing,Integer} = nothing,
    center_name::Union{Nothing,AbstractString} = nothing,
    ref_frame_name::AbstractString = "J2000",
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    polynom_degree::Integer = 7,
    producer_id::AbstractString = "HighFidelityEphemerisModel.jl",
    internal_file_name::Union{Nothing,AbstractString} = nothing,
    ncomch::Integer = 0,
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    overwrite::Bool = true,
    tmp_spk::Union{Nothing,AbstractString} = nothing,
    verbose::Bool = true,
    show_progress::Bool = true,
)
    length(segments) > 0 || error("No SPK segments were provided.")

    body_code = _resolve_spice_integer_id(spice_id, object_name; label = "spice")
    center_code = _resolve_spice_integer_id(center_id, center_name; label = "center")

    output_spk_abs = _prepare_spk_final_path(output_spk; overwrite = overwrite)

    # Full-kernel writes are staged in a temporary BSP and installed only after
    # SPICE has closed the file. This keeps a previous output file recoverable if
    # `spkw13` fails partway through segment writing.
    tmp_spk_abs = tmp_spk === nothing ? joinpath(dirname(output_spk_abs), ".$(basename(output_spk_abs)).tmp_$(time_ns()).bsp") : abspath(tmp_spk)
    isfile(tmp_spk_abs) && rm(tmp_spk_abs; force = true)
    mkpath(dirname(tmp_spk_abs))

    if frame_def_file !== nothing
        # Furnish an optional frame kernel here so custom frame names are known
        # before SPICE validates the segment descriptor.
        SPICE.furnsh(String(frame_def_file))
    end

    ifname = internal_file_name === nothing ? String(producer_id) : String(internal_file_name)
    length(ifname) > 60 && (ifname = ifname[1:60])

    handle = nothing
    closed = false
    progress_enabled = verbose && show_progress

    try
        # `spkopn` creates a new SPK file and returns the DAF handle used by
        # `spkw13`. Always close the handle with `spkcls` before moving the file.
        handle = SPICE.spkopn(tmp_spk_abs, ifname, Int(ncomch))

        for (idx, segment) in enumerate(segments)
            epochs = Float64.(segment.epochs)
            states = [Float64.(state) for state in segment.states]
            segid = _spkw13_segment_id(segment_id, idx; segment_id_per_seg = segment_id_per_seg)
            first = hasproperty(segment, :first) ? Float64(segment.first) : epochs[1]
            last = hasproperty(segment, :last) ? Float64(segment.last) : epochs[end]

            _validate_spkw13_segment(
                epochs,
                states;
                degree = polynom_degree,
                first = first,
                last = last,
                segid = segid,
            )

            # Type 13 stores position, velocity, and epoch samples. States must
            # be dimensional `[x,y,z,vx,vy,vz]` in km and km/s.
            SPICE.spkw13(
                handle,
                body_code,
                center_code,
                String(ref_frame_name),
                first,
                last,
                segid,
                Int(polynom_degree),
                states,
                epochs,
            )

            _print_progress("writing SPK segments", idx, length(segments); enabled = progress_enabled)
        end

        # Closing flushes the DAF/SPK file to disk and balances `spkopn`.
        SPICE.spkcls(handle)
        closed = true
        _safe_install_spk!(tmp_spk_abs, output_spk_abs; overwrite = overwrite)
    catch err
        if handle !== nothing && !closed
            try
                SPICE.spkcls(handle)
            catch close_err
                verbose && println("Warning: failed to close temporary SPK after an error: ", close_err)
            end
        end
        isfile(tmp_spk_abs) && rm(tmp_spk_abs; force = true)
        rethrow()
    end

    verbose && !show_progress && println("SPK complete: ", _display_path(output_spk_abs))
    return output_spk_abs
end

"""
    append_spkw13_segment_to_spk!(segment; output_spk, kwargs...)

Create or append one native SPICE type-13 segment to an SPK file.
"""
function append_spkw13_segment_to_spk!(
    segment;
    output_spk::AbstractString,
    segment_index::Integer,
    append::Union{Nothing,Bool} = nothing,
    spice_id::Union{Nothing,Integer} = nothing,
    object_name::Union{Nothing,AbstractString} = nothing,
    center_id::Union{Nothing,Integer} = nothing,
    center_name::Union{Nothing,AbstractString} = nothing,
    ref_frame_name::AbstractString = "J2000",
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    segment_id_per_seg::Bool = false,
    polynom_degree::Integer = 7,
    producer_id::AbstractString = "HighFidelityEphemerisModel.jl",
    internal_file_name::Union{Nothing,AbstractString} = nothing,
    ncomch::Integer = 0,
    frame_def_file::Union{Nothing,AbstractString} = nothing,
)
    body_code = _resolve_spice_integer_id(spice_id, object_name; label = "spice")
    center_code = _resolve_spice_integer_id(center_id, center_name; label = "center")

    output_spk_abs = abspath(output_spk)
    splitext(output_spk_abs)[2] == ".bsp" || error("`output_spk` must end in `.bsp`: $(output_spk_abs)")
    mkpath(dirname(output_spk_abs))

    append_flag = append === nothing ? isfile(output_spk_abs) : Bool(append)
    if append_flag
        isfile(output_spk_abs) || error("Cannot append: output SPK does not exist: $(output_spk_abs)")
    elseif isfile(output_spk_abs)
        error("Output SPK already exists and `append=false`: $(output_spk_abs)")
    end

    if frame_def_file !== nothing
        SPICE.furnsh(String(frame_def_file))
    end

    epochs = Float64.(segment.epochs)
    states = [Float64.(state) for state in segment.states]
    segid = _spkw13_segment_id(segment_id, segment_index; segment_id_per_seg = segment_id_per_seg)
    first = hasproperty(segment, :first) ? Float64(segment.first) : epochs[1]
    last = hasproperty(segment, :last) ? Float64(segment.last) : epochs[end]

    _validate_spkw13_segment(
        epochs,
        states;
        degree = polynom_degree,
        first = first,
        last = last,
        segid = segid,
    )

    ifname = internal_file_name === nothing ? String(producer_id) : String(internal_file_name)
    length(ifname) > 60 && (ifname = ifname[1:60])

    # `spkopa` opens an existing BSP for appending; `spkopn` creates the first
    # file. This is the core path used by station-keeping loops that append one
    # propagated arc at a time.
    handle = append_flag ? SPICE.spkopa(output_spk_abs) : SPICE.spkopn(output_spk_abs, ifname, Int(ncomch))
    closed = false

    try
        SPICE.spkw13(
            handle,
            body_code,
            center_code,
            String(ref_frame_name),
            first,
            last,
            segid,
            Int(polynom_degree),
            states,
            epochs,
        )
        # Closing flushes the DAF/SPK file to disk and balances `spkopn`.
        SPICE.spkcls(handle)
        closed = true
    catch err
        if !closed
            try
                SPICE.spkcls(handle)
            catch close_err
                println("Warning: failed to close SPK after an error: ", close_err)
            end
        end
        rethrow()
    end

    return output_spk_abs
end

"""
    _read_spk_states_file_for_spkw13(state_file)

Read a debug state text file back into the in-memory layout expected by the
native type-13 writer. This is mainly used by compatibility tests and diagnostic
workflows.
"""
function _read_spk_states_file_for_spkw13(state_file::AbstractString)
    isfile(state_file) || error("Missing states file: $(state_file)")

    epochs = Float64[]
    states = Vector{Vector{Float64}}()

    open(state_file, "r") do io
        first_line_skipped = false
        for line in eachline(io)
            if !first_line_skipped
                first_line_skipped = true
                continue
            end
            s = strip(line)
            isempty(s) && continue
            fields = split(s, ",")
            length(fields) == 7 || error("Expected 7 comma-separated fields in $(state_file), got $(length(fields)): $(s)")
            push!(epochs, parse(Float64, strip(fields[1])))
            push!(states, [parse(Float64, strip(fields[i])) for i in 2:7])
        end
    end

    length(epochs) > 1 || error("State file $(state_file) must contain at least two states.")
    return (epochs = epochs, states = states, first = epochs[1], last = epochs[end])
end