"""Native SPICE type-13 SPK writers"""


# Internal file name written into the header of new SPK files (at most 60 characters).
const _SPK_INTERNAL_FILE_NAME = "HighFidelityEphemerisModel.jl"


"""
    _spkw13_segment_id(segment_id, segment_index)

Segment identifier `"<segment_id>_<index>"`, with the index zero-padded to three
digits (e.g. `"HFEM_SPK_SEGMENT_002"`; four digits from index 1000 on). NAIF segment identifiers are limited to
40 printable characters, so this helper checks the limit before calling SPICE.
"""
function _spkw13_segment_id(segment_id::AbstractString, segment_index::Integer)
    isempty(strip(segment_id)) && error("`segment_id` must not be empty.")
    all(isprint, segment_id) || error("`segment_id` contains non-printable characters: $(repr(segment_id))")
    segid = string(segment_id, "_", lpad(string(segment_index), 3, '0'))
    length(segid) <= 40 || error("SPK segment ID $(segid) is too long ($(length(segid)) characters); type-13 segment IDs must be at most 40 characters, including the `_<index>` suffix.")
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
    _close_spk_handle_after_error(handle; verbose=true)

Release an SPK handle on an error path. `spkcls` refuses to close a file with no
segments (`SPICE(NOSEGMENTSFOUND)`), so fall back to `dafcls`, which closes the
underlying DAF unconditionally. Leaving the handle open would keep the file
locked on Windows and leak a SPICE file-table slot.
"""
function _close_spk_handle_after_error(handle; verbose::Bool = true)
    try
        SPICE.spkcls(handle)
    catch close_err
        try
            SPICE.dafcls(handle)
        catch daf_err
            verbose && println("Warning: failed to close SPK after an error: ", close_err, "; ", daf_err)
        end
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
    _prepare_spkw13_segment(segment, segment_index; segment_id, polynom_degree)

Convert one sampled segment (`epochs`, `states`, optional `first`/`last`) to the
`Float64` layout expected by `spkw13`, build its segment ID and validate it.
"""
function _prepare_spkw13_segment(segment, segment_index::Integer; segment_id::AbstractString, polynom_degree::Integer)
    epochs = Float64.(segment.epochs)
    states = [Float64.(state) for state in segment.states]
    segid = _spkw13_segment_id(segment_id, segment_index)
    first = hasproperty(segment, :first) ? Float64(segment.first) : epochs[1]
    last = hasproperty(segment, :last) ? Float64(segment.last) : epochs[end]
    _validate_spkw13_segment(epochs, states; degree = polynom_degree, first = first, last = last, segid = segid)
    return (; epochs, states, segid, first, last)
end

function _spkw13!(handle, seg, spice_id::Integer, center_id::Integer, ref_frame_name::AbstractString, polynom_degree::Integer)
    # Type 13 stores position, velocity, and epoch samples. States must be
    # dimensional `[x,y,z,vx,vy,vz]` in km and km/s.
    SPICE.spkw13(handle, Int(spice_id), Int(center_id), String(ref_frame_name),
        seg.first, seg.last, seg.segid, Int(polynom_degree), seg.states, seg.epochs)
    return nothing
end

"""
    write_spkw13_spk!(segments, output_spk; spice_id, center_id, kwargs...)

Write one native SPICE type-13 SPK file from sampled segment data. `spice_id`
and `center_id` are integer SPICE codes.

Each element of `segments` must contain `epochs` and `states`, where epochs are
seconds past J2000 TDB and states are six-component vectors in km and km/s.

The kernel is written to a temporary file next to `output_spk` and only moved
into place after SPICE has closed it, so an existing output is preserved if
writing fails.
"""
function write_spkw13_spk!(
    segments,
    output_spk::AbstractString;
    spice_id::Integer,
    center_id::Integer,
    ref_frame_name::AbstractString = "J2000",
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    polynom_degree::Integer = 7,
    frame_def_file::Union{Nothing,AbstractString} = nothing,
    overwrite::Bool = true,
    verbose::Bool = true,
    show_progress::Bool = true,
)
    length(segments) > 0 || error("No SPK segments were provided.")
    output_spk_abs = _prepare_spk_final_path(output_spk; overwrite = overwrite)

    # Furnish an optional frame kernel first so custom frame names are known
    # before SPICE validates the segment descriptor.
    frame_def_file !== nothing && SPICE.furnsh(String(frame_def_file))

    # Convert and validate every segment before the temporary BSP is opened.
    # `spkcls` refuses to close an SPK that contains no segments, so failing
    # validation after `spkopn` would otherwise leave the file handle open
    # (and the file locked on Windows).
    prepared = [_prepare_spkw13_segment(segment, idx; segment_id = segment_id, polynom_degree = polynom_degree)
                for (idx, segment) in enumerate(segments)]

    tmp_spk = joinpath(dirname(output_spk_abs), ".$(basename(output_spk_abs)).tmp_$(time_ns()).bsp")
    handle = nothing
    closed = false
    try
        handle = SPICE.spkopn(tmp_spk, _SPK_INTERNAL_FILE_NAME, 0)
        for (idx, seg) in enumerate(prepared)
            _spkw13!(handle, seg, spice_id, center_id, ref_frame_name, polynom_degree)
            _print_progress("writing SPK segments", idx, length(prepared); enabled = verbose && show_progress)
        end
        # Closing flushes the DAF/SPK file to disk and balances `spkopn`.
        SPICE.spkcls(handle)
        closed = true
        _safe_install_spk!(tmp_spk, output_spk_abs; overwrite = overwrite)
    catch
        handle !== nothing && !closed && _close_spk_handle_after_error(handle; verbose = verbose)
        # Never let a cleanup failure mask the original error.
        try
            isfile(tmp_spk) && rm(tmp_spk; force = true)
        catch rm_err
            verbose && println("Warning: failed to remove temporary SPK after an error: ", rm_err)
        end
        rethrow()
    end

    return output_spk_abs
end

"""
    append_spkw13_segment_to_spk!(segment, output_spk; segment_index, spice_id, center_id, kwargs...)

Append one native SPICE type-13 segment to an existing SPK file. `spice_id` and
`center_id` are integer SPICE codes. New kernels are created with
`write_spkw13_spk!`.
"""
function append_spkw13_segment_to_spk!(
    segment,
    output_spk::AbstractString;
    segment_index::Integer,
    spice_id::Integer,
    center_id::Integer,
    ref_frame_name::AbstractString = "J2000",
    segment_id::AbstractString = "HFEM_SPK_SEGMENT",
    polynom_degree::Integer = 7,
    frame_def_file::Union{Nothing,AbstractString} = nothing,
)
    output_spk_abs = abspath(output_spk)
    isfile(output_spk_abs) || error("Cannot append: output SPK does not exist: $(output_spk_abs)")

    frame_def_file !== nothing && SPICE.furnsh(String(frame_def_file))
    seg = _prepare_spkw13_segment(segment, segment_index; segment_id = segment_id, polynom_degree = polynom_degree)

    handle = SPICE.spkopa(output_spk_abs)
    try
        _spkw13!(handle, seg, spice_id, center_id, ref_frame_name, polynom_degree)
    catch
        _close_spk_handle_after_error(handle)
        rethrow()
    end
    SPICE.spkcls(handle)

    return output_spk_abs
end

"""
    read_spk_segments(spk) -> Vector{NamedTuple}

Read the descriptor of every segment of an SPK file, in file order: `spice_id`,
`center_id`, `ref_frame_name`, `spk_type`, `first`/`last` (coverage, ET seconds)
and `segment_id`. For type-13 segments, `polynom_degree` and `point_count` are
read from the segment data; they are `nothing` for other types.
"""
function read_spk_segments(spk::AbstractString)
    spk_abs = abspath(spk)
    isfile(spk_abs) || error("SPK file not found: $(spk_abs)")

    segments = NamedTuple[]
    handle = SPICE.dafopr(spk_abs)
    try
        SPICE.dafbfs(handle)
        while SPICE.daffna()
            dc, ic = SPICE.dafus(SPICE.dafgs(), 2, 6)
            segid = strip(SPICE.dafgn())
            frame_code, spk_type, data_end = ic[3], ic[4], ic[6]
            degree = nothing
            point_count = nothing
            if spk_type == 13
                # Type-13 segments end with (window size - 1, number of states);
                # the Hermite degree is 2 * window size - 1.
                window_minus_one, n = SPICE.dafgda(handle, data_end - 1, data_end)
                degree = 2 * round(Int, window_minus_one) + 1
                point_count = round(Int, n)
            end
            frame = SPICE.frmnam(frame_code)
            push!(segments, (
                spice_id = Int(ic[1]),
                center_id = Int(ic[2]),
                ref_frame_name = isempty(frame) ? string(frame_code) : String(frame),
                spk_type = Int(spk_type),
                first = Float64(dc[1]),
                last = Float64(dc[2]),
                segment_id = String(segid),
                polynom_degree = degree,
                point_count = point_count,
            ))
        end
    finally
        SPICE.dafcls(handle)
    end
    return segments
end
