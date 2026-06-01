# ODE Solutions to SPK Files

This tutorial shows how to convert ODE solution segments into files that can be used with SPICE-based analysis and visualization workflows.

The main helper is `ode_sol_to_spk`. It writes a BSP/SPK file from one or more ODE solution segments and can also write maneuver and metadata sidecar files.

!!! tip

    Before running this workflow, make sure you have a local copy of NAIF `mkspk` available. You can either put `mkspk` on your system path or set `ENV["MKSPK_CMD"]` to the full path of the executable.

    The automated tests use the same pattern: GitHub Actions downloads the Linux `mkspk` executable and passes its path through `ENV["MKSPK_CMD"]`.


## Expected ODE solution format

The SPK writer expects each coast arc to behave like a standard `OrdinaryDiffEq.jl` solution:

```julia
sol.t       # nondimensional time grid
sol(t_nd)   # interpolated nondimensional state at time t_nd
```

Each state returned by `sol(t_nd)` should contain at least

```julia
[x, y, z, vx, vy, vz]
```

in nondimensional units.

The parameter object must provide the canonical scaling values

```julia
parameters.TU   # seconds per nondimensional time unit
parameters.DU   # kilometers per nondimensional distance unit
parameters.VU   # kilometers/second per nondimensional velocity unit
```

Internally, states are converted before writing the MKSPK `STATES` files:

```julia
r_km   = x_nd[1:3] .* parameters.DU
v_kmps = x_nd[4:6] .* parameters.VU
```


## Basic SPK generation

A minimal call looks like this:

```julia
using HighFidelityEphemerisModel

mkspk_cmd = get(ENV, "MKSPK_CMD", "mkspk")

result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 301,
    ref_frame_name = "J2000",
    leapseconds_file = "naif0012.tls",
    mkspk_cmd = mkspk_cmd,
    dt_sec = 1800.0,
    segment_gap_sec = 1e-7,
    write_maneuvers = true,
    write_metadata = true,
    keep_intermediates = false,
)
```

Here:

- `sols` is a vector of ODE solution segments,
- `et0` is the reference epoch in seconds past J2000,
- `spice_id` is the NAIF object ID assigned to the generated trajectory,
- `center_id` is the NAIF ID of the central body,
- `ref_frame_name` is the frame used for the states written to the SPK,
- `dt_sec` is the sampling interval used to create the MKSPK input files,
- `segment_gap_sec` trims the right endpoint of non-final segments to avoid overlapping SPK coverage at impulsive boundaries.

The returned `result` is a `NamedTuple` containing generated paths and summaries:

```julia
result.output_spk
result.maneuver_txt
result.metadata_json
result.segment_count
result.maneuver_summary
```


## BSP/SPK file

The main output is the BSP file:

```julia
result.output_spk
```

By default, `ode_sol_to_spk` writes one SPK segment per ODE solution segment. This is useful for impulsive trajectories because it avoids interpolating across discontinuous velocity jumps.

If several ODE solution segments should be grouped into one SPK segment, pass explicit coast windows:

```julia
result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 301,
    coast_windows = [(1, 2), (3, 3)],
)
```

Each tuple in `coast_windows` gives the first and last ODE solution segment included in that SPK segment.


## Maneuver file

If `write_maneuvers = true`, the helper writes a maneuver file next to the BSP:

```julia
result.maneuver_txt
```

This file contains reconstructed velocity jumps between adjacent ODE solution segments. For segments `sols[k]` and `sols[k+1]`, the maneuver is computed from the difference between the terminal velocity of `sols[k]` and the initial velocity of `sols[k+1]`.

The file format is

```text
# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps
```

where the maneuver components are dimensional and expressed in m/s.

The total maneuver cost is summarized in the returned result:

```julia
result.maneuver_summary["total_delta_v_mps"]
result.maneuver_summary["total_delta_v_cmps"]
```


## OCP maneuver file

For optimal-control or station-keeping workflows, the optimizer control history can be written as a separate maneuver file:

```julia
result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 301,
    ocp_control = solution.u,
    ocp_control_times = times,
    write_ocp_maneuvers = true,
)
```

This writes an additional file:

```julia
result.ocp_maneuver_txt
```

The OCP maneuver file is separate from the reconstructed trajectory-jump maneuver file. The reconstructed maneuver file describes velocity discontinuities between adjacent ODE solution segments. The OCP maneuver file describes the optimizer-side commanded control history.

Rows 1:3 of `ocp_control` are treated as the vector control components. If row 4 is present, it is treated as the scalar magnitude or slack variable used by some OCP formulations.


## Metadata JSON

If `write_metadata = true`, the helper writes a metadata sidecar file:

```julia
result.metadata_json
```

The metadata file records information such as:

- generated product paths,
- SPICE object ID or object name,
- center ID or center name,
- reference frame,
- SPK type and polynomial degree,
- leapseconds and frame-definition files,
- segment coverage windows,
- sampling interval and segment gap,
- nondimensional scaling values,
- optional force-model metadata,
- maneuver summary information.

Force-model metadata can be supplied explicitly:

```julia
result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 301,
    force_model_metadata = Dict(
        "dynamics" => "eom_NbodySH_SPICE!",
        "SRP" => Dict(
            "enabled" => true,
            "Cr" => 1.15,
            "A_over_m" => 0.002,
            "P0" => 4.56e-6,
        ),
        "spherical_harmonics" => Dict(
            "enabled" => true,
            "body" => "Moon",
            "degree" => 4,
            "body_fixed_frame" => "MOON_PA",
        ),
    ),
)
```

Alternatively, common entries can be supplied through keyword arguments:

```julia
result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 301,
    dynamics_name = "eom_NbodySH_SPICE!",
    srp_enabled = true,
    srp_Cr = 1.15,
    srp_Am = 0.002,
    srp_P0 = 4.56e-6,
    spherical_harmonics_enabled = true,
    spherical_harmonics_body = "Moon",
    spherical_harmonics_nmax = 4,
    spherical_harmonics_frame = "MOON_PA",
)
```


## Label files

A label file is a lightweight sidecar file used by downstream visualization or analysis scripts to associate the generated BSP with an object name, SPICE ID, center, frame, and display settings.

The SPK writer returns the information needed to create such a label file:

```julia
result.output_spk
spice_id
center_id
ref_frame_name
```

The exact label format depends on the downstream tool. A simple label-writing step can therefore be kept separate from `ode_sol_to_spk`, for example:

```julia
label_path = replace(result.output_spk, ".bsp" => "_label.json")

open(label_path, "w") do io
    println(io, """
    {
      "spk_file": "$(basename(result.output_spk))",
      "object_id": "$(spice_id)",
      "object_name": "TRAJECTORY",
      "center_id": "$(center_id)",
      "frame": "$(ref_frame_name)"
    }
    """)
end
```

For mission-specific visualization, add any additional display fields required by the viewer, such as trajectory color, label text, or object radius.


## Intermediate files

Internally, the helper writes MKSPK `STATES` files and MKSPK setup files. By default, these are written into a temporary directory and removed after the BSP is generated:

```julia
keep_intermediates = false
```

For debugging, set:

```julia
keep_intermediates = true
```

When intermediate files are kept, their directory is returned in

```julia
result.intermediate_dir
```


## Validating the generated BSP

A generated BSP can be validated by furnishing it with SPICE, querying states from the BSP, and comparing those states against the original ODE solution:

```julia
using SPICE

furnsh(result.output_spk)

try
    state_spk, _ = spkezr(
        string(spice_id),
        et_query,
        "J2000",
        "NONE",
        string(center_id),
    )

    t_nd = (et_query - et0) / parameters.TU
    x_ref = sol(t_nd)

    state_ref = vcat(
        x_ref[1:3] .* parameters.DU,
        x_ref[4:6] .* parameters.VU,
    )

    @assert state_spk[1:3] ≈ state_ref[1:3]
    @assert state_spk[4:6] ≈ state_ref[4:6]
finally
    unload(result.output_spk)
end
```

This is the same validation strategy used in the automated end-to-end test. The test generates a small BSP, furnishes it, queries states from the BSP, compares them against the original ODE solution, unloads the BSP, and checks that the generated files are removed after the test.
