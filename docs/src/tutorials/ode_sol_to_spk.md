# ODE Solutions to SPK Files

This tutorial shows how to convert ODE solution segments into files that can be used with SPICE-based analysis and visualization workflows.

The main helper is `ode_sol_to_spk`. It writes a BSP/SPK file from one or more ODE solution segments using native SPICE SPK writing routines. It can also write maneuver and metadata sidecar files.

The current implementation writes SPK type-13 segments directly through SPICE. No external SPK-writing executable is required.

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

Internally, states are converted before writing the SPK:

```julia
r_km   = x_nd[1:3] .* parameters.DU
v_kmps = x_nd[4:6] .* parameters.VU
et     = et0 + t_nd * parameters.TU
```

## Basic SPK generation

A minimal call looks like this:

```julia
using HighFidelityEphemerisModel

result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 301,
    ref_frame_name = "J2000",
    dt_sec = 1800.0,
    segment_gap_sec = 1e-7,
    write_metadata = true,
)
```

Here:

- `sols` is a vector of ODE solution segments,
- `et0` is the reference epoch in seconds past J2000,
- `spice_id` is the NAIF object ID assigned to the generated trajectory,
- `center_id` is the NAIF ID of the central body,
- `ref_frame_name` is the frame used for the states written to the SPK,
- `dt_sec` is the sampling interval used for each SPK segment,
- `segment_gap_sec` trims the right endpoint of non-final segments to avoid overlapping SPK coverage at impulsive boundaries.

The returned `result` is a `NamedTuple` containing generated paths and summaries:

```julia
result.output_spk
result.maneuver_txt
result.ocp_maneuver_txt
result.trajectory_maneuver_txt
result.metadata_json
result.segment_count
result.epoch_ranges
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

## Incremental append workflow

For station-keeping simulations, the function can append one propagated ODE arc at a time. This avoids storing every ODE solution in memory until the end of the run.

```julia
output_spk = "sk_seed_001_native_true.bsp"

for k in 1:N_recurse
    # solve OCP, apply/corrupt the first maneuver, and propagate truth here
    # sol_recurse is the ODE solution for this one recursion interval

    result = ode_sol_to_spk(
        sol_recurse,
        et0,
        parameters;
        output_spk = output_spk,
        spice_id = -200001,
        center_id = 301,
        ref_frame_name = "J2000",
        append = k > 1,
        overwrite = k == 1,
        segment_index = k,
        dt_sec = 1800.0,
        segment_gap_sec = k < N_recurse ? 1e-7 : 0.0,
        write_metadata = false,
    )
end
```

The first call creates the BSP. Later calls append new SPK type-13 segments to the same BSP.

## Maneuver files

There are two maneuver-file concepts.

### Primary executed/OCP maneuver file

For optimal-control or station-keeping workflows, the main maneuver product can be written from a control history:

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

Rows 1:3 of `ocp_control` are the vector control components. If row 4 is present, it is treated as the scalar control/slack variable used by the OCP formulation. The scalar convention is summarized in metadata.

For station-keeping truth products, pass the actually applied/corrupted control history as the `ocp_control` input. In that case, the maneuver file represents executed maneuvers, not the originally planned control.

### Trajectory-jump diagnostic maneuver file

If `write_maneuvers = true`, the helper can also write a diagnostic maneuver file reconstructed from velocity jumps between adjacent ODE solution segments:

```julia
result = ode_sol_to_spk(
    sols,
    et0,
    parameters;
    output_spk = "trajectory.bsp",
    spice_id = -123456,
    center_id = 301,
    write_maneuvers = true,
)
```

The diagnostic file is separate from the executed/OCP control file. It contains:

```text
# ETSECONDS,DVX_mps,DVY_mps,DVZ_mps
```

For incremental single-arc station-keeping output, trajectory-jump diagnostics are usually disabled because each call only has one arc.

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
- optional leapseconds/frame-definition paths for traceability,
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
            "A_over_m" => 0.016,
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
    srp_Am = 0.016,
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

The exact label format depends on the downstream tool. A simple label-writing step can therefore be kept separate from `ode_sol_to_spk`.

## Intermediate/debug files

The native pipeline samples states in memory and writes the BSP directly with SPICE. It does not need setup files or an external SPK-writing executable.

For debugging, state text files can still be written with lower-level helper functions such as `write_segmented_states_for_spk!` or `write_solution_segment_states_for_spk!`. These files are for inspection only; they are not required for normal BSP generation.

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

This is the same validation strategy used in the automated end-to-end tests. The tests generate small BSPs with native SPICE, furnish them, query states from the BSP, compare them against the original ODE solutions, unload the BSPs, and check that temporary files are cleaned up.
