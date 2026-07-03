# Core routines

## Parameters

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "parameters.jl",
]
```

## Ephemerides and interpolation

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "ephemerides.jl",
    "ephemeris_interpolation.jl",
    "transformation_interpolation.jl",
]
```

## Equations of motion

The preferred translational EOM names are `eom_Nbody!`, `eom_Nbody`, `eom_NbodySH!`, and `eom_NbodySH`. The backend-specific files below are still part of the public API for compatibility and backend-specific tests.

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "eoms/eom_Nbody_SPICE.jl",
    "eoms/eom_Nbody_Interp.jl",
    "eoms/eom_Nbody_Ephemerides.jl",
    "eoms/eom_NbodySH_SPICE.jl",
    "eoms/eom_NbodySH_Interp.jl",
    "eoms/eom_NbodySH_Ephemerides.jl",
    "eoms/eom_dispatch.jl",
]
```

## Utility routines

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "utils.jl",
]
```

## Perturbation routines

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "perturbations/third_body.jl",
    "perturbations/spherical_harmonics.jl",
    "perturbations/solar_radiation_pressure.jl",
    "perturbations/drag.jl",
    "perturbations/harrispriester.jl",
    "perturbations/jacchiaroberts.jl",
]
```

## Events

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "events.jl",
]
```

## SPK generation helpers

```@autodocs
Modules = [HighFidelityEphemerisModel]
Order   = [:type, :constant, :function]
Pages   = [
    "spk/states.jl",
    "spk/spkw13.jl",
    "spk/maneuvers.jl",
    "spk/metadata.jl",
    "spk/incremental.jl",
    "spk/ode_sol_to_spk.jl",
]
```