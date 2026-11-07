# HighFidelityEphemerisModel.jl release notes

## Unreleased

Changes relative to v0.2.3 (the `Project.toml` version is not bumped yet).

### Breaking changes

- **`harris_priester_f_density` default output changed.** The default callback is now the
  full Montenbruck & Gill (*Satellite Orbits*, §3.5.2) Harris-Priester model:
  - it includes the diurnal bulge, using a SPICE-free analytic Sun;
  - altitude is geodetic, above the WGS-84 ellipsoid;
  - the default reference radius is `R_earth_km = 6378.137` (was `6378.0`, spherical);
  - densities are interpolated exponentially between table heights (was a monotone cubic
    spline in ρ);
  - the density is **zero outside 100–1000 km** (the altitude used to be clamped to that
    range).

  `use_min = true` is no longer the default. It now selects the bulge-antapex column with
  no bulge, but it still differs from the old result because of the geodetic altitude,
  the exponential interpolation, and the zero density below 100 km (and above 1000 km).
  To approximate the old behaviour, use
  `harris_priester_f_density(6378.0; flattening = 0, use_min = true)`.
  `HarrisPriesterModel(h)` gives the same values as before at the table heights but now
  uses exponential interpolation between them.
- **The Harris-Priester profile is only C⁰ in altitude.** `∂ρ/∂h` jumps at every table
  height (every 10 km below 300 km, every 20–40 km above). ForwardDiff STMs/Jacobians are
  still the model's derivative almost everywhere. On orbits that cross table heights they
  lose accuracy against finite-difference STMs (~4e-4 observed over a few hours, versus
  ≲1e-5 inside one interval), and gradient-based optimization may see a non-smooth
  objective. The density also drops discontinuously to 0 at 100 km.
- **The default Harris-Priester callback needs an Earth-fixed `frame_PCPF`.** The bulge is
  placed using the Sun in an Earth-fixed frame, so `frame_PCPF` must be e.g. `"ITRF93"`
  or `"IAU_EARTH"`. With an inertial `frame_PCPF` (e.g. `"J2000"`) the bulge lands at the
  wrong local time and **no error is raised**; a factor of ≈0.6 in density at 400 km was
  seen in one case. A user-supplied `sun_position` must return the Sun in the same frame
  as `r_km`.
- **`setproperty!` on parameter objects restricts some field types.**
  `SpiceParameters`, `InterpParameters` and `EphemeridesParameters` are now parametric.
  The fields `spherical_harmonics_data`, `factorial_alias`, `f_density` and (Ephemerides)
  `ephemerides_backend` are type parameters and only accept a value of the same type.
  Assigning another type throws an `ArgumentError` that tells you to build new
  parameters. For example, `params.f_density = (et, r) -> ...` on parameters built with
  another density model, or `params.spherical_harmonics_data = nothing`, now throw.
  Other fields work as before, with conversion. `DU`, `TU` and `VU` are now always
  stored as `Float64`: an `Int` `DU` passed to the constructor is converted, so
  `params.DU` is `3000.0`, not `3000`.
- **`load_spherical_harmonics(...)["Cnm"]` / `["Snm"]` are no longer `Dict`s.** They are
  now `SphericalHarmonicsCoefficients <: AbstractDict{Tuple{Int,Int},Float64}` views of a
  `SphericalHarmonicsField`, which is stored under the new key `"FIELD"`. Indexing,
  `get`, `haskey`, iteration and `length` work as before. Assigning an existing key
  writes through to the field, and new keys cannot be added. Code that checks
  `isa Dict` or uses `Dict`-only methods must be adapted.
- **`load_spherical_harmonics(path, nmax, false)` gives different (correct) accelerations.**
  v0.2.3 stored the normalized file coefficients but the acceleration routines treated
  them as un-normalized, which was wrong. With `denormalize = false` the data are now
  used with the correct normalization, so results differ from v0.2.3. Indexing `"Cnm"`
  returns normalized values for `denormalize = false` and un-normalized values for
  `denormalize = true`, as before.
- **`build_ephemerides_frame_system` throws on unsupported `frame_PCPF`.** The supported
  values are `"MOON_PA"`, `"ITRF93"`, `"IAU_EARTH"`, and the inertial
  `"J2000"`/`"ICRF"`, `"EME2000"`, `"ECLIPJ2000"`. Other values used to be silently
  ignored and only failed at the first rotation query. A custom
  `FrameTransformations.FrameSystem` can still be passed as `ephemerides_frame_system`.
- **`show(params)` prints the bare type name.** `typeof(params)` now includes type
  parameters, e.g. `SpiceParameters{Nothing, typeof(factorial), Nothing}`.

### New features

- **Ephemerides.jl backend with Earth frames.** `EphemeridesParameters` (and
  `build_ephemerides_frame_system`) now support Earth-centred problems with
  `frame_PCPF = "ITRF93"` (binary Earth PCK, e.g. `earth_latest_high_prec.bpc`) or
  `"IAU_EARTH"` (IAU 2009 rotation model, no kernel needed), as well as `"ECLIPJ2000"`.
  Frame names are matched case-insensitively, and construction and the EOMs use the same
  mapping (`ephemerides_axes_symbol`).
- **Harris-Priester and Sun utilities.**
  - New `harris_priester_density(h_km, cos_psi; n = 6)`.
  - `harris_priester_f_density` gains the keywords `flattening`, `n`, `lag_deg`,
    `sun_position` and `use_min = nothing`.
  - New SPICE-free Sun utilities: `sun_position_analytic`,
    `sun_position_earth_fixed_analytic`, and `ephemerides_sun_position_function`, which
    builds a Sun callback from a FrameTransformations frame system.
  - The default Harris-Priester callback is thread-safe and ForwardDiff-compatible.
- **Jacchia-Roberts.** `jacchia_roberts_f_density` gains the keywords `sun_model`
  (`:spice` default, or `:analytic` for SPICE-free and thread-safe), `sun_position`,
  `R_equatorial_km` and `flattening`.
- **Spherical-harmonics types.**
  - New exported types `SphericalHarmonicsField` (fully normalized coefficients with
    precomputed recursion tables) and `SphericalHarmonicsCoefficients` (the dictionary
    views).
  - New `spherical_harmonics_accel_normalized(x, y, z, field, GM, R, nmax)`, which works
    on scalars and returns a tuple.
  - Any degree available in the gravity file can now be used; this is tested to degree 180.
- **Spherical-harmonics coefficient changes and sensitivities.**
  - Coefficients can be modified in place (`sh["Cnm"][2, 0] = value`), or `sh["Cnm"]` /
    `sh["Snm"]` can be replaced by ordinary `Dict`s of un-normalized coefficients. Both
    are honoured by the EOMs.
  - Dictionaries with non-`Float64` values (e.g. ForwardDiff duals) and a Dual `GM` are
    supported by `spherical_harmonics_accel(_PCPF)`. In the EOMs this needs a state of the
    same element type (e.g. `x .+ zero(c)` inside the differentiated closure); otherwise a
    clear `ArgumentError` is thrown. This allows sensitivities with respect to the
    coefficients or GM.
- **Ephemerides cache padding.** `ensure_ephemerides_thread_caches!(x; nslots)` accepts a
  provider, `nothing`, or any object such as a frame system. The new
  `find_ephemerides_providers(x)` lists the `Ephemerides.EphemerisProvider`s reachable
  from an object.
- **Docstrings** for `SpiceParameters`, `InterpParameters` and `EphemeridesParameters`.

### Fixes

- **Thread safety with a prebuilt frame system.** When `EphemeridesParameters` were built
  from only `ephemerides_frame_system`, the Ephemerides.jl per-thread segment caches were
  never padded to `Threads.maxthreadid()`. `Threads.@threads` / `EnsembleThreads` then
  threw `BoundsError` whenever thread IDs exceed `nthreads()`, which is the default on
  Julia ≥ 1.12 because of the interactive thread. Every provider reachable from the
  frame system (through the closures in its point/axes nodes) is now padded. If none is
  reachable, a warning suggests also passing `ephemerides_provider`.
- **Spherical-harmonics accuracy.**
  - The acceleration now uses a stable fully-normalized Cunningham recursion, replacing
    the `BigInt`/factorial-based un-normalized evaluation. The old code overflowed or
    lost precision at moderate degree and allocated heavily.
  - The recursion is non-singular at the poles and ForwardDiff-compatible.
- **Gauss variational equations.** The perturbing acceleration is now accumulated
  directly rather than as (total − central), which avoids cancellation.
- **Jacchia-Roberts geodetic conversion.** It is now smooth and correct on and near the
  polar axis, including for dual numbers. The height formula has no `p / cos φ`
  singularity.
- **Clearer errors.** A missing `frame_PCPF` (or interpolated transformation) with drag
  or spherical harmonics enabled now gives an explicit error for every backend.

### Performance

All EOMs share one kernel that works on scalars and tuples. Parameter structs have
concrete field types, and the EOMs infer concretely for `Float64` and ForwardDiff `Dual`
states. No buffers are stored in the parameters, so one parameter object can be shared
across threads. Results match v0.2.3 to ≤ 3.5e-16 relative, except where the physics
changed (see above).

Measurements: median of BenchmarkTools runs on one desktop CPU. Moon-centred problem
with Earth and Sun, SRP on; in-place EOM unless stated otherwise.

| EOM / backend | v0.2.3 | now |
|---|---|---|
| `eom_Nbody!` SPICE | 3.87 µs, 75 allocs | 2.37 µs, 4 allocs |
| `eom_Nbody!` Interp | 1.30 µs, 73 allocs | 0.33 µs, 0 allocs |
| `eom_Nbody!` Ephemerides | 1.74 µs, 85 allocs | 0.65 µs, 0 allocs |
| `eom_Nbody!` + drag (constant density), SPICE / Interp / Ephemerides | 5.60 / 2.33 / 2.79 µs | 3.83 / 0.50 / 0.86 µs |
| `eom_NbodySH!` nmax = 10, SPICE / Interp / Ephemerides | 15.6 / 12.2 / 12.7 µs, ~700 allocs | 4.3 / 0.94 / 1.4 µs, 20 / 12 / 12 allocs |
| `eom_NbodySH!` nmax = 50, SPICE / Interp / Ephemerides | 434 / 429 / 430 µs, ~12,000 allocs | 8.9 / 5.6 / 6.0 µs, 20 / 12 / 12 allocs |
| ForwardDiff Jacobian of `eom_NbodySH`, nmax = 50 | 1,773 µs | 59–64 µs |
| `eom_stm_Nbody_*_fd!`, SPICE / Interp / Ephemerides | 9.2 / 4.1 / 5.1 µs, ~200 allocs | 6.5 / 2.1 / 3.1 µs, 24 / 16 / 16 allocs |
| `gve_mee_Nbody!`, SPICE / Interp / Ephemerides | 4.1 / 1.6 / 2.0 µs, ~100 allocs | 2.5 / 0.41 / 0.84 µs, 10 / 6 / 6 allocs |

Allocations that remain:
- The SPICE backend allocates inside SPICE.jl's `spkpos` / `pxform`.
- With drag, the `r_km::Vector` handed to `f_density` costs up to 2 allocations; the
  compiler removes it for simple callbacks.
- The spherical-harmonics recursion allocates its O(nmax) work vectors.
- Out-of-place EOMs allocate the returned vector.
