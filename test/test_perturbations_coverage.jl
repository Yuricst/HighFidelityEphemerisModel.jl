"""Additional tests for perturbation models and GVE helpers (Jacchia-Roberts low-altitude
branches, spherical-harmonics coefficient containers/loaders, Keplerian GVE wrappers)."""

using AstrodynamicsCore
using ForwardDiff
using LinearAlgebra
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


# ---------------------------------------------------------------------------
# Gauss variational equations (Keplerian, out-of-place forms, perturbing accel)
# ---------------------------------------------------------------------------

function _pertD_spice_params()
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2020-01-01T00:00:00")
    return HighFidelityEphemerisModel.SpiceParameters(
        et0, 3000.0, GMs, naif_ids, "J2000", "NONE")
end


@testset "Keplerian two-body drift (out-of-place)" begin
    H = HighFidelityEphemerisModel
    mu = 1.3
    for kep in ([2.0, 0.0, 0.3, 0.1, 0.0, 0.4],
                [3.0, 0.4, 0.8, 1.0, 0.5, 2.4],
                [8.0, 0.85, 0.3, 2.0, 1.1, 0.2])
        dkep = H.eom_kep_twobody(kep, mu)
        dkep_inplace = similar(kep)
        H.eom_kep_twobody!(dkep_inplace, kep, mu, 0.0)
        @test dkep == dkep_inplace
        @test dkep[1:5] == zeros(5)
        # independent check: true-anomaly rate is h / r^2 from the Cartesian state
        rv = AstrodynamicsCore.kep2rv(kep, mu)
        h = norm(cross(rv[1:3], rv[4:6]))
        @test dkep[6] ≈ h / norm(rv[1:3])^2 rtol = 1e-12
        # the explicit time argument is accepted and ignored
        @test H.eom_kep_twobody(kep, mu, 12.5) == dkep
    end
    @test_throws DomainError H.eom_kep_twobody([1.0, 1.2, 0.1, 0.0, 0.0, 0.0], mu)
    @test_throws DomainError H.eom_kep_twobody([1.0, 0.1, 0.1, 0.0, 0.0, 0.0], -mu)
    @test_throws DimensionMismatch H.eom_kep_twobody(ones(5), mu)
end


@testset "Perturbing acceleration and Keplerian GVE (SPICE)" begin
    H = HighFidelityEphemerisModel
    params = _pertD_spice_params()
    mu = params.mus[1]
    t = 0.37

    for kep in ([1.2, 0.1, 0.4, 0.3, 0.2, 0.9],
                [2.0, 0.3, 0.6, 1.0, 0.5, 2.0])
        rv = AstrodynamicsCore.kep2rv(kep, mu)

        # perturbing acceleration = full N-body acceleration minus central-body Kepler term
        drv = zeros(6)
        H.eom_Nbody!(drv, rv, params, t)
        r = rv[1:3]
        a_expected = drv[4:6] + mu * r / norm(r)^3
        a_pert = fill(NaN, 3)
        @test H._perturbing_accel!(a_pert, rv, params, t, Val(false)) === a_pert
        @test a_pert ≈ a_expected rtol = 1e-9
        # third-body perturbations are small but nonzero relative to the central body
        @test 0 < norm(a_pert) < 1e-2 * mu / norm(r)^2

        # out-of-place Keplerian GVE matches the in-place form
        dkep = H.gve_kep_Nbody(kep, params, t)
        dkep_inplace = similar(kep)
        H.gve_kep_Nbody!(dkep_inplace, kep, params, t)
        @test dkep == dkep_inplace

        # independent check: chain rule d(kep)/dt = (∂kep/∂rv) * d(rv)/dt
        J = ForwardDiff.jacobian(x -> AstrodynamicsCore.rv2kep(x, mu), rv)
        @test dkep ≈ J * drv rtol = 1e-8 atol = 1e-12
    end

    @test_throws DomainError H._perturbing_accel!(zeros(3), zeros(6), params, t, Val(false))
end


# ---------------------------------------------------------------------------
# Jacchia-Roberts: 90-100 km branch and SPICE-free altitude guard
# ---------------------------------------------------------------------------

@testset "Jacchia-Roberts below 100 km" begin
    H = HighFidelityEphemerisModel

    # At and below 90 km the model returns the boundary density ρ₀ = 3.46e-9 g/cm³
    # times the (height-dependent) seasonal-latitudinal/geomagnetic correction.
    geo = H.JacchiaRobertsGeomagneticExposphericParams(150.0, 150.0, 3.0)
    mjd = 58849.0
    for h in (60.0, 85.0, 90.0)
        @test H.JacchiaRobertsModel(h; utc_mjd = mjd) ≈
              1e3 * 3.46e-9 * H.rho_cor(h, mjd, 0.0, geo) rtol = 1e-14
    end
    ρ90 = H.JacchiaRobertsModel(90.0)
    @test ρ90 > 0

    # 90 < h < 100 km (`rho_100`): the hydrostatic integral starts at ρ₀, T₀ at 90 km,
    # so the density is continuous at 90 km.
    @test H.JacchiaRobertsModel(90.0 + 1e-9) ≈ ρ90 rtol = 1e-7
    @test H.JacchiaRobertsModel(90.0 + 1e-6) < ρ90

    # strictly decreasing with height across the branch
    hs = range(90.5, 99.5; length = 10)
    ρs = [H.JacchiaRobertsModel(h) for h in hs]
    @test all(isfinite, ρs)
    @test all(>(0), ρs)
    @test all(diff(ρs) .< 0)

    # the explicit-geometry method agrees with the convenience method
    h = 95.0
    @test H.JacchiaRobertsModel(h, [6378.0 + h, 0.0, 0.0], [1.0, 0.0, 0.0],
        58849.0, 0.0, geo) ≈ H.JacchiaRobertsModel(h) rtol = 1e-14
end


@testset "Jacchia-Roberts SPICE-free altitude guard" begin
    H = HighFidelityEphemerisModel
    f_density = H.jacchia_roberts_f_density(sun_model = :analytic)
    Re = 6378.1366
    @test_throws ErrorException f_density(0.0, [Re + 80.0, 0.0, 0.0])
    @test_throws ErrorException f_density(0.0, [0.0, 0.0, 6356.7519 + 99.0])
    # 105 km *geodetic* over the pole (about 84 km if altitude were measured from Re):
    # must be accepted, so the guard uses geodetic altitude
    @test isfinite(f_density(0.0, [0.0, 0.0, 6356.7519 + 105.0]))
    ρ = f_density(0.0, [Re + 400.0, 0.0, 0.0])
    @test isfinite(ρ) && ρ > 0
end


# ---------------------------------------------------------------------------
# Spherical harmonics: coefficient container, generic Dict coefficients, loader
# ---------------------------------------------------------------------------

# Independent normalization factor N_nm (M&G Eq. (3.13)): C_nm = N_nm * Cbar_nm
_pertD_Nnm(n, m) = sqrt((m == 0 ? 1 : 2) * (2n + 1) * factorial(big(n - m)) / factorial(big(n + m)))


@testset "SphericalHarmonicsCoefficients Dict interface" begin
    H = HighFidelityEphemerisModel
    filepath = joinpath(@__DIR__, "../data/earth/GGM03S.tab")
    data_norm = H.load_spherical_harmonics(filepath, 4, false)
    data_denorm = H.load_spherical_harmonics(filepath, 4, true)
    Cn, Sd = data_norm["Cnm"], data_denorm["Snm"]

    # `get` with tuple keys: present keys return the value, absent ones the default
    @test get(Cn, (2, 0), NaN) == Cn[2, 0]
    @test get(Cn, (4, 4), NaN) == data_norm["FIELD"].Cbar[5, 5]
    @test get(Sd, (3, 1), NaN) ≈ data_norm["Snm"][3, 1] * Float64(_pertD_Nnm(3, 1)) rtol = 1e-14
    @test get(Cn, (5, 0), -1.0) == -1.0      # beyond nmax
    @test get(Cn, (1, 0), -2.0) == -2.0      # degree 1 not stored
    @test get(Cn, (3, 4), -3.0) == -3.0      # m > n
    @test get(Cn, "C20", -4.0) == -4.0       # non-tuple key
    # `haskey` with non-tuple keys is always false
    @test !haskey(Cn, "C20")
    @test !haskey(Cn, 2)
    for key in Any["C20", 2, (2, 0, 0), :C20]   # runtime dispatch on non-tuple keys
        @test !haskey(Cn, key)
    end
    @test haskey(Cn, (2, 0))
end


@testset "Spherical harmonics from generic-valued Dicts" begin
    H = HighFidelityEphemerisModel
    filepath = joinpath(@__DIR__, "../data/earth/GGM03S.tab")
    nmax = 5
    data = H.load_spherical_harmonics(filepath, nmax, true)
    GM, R = data["GM"], data["REFERENCE RADIUS"]
    rvec = [3000.0, -4000.0, 4500.0]
    a_ref = H.spherical_harmonics_accel_PCPF(rvec, data["Cnm"], data["Snm"], GM, R, nmax)

    # abstractly-typed values (e.g. assembled by hand): coefficient type is inferred
    # from the stored values
    C_any = Dict{Tuple{Int,Int},Any}(k => v for (k, v) in data["Cnm"])
    S_real = Dict{Tuple{Int,Int},Real}(k => v for (k, v) in data["Snm"])
    @test H._sh_coef_type(C_any) == Float64
    @test H._sh_coef_type(S_real) == Float64
    @test H._sh_coef_type(Dict{Tuple{Int,Int},Real}((2, 0) => big(1.0))) == BigFloat
    @test H._sh_coef_type(Dict{Tuple{Int,Int},Real}((2, 0) => 1)) == Float64
    a_any = H.spherical_harmonics_accel_PCPF(rvec, C_any, S_real, GM, R, nmax)
    @test norm(a_any - a_ref) < 1e-13 * norm(a_ref)

    # missing entries are treated as zero: J2-only field vs. closed-form J2 acceleration
    C20 = data["Cnm"][2, 0]
    C_j2 = Dict{Tuple{Int,Int},Any}((2, 0) => C20)
    S_j2 = Dict{Tuple{Int,Int},Any}()
    a_j2 = H.spherical_harmonics_accel_PCPF(rvec, C_j2, S_j2, GM, R, 2)
    x, y, z = rvec
    r = norm(rvec)
    J2 = -C20
    k = -1.5 * J2 * GM * R^2 / r^5
    a_j2_ref = k * [x * (1 - 5z^2 / r^2), y * (1 - 5z^2 / r^2), z * (3 - 5z^2 / r^2)]
    @test a_j2 ≈ a_j2_ref rtol = 1e-12
end


@testset "Loading an un-normalized gravity field" begin
    H = HighFidelityEphemerisModel
    src = joinpath(@__DIR__, "../data/earth/GGM03S.tab")
    nmax = 6
    ref = H.load_spherical_harmonics(src, nmax, false)
    field_ref = ref["FIELD"]

    mktempdir() do dir
        # Write the same field un-normalized (header column 6 = 0), using an independent
        # BigInt-factorial normalization factor.
        path = joinpath(dir, "unnormalized.tab")
        lines = readlines(src)
        header = strip.(split(lines[1], ","))
        header[6] = "0"
        open(path, "w") do io
            println(io, join(header, ", "))
            println(io, "")   # blank lines are skipped
            for n in 0:nmax, m in 0:n
                if n < 2
                    C, S = n == 0 ? 1.0 : 0.0, 0.0
                else
                    N = Float64(_pertD_Nnm(n, m))
                    C = N * field_ref.Cbar[n+1, m+1]
                    S = N * field_ref.Sbar[n+1, m+1]
                end
                println(io, "$n, $m, $(repr(C)), $(repr(S)), 0.0, 0.0")
            end
            # a higher-degree line stops reading
            println(io, "$(nmax + 1), 0, 1.0, 0.0, 0.0, 0.0")
        end

        data = H.load_spherical_harmonics(path, nmax, true)
        @test data["NORMALIZATION"] == 0
        @test data["GM"] == ref["GM"]
        @test data["REFERENCE RADIUS"] == ref["REFERENCE RADIUS"]
        @test data["FIELD"].Cbar ≈ field_ref.Cbar rtol = 1e-13 atol = 1e-25
        @test data["FIELD"].Sbar ≈ field_ref.Sbar rtol = 1e-13 atol = 1e-25
        # denormalized view returns the un-normalized coefficients that were written
        @test data["Cnm"][4, 3] ≈ Float64(_pertD_Nnm(4, 3)) * field_ref.Cbar[5, 4] rtol = 1e-13
        @test data["Snm"][6, 6] ≈ Float64(_pertD_Nnm(6, 6)) * field_ref.Sbar[7, 7] rtol = 1e-13

        rvec = [-2000.0, 1000.0, -6500.0]
        a_ref = H.spherical_harmonics_accel_PCPF(rvec, ref["Cnm"], ref["Snm"],
            ref["GM"], ref["REFERENCE RADIUS"], nmax)
        a_un = H.spherical_harmonics_accel_PCPF(rvec, data["Cnm"], data["Snm"],
            data["GM"], data["REFERENCE RADIUS"], nmax)
        @test norm(a_un - a_ref) < 1e-12 * norm(a_ref)

        # unsupported normalization flag
        bad = joinpath(dir, "bad_normalization.tab")
        header[6] = "2"
        open(bad, "w") do io
            println(io, join(header, ", "))
            println(io, "2, 0, 1.0, 0.0, 0.0, 0.0")
        end
        @test_throws ErrorException H.load_spherical_harmonics(bad, 2, true)
    end
end
