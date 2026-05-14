"""
generate_data.jl
================
Exact diagonalisation of the 2D Bose-Hubbard model on an Lx×Ly square lattice
with periodic boundaries.  For each (filling N, U/t) pair the ground-state
wavefunction |ψ₀⟩ is computed in the Fock basis and written to CSV.

Configuration
-------------
  Edit Lx, Ly, and FILLINGS below.  FILLINGS is a list of integer particle
  numbers; [M_SITES] gives unit filling, [M_SITES, 2*M_SITES, 3*M_SITES] gives
  lobes n = 1, 2, 3.  Any filling whose Hilbert-space dimension exceeds D_MAX
  is skipped automatically.

  Hilbert-space dimension:  D = C(N + M − 1, N)
    2×3 lattice  n=1 (N=6):   D =   462
    2×3 lattice  n=2 (N=12):  D =  6188
    2×3 lattice  n=3 (N=18):  D = 33649
    3×3 lattice  n=1 (N=9):   D = 24310

Output (one file per filling)
------------------------------
  ../output/bose_hubbard_psi_Lx{Lx}_Ly{Ly}_n{N}.csv
  Columns:  U_over_t | psi_0 | psi_1 | … | psi_{D-1}
"""

using LatticeModels
using LinearAlgebra
using Arpack
using CSV
using DataFrames
using Printf

# ─── Configuration ────────────────────────────────────────────────────────────
const Lx, Ly  = 2, 3
const M_SITES = Lx * Ly

# Integer particle numbers to sweep.  Add multiples of M_SITES for more lobes.
const FILLINGS = [M_SITES*2, M_SITES*3]
const U_MIN = 0.5
const U_MAX = 40.0
const N_U   = 1000

# Skip fillings where dense ED would be too expensive
const D_MAX = 50_000


# ─── Hilbert-space dimension  D = C(N + M − 1, N) ────────────────────────────
function hilbert_dim(N::Int, M::Int)::Int
    Int(binomial(big(N + M - 1), big(N)))
end


# ─── Hamiltonian ──────────────────────────────────────────────────────────────
function build_hamiltonian(lattice, N_part::Int, U_over_t::Float64)
    Symmetric(real.(bosehubbard(lattice, N_part; U=U_over_t, t1=-1.0).data))
end


# ─── Ground state ─────────────────────────────────────────────────────────────
# Canonical sign convention: component with largest |ψᵢ| is forced positive.
function ground_state_wavefunction(H::AbstractMatrix{Float64})::Vector{Float64}
    E0, Ψ0 = eigs(H; nev=1, which=:SR, tol=eps(Float64))
    Ψ0[:, 1] .*= sign(Ψ0[argmax(abs.(Ψ0[:, 1])), 1])
    return Ψ0[:,1] ./ norm(Ψ0[:, 1])
end



# ─── Sweep U/t for one particle number ────────────────────────────────────────
function sweep_filling(lattice, N_part::Int, outdir::String)
    D = hilbert_dim(N_part, M_SITES)
    n = N_part / M_SITES
    println()
    println("=" ^ 60)
    @printf(" N = %d  (n = %d/%d = %.4f)  |  D = %d\n",
            N_part, N_part, M_SITES, n, D)
    println("=" ^ 60)

    if D > D_MAX
        @printf("  Skipping — D = %d > D_MAX = %d\n\n", D, D_MAX)
        return
    end

    add = 0.0
    if n > 1.0
        add += 2 * (n^2 + n)
    end
    U_values = range(U_MIN, U_MAX+add, length=N_U)
    U_out = Vector{Float64}(undef, N_U)
    Ψ_out = Matrix{Float64}(undef, N_U, D)

    for (k, U) in enumerate(U_values)
        H  = build_hamiltonian(lattice, N_part, Float64(U))
        ψ  = ground_state_wavefunction(H)
        U_out[k]    = Float64(U)
        Ψ_out[k, :] = ψ

        if k == 1 || k % 50 == 0 || k == N_U
            @printf("  [%3d/%d]  U/t = %6.3f   ‖ψ‖ = %.8f\n",
                    k, N_U, Float64(U), norm(ψ))
        end
    end

    psi_names = ["psi_$i" for i in 0:D-1]
    df = DataFrame(Ψ_out, psi_names)
    insertcols!(df, 1, :U_over_t => U_out)

    fname   = @sprintf("bose_hubbard_psi_Lx%d_Ly%d_n%d.csv", Lx, Ly, N_part)
    outpath = joinpath(outdir, fname)
    CSV.write(outpath, df)

    @printf("  Written : %s\n", outpath)
    @printf("  Shape   : %d rows × %d columns  (1 U/t + D=%d amplitudes)\n",
            nrow(df), ncol(df), D)
end


# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    lattice = SquareLattice(Lx, Ly, boundaries=(:axis1 => true, :axis2 => true))

    println("=" ^ 60)
    println(" Bose-Hubbard ED  |  $(Lx)×$(Ly) PBC lattice")
    println(" Fillings (N) : $(FILLINGS)")
    println("=" ^ 60)

    outdir = joinpath(@__DIR__, "..", "output")
    mkpath(outdir)

    for N_part in FILLINGS
        sweep_filling(lattice, N_part, outdir)
    end

    println("\nAll done.\n")
end

main()
