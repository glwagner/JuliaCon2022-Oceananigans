using Statistics
using JLD2
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Operators

using Oceananigans.Architectures: arch_array
using Oceananigans.TurbulenceClosures
using Oceananigans.Advection: VelocityStencil

#####
##### Define Parameters and load data
#####

latitude = (-84.375, 84.375)
Δφ = latitude[2] - latitude[1]

# 2.8125 degree resolution
Nx = 128
Ny = 60
Nz = 12
Lz = 3600

# Running on a GPU
arch = GPU()

#### Load in all the required data

file_bathymetry          = jldopen("bathymetry_juliacon.jld2")
file_boundary_conditions = jldopen("boundary_conditions_juliacon.jld2")
file_initial_conditions  = jldopen("initial_conditions_juliacon.jld2")

τˣ = file_boundary_conditions["τˣ"]
τʸ = file_boundary_conditions["τʸ"]
Tˢ = file_boundary_conditions["surface_T"]

bathymetry = file_bathymetry["bathymetry"]

### Move all required data to GPU!

bathymetry = arch_array(arch, bathymetry_data)
τˣ = arch_array(arch, τˣ)
τʸ = arch_array(arch, τʸ)
Tˢ = arch_array(arch, Tˢ)

#####
##### Build a Grid
#####

σ = 1.15 # linear stretching factor

Δz_center_linear(k)      = Lz * (σ - 1) * σ^(Nz - k) / (σ^Nz - 1) # k=1 is the bottom-most cell, k=Nz is the top cell
linearly_spaced_faces(k) = k==1 ? -Lz : - Lz + sum(Δz_center_linear.(1:k-1))

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (5, 5, 5),
                                              z = linearly_spaced_faces)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

#####
##### Physics 
#####

# Coriolis force
coriolis = HydrostaticSphericalCoriolis()

# Buoyancy
buoyancy = SeawaterBuoyancy(constant_salinity=30)

# Diffusivities (vertical, horizontal and convective adjustment)
vertical_diffusivity   = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = 1.0, κ = 1e-4)
horizontal_diffusivity = HorizontalScalarDiffusivity(ν = 1e+4, κ = 1e+3)
convective_adjustment  = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

closure = (vertical_diffusivity, horizontal_diffusivity, convective_adjustment)

#####
##### Numerical methods (advection, free surface)
#####

momentum_advection = WENO(vector_invariant = VelocityStencil())
tracer_advection   = WENO()

free_surface       = ImplicitFreeSurface()

#####
##### Boundary conditions / constant-in-time surface forcing
#####

λ = 0.001

@inline function surface_temperature_relaxation(i, j, grid, clock, fields, p)
    T_surface = fields.T[i, j, grid.Nz]    
    return p.λ * (T_surface - p.T★[i, j, 1])
end

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation, discrete_form = true, parameters = (λ = λ, T★ = Tˢ))

@inline wind_stress(i, j, grid, clock, fields, τ) = τ[i, j]

u_wind_stress_bc = FluxBoundaryCondition(wind_stress, discrete_form = true, parameters = τˣ)
v_wind_stress_bc = FluxBoundaryCondition(wind_stress, discrete_form = true, parameters = τʸ)

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

@inline u_immersed_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * fields.u[i, j, k]
@inline v_immersed_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * fields.v[i, j, k]

# Linear bottom drag:
μ = 0.001

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_immersed_drag_bc = FluxBoundaryCondition(u_immersed_drag, discrete_form = true, parameters = μ)
v_immersed_drag_bc = FluxBoundaryCondition(v_immersed_drag, discrete_form = true, parameters = μ)

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc, immersed = u_immersed_drag_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc, immersed = v_immersed_drag_bc)
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc)

boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs)

#####
##### Model Setup
#####

model = HydrostaticFreeSurfaceModel(;grid,
                                     tracers = :T,
                                     coriolis,
                                     buoyancy,
                                     closure,
                                     free_surface,
                                     momentum_advection,
                                     tracer_advection,
                                     boundary_conditions) 

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T

T_init = file_initial_conditions["T"] 
u_init = file_initial_conditions["u"] 
v_init = file_initial_conditions["v"]
η_init = file_initial_conditions["η"] .* 3.0

set!(model, u = u_init, v = v_init, η = η_init, T = T_init)

#####
##### Simulation setup
#####

Δt = 20minutes

simulation = Simulation(model, Δt = Δt, stop_time = 30days)

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.velocities.u
    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration,
                    maximum(u), 
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

u, v, w = model.velocities

T = model.tracers.T
η = model.free_surface.η

save_interval = 1days

output_prefix = "juliacon_ocean_simulation"

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, T, η),
                                                              schedule = TimeInterval(save_interval),
                                                              filename = output_prefix * "_surface",
                                                              indices = (:, :, grid.Nz),
                                                              overwrite_existing = true)

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(1year),
                                                        prefix = output_prefix * "_checkpoint",
                                                        overwrite_existing = true)

# Let's goo!
@info "Running with Δt = $(prettytime(simulation.Δt))"

run!(simulation)

@info """
    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""
