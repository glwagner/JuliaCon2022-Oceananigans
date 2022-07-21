using Oceananigans
using GLMakie
using JLD2

arch = GPU()

Nx = 128
Ny = 60
Nz = 12

Lz = 3600

σ = 1.15
Δz(k) = 3600 * (σ - 1) * σ^(Nz - k) / (σ^Nz - 1)

z_faces(k) = k == 1 ? -Lz : -Lz + sum(Δz.(1:k-1))


underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, Nz),
                                              latitude = (-84.375, 84.375), 
                                              longitude = (-180, 180),
                                              z = z_faces)

bathymetry = jldopen("bathymetry_juliacon.jld2")["bathymetry"]

# Overlay a bottom to the grid (everything is masked to zero below bathymetry[i, j])

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))


### Physics

coriolis = HydrostaticSphericalCoriolis()
buoyancy = SeawaterBuoyancy()

### Vertical and Horizontal Diffusion

horizontal_diffusion = HorizontalScalarDiffusivity(ν = 1e+5, κ = 1e+2)
vertical_diffusion   = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = 1, κ = 1e-3)

### Convective Adjustment

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

##
## Boundary Conditions (u, v, T, S)
##

file_boundary = jldopen("boundary_conditions_juliacon.jld2")

# 2D Arrays dimensions (Nx, Ny)
τx = file_boundary["τˣ"];
τy = file_boundary["τʸ"];
Ts = file_boundary["surface_T"];

# ValueBoundaryCondition, FluxBoundaryCondition

u_top_bc = FluxBoundaryCondition(arch_array(arch, τx))
v_top_bc = FluxBoundaryCondition(arch_array(arch, τy))

@inline function restoring_T(i, j, grid, clock, fields, parameters)
    
    T_target  = parameters.Ts[i, j]
    T_surface = fields.T[i, j, grid.Nz]

    return parameters.λ * (T_surface - T_target)
end

Ts = arch_array(arch, Ts)

T_top_bc = FluxBoundaryCondition(restoring_T, discrete_form = true, parameters = (Ts = Ts, λ = 0.001))


@inline speed(i, j, k, grid, u, v) = (u[i, j, k]^2 + v[i, j, k]^2)^(0.5)

@inline u_linear_drag(i, j, grid, clock, fields, Cd) = - Cd * fields.u[i, j, 1] * speed(i, j, 1, grid, fields.u, fields.v)
@inline v_linear_drag(i, j, grid, clock, fields, Cd) = - Cd * fields.v[i, j, 1] * speed(i, j, 1, grid, fields.u, fields.v)

u_bottom_bc = FluxBoundaryCondition(u_linear_drag, discrete_form=true, parameters = 0.01)
v_bottom_bc = FluxBoundaryCondition(v_linear_drag, discrete_form=true, parameters = 0.01)

u_bcs = FieldBoundaryConditions(top = u_top_bc, bottom = u_bottom_bc)
v_bcs = FieldBoundaryConditions(top = v_top_bc, bottom = v_bottom_bc)
T_bcs = FieldBoundaryConditions(top = T_top_bc)

### Model!!!!!

model = HydrostaticFreeSurfaceModel(; grid,
                                      coriolis,
                                      free_surface = ImplicitFreeSurface(),
                                      buoyancy,
                                      tracers = (:T, :S),
                                      tracer_advection = WENO5(grid),
                                      closure = (vertical_diffusion, horizontal_diffusion, convective_adjustment),
                                      boundary_conditions = (u = u_bcs, v = v_bcs, T = T_bcs))
                                      
set!(model, T = -1.0, S = 35.0)

using Oceananigans.Units
using Oceananigans.Utils

Δt        = 20minutes
stop_time = 30days

simulation = Simulation(model; Δt, stop_time)

@inline progress(sim) = @info "Iteration: $(sim.model.clock.iteration), Time: $(prettytime(sim.model.clock.time)), max(|u|): $(maximum(abs, sim.model.velocities.u)), max(|T|): $(maximum(abs, sim.model.tracers.T))"

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)

