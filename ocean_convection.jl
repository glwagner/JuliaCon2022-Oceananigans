using Oceananigans
using Oceananigans.Units
using GLMakie

#####
##### Model setup
#####

# A two-dimensional grid
grid = RectilinearGrid(size=(128, 128), x=(0, 128), z=(-64, 0),
                       topology=(Periodic, Flat, Bounded))

# Nonhydrotatic model of a buoyancy, viscous fluid with
# high-order numerics for advection
model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = WENO(),
                            tracers = :b,
                            closure = ScalarDiffusivity(ν=1e-1, κ=1e-1),
                            buoyancy = BuoyancyTracer())

# Initial condition: cold fluid over hot fluid
z₀ = -32 # m
δz = 4 # m
Δb = 1 # m s⁻²
ϵ(x, y, z) = randn()
bᵢ(x, y, z) = - Δb * tanh((z - z₀) / δz) + 1e-2 * ϵ(x, y, z) 
set!(model, b=bᵢ)

#####
##### Simulation
#####

# Simulation with initial time-step of 0.1 seconds
simulation = Simulation(model, Δt=0.1, stop_time=2minutes)

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Logging via callback
progress(sim) = @info string("Iter: ", iteration(sim),
                             ", time: ", prettytime(sim),
                             ", Δt: ", prettytime(sim.Δt))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# Some non-trivial output: buoyancy and vorticity
filename = "ocean_convection.jld2"

u, v, w = model.velocities
ξ = ∂z(w) - ∂x(u) # abstract operation

b = model.tracers.b

outputs = (; b, ξ)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs; filename,
                                                    schedule = TimeInterval(0.1),
                                                    overwrite_existing = true)

run!(simulation)

#####
##### Analysis / visualization with Makie
#####

bt = FieldTimeSeries(filename, "b")
ξt = FieldTimeSeries(filename, "ξ")
t = bt.times
Nt = length(t)

fig = Figure(resolution=(1100, 1200))
axb = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", title="Buoyancy", aspect=2)
axξ = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)", title="Vorticity", aspect=2)

slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Convective instability at t = ", prettytime(t[$n]))
Label(fig[1, 1:2], title, tellwidth=false)

b = @lift interior(bt[$n], :, 1, :)
ξ = @lift interior(ξt[$n], :, 1, :)

x, y, z = nodes(bt)
hm = heatmap!(axb, x, z, b, colormap=:thermal, colorrange=(-0.5, 0.5))
Colorbar(fig[2, 2], hm)

x, y, z = nodes(ξt)
ξlim = maximum(abs, interior(ξt)) / 2
hm = heatmap!(axξ, x, z, ξ, colormap=:balance, colorrange=(-ξlim, ξlim))
Colorbar(fig[3, 2], hm)

display(fig)

record(fig, "ocean_convection.mp4", 1:Nt, framerate=24) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

