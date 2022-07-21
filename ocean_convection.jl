using Oceananigans
using GLMakie

# Model
grid = RectilinearGrid(size=(64, 64), x=(0, 128), z=(-64, 0),
                       topology=(Periodic, Flat, Bounded))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = WENO5(),
                            tracers = :b,
                            closure = ScalarDiffusivity(ν=1e-1, κ=1e-1),
                            buoyancy = BuoyancyTracer())

Δb = 1
step(z, z₀, δ) = 1/2 * (tanh((z - z₀) / δ) + 1)
bᵢ(x, y, z) = - Δb * step(z, -32, 4) + 1e-2 * ϵ(x, y, z) 
ϵ(x, y, z) = randn()
set!(model, b=bᵢ)

# Simulation
simulation = Simulation(model, Δt=1e-1, stop_time=200)

progress(sim) = @info string("Iter: ", iteration(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

filename = "ocean_convection.jld2"
outputs = model.tracers
simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs; filename,
                                                    schedule = TimeInterval(0.1),
                                                    overwrite_existing = true)

run!(simulation)

# Analysis
bt = FieldTimeSeries(filename, "b")
t = bt.times
Nt = length(t)

fig = Figure(resolution=(1200, 800))
ax = Axis(fig[1, 1], xlabel="x", ylabel="z")
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value

b = @lift interior(bt[$n], :, 1, :)
heatmap!(ax, b)

display(fig)

