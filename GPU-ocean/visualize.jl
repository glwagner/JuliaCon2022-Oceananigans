using Oceananigans.Grids: on_architecture

function surface_bathymetry(grid)

    cpu_grid = on_architecture(CPU(), grid)
    bat = deepcopy(cpu_grid.immersed_boundary.bottom_height[1:grid.Nx, 1:grid.Ny])

    z = cpu_grid.zᵃᵃᶜ[1:grid.Nz]

    for i in 1:grid.Nx, j in 1:grid.Ny
        for k in 1:grid.Nz
            if z[k] < bat[i, j]
                bat2[i, j] = z[k]
            end
        end
    end

    return  bat2
end

function visualize_solution(output_prefix, grid)
    
    cpu_grid   = on_architecture(CPU(), grid)
    
    bathymetry_mask = deepcopy(cpu_grid.immersed_boundary.bottom_height[1:grid.Nx, 1:grid.Ny])
    bathymetry_mask[bathymetry_mask .> 0] .= NaN 
    bathymetry_mask[bathymetry_mask .< 0] .= 0.0

    surface_file    = jldopen(output_prefix * "_surface.jld2")

    iterations = parse.(Int, keys(surface_file["timeseries/t"]))

    iter = Observable(0)

    ηi(iter) = surface_file["timeseries/η/" * string(iter)][:, :, 1]       .+ bathymetry_mask
    ui(iter) = surface_file["timeseries/u/" * string(iter)][:, :, 1]       .+ bathymetry_mask
    vi(iter) = surface_file["timeseries/v/" * string(iter)][:, 1:end-1, 1] .+ bathymetry_mask
    Ti(iter) = surface_file["timeseries/T/" * string(iter)][:, :, 1]       .+ bathymetry_mask

    ti(iter) = string(surface_file["timeseries/t/" * string(iter)] / day)

    η = @lift ηi($iter) 
    u = @lift ui($iter)
    v = @lift vi($iter)
    T = @lift Ti($iter)

    max_η = 2
    min_η = - max_η
    max_u = 0.2
    min_u = - max_u
    max_T = 32
    min_T = 0

    fig = Figure(resolution = (1200, 900))

    ax = Axis(fig[1, 1], title="Free surface displacement (m)")
    hm = GLMakie.heatmap!(ax, η, colorrange=(min_η, max_η), colormap=:balance, nan_color = :black)
    cb = Colorbar(fig[1, 2], hm)

    ax = Axis(fig[2, 1], title="Sea surface temperature (ᵒC)")
    hm = GLMakie.heatmap!(ax, T, colorrange=(min_T, max_T), colormap=:thermal, nan_color = :black)
    cb = Colorbar(fig[2, 2], hm)

    ax = Axis(fig[1, 3], title="South-north surface velocity (m s⁻¹)")
    hm = GLMakie.heatmap!(ax, v, colorrange=(min_u, max_u), colormap=:balance, nan_color = :black)
    cb = Colorbar(fig[1, 4], hm)

    ax = Axis(fig[2, 3], title="West-east surface velocity (m s⁻¹)")
    hm = GLMakie.heatmap!(ax, u, colorrange=(min_u, max_u), colormap=:balance, nan_color = :black)
    cb = Colorbar(fig[2, 4], hm)

    title_str = @lift "Earth day = " * ti($iter)
    ax_t = fig[0, :] = Label(fig, title_str)

    GLMakie.record(fig, output_prefix * ".mp4", iterations, framerate=8) do i
        @info "Plotting iteration $i of $(iterations[end])..."
        iter[] = i
    end

    display(fig)

    close(surface_file)
end