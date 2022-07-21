
using Oceananigans
using Oceananigans.Utils
using Oceananigans.Utils: Reference

grid = RectilinearGrid(GPU(), size = (8, 1, 1), extent = (1, 1, 1))

mrg_serial   = MultiRegionGrid(grid, partition = XPartition(2), devices = (0, 0))
mrg_parallel = MultiRegionGrid(grid, partition = XPartition(2), devices = (0, 1))

# This model is serial (has one region)
serial_model_one_region  = HydrostaticFreeSurfaceModel(; grid)

# This model has two regions running in serial
serial_model_two_regions = HydrostaticFreeSurfaceModel(grid = mrg_serial)

# This model has two regions running in parallel (on devices 0 and 1)
parallel_model_two_regions = HydrostaticFreeSurfaceModel(grid = mrg_parallel)


using CUDA

f(obj) = CUDA.device().handle + 1

@apply_regionally h = f(serial_model_one_region)
@apply_regionally h = f(serial_model_two_regions)
@apply_regionally h = f(parallel_model_two_regions)

@apply_regionally h = f(Reference(parallel_model_two_regions))
 
set!(serial_model_one_region,    T = (x, y, z) -> f(x))
set!(serial_model_two_regions,   T = (x, y, z) -> f(x))
set!(parallel_model_two_regions, T = (x, y, z) -> f(x))