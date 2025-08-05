# DynamicRRT.jl

A Julia implementation of RRT* (Rapidly-exploring Random Trees) path planning algorithm with spatial hash maps for efficient navigation in dynamic environments.

## Overview

DynamicRRT.jl provides a high-performance implementation of the RRT* algorithm designed for path planning in environments with both static and dynamic (time-varying) obstacles. The package features:

- **Efficient spatial queries** using custom spatial hash maps for fast neighbor lookups
- **Dynamic obstacle support** for time-varying environments where obstacles move or appear/disappear over time
- **Dubins vehicle dynamics** for vehicles with constrained turning radius
- **Optimal path planning** with the RRT* algorithm that asymptotically converges to optimal solutions

The implementation is particularly well-suited for robotics applications requiring real-time path planning in complex, changing environments such as autonomous vehicle navigation, drone path planning, and mobile robot control.

## Installation

To add this package to your Julia project, open the Julia REPL, enter package mode by pressing `]`, and then run:

```julia
add https://github.com/MarshallVielmetti/DynamicRRT.jl.git
```

This will download the package and install it as a dependency in your current project's environment.

## Testing

To run the test suite for `DynamicRRT.jl`, navigate to the root of the package directory and run the following command in your terminal:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This will execute the tests defined in the `test/` directory and ensure that all components of the package are functioning correctly.

## Example

Here is a basic example of how to use `DynamicRRT.jl` to find a path for a Dubins vehicle in an environment with both static and dynamic obstacles.

```julia
using DynamicRRT
using StaticArrays
using Random

# Set a random seed for reproducibility
Random.seed!(0)

# Define the problem domain [min_bounds, max_bounds] for (x, y, θ)
min_domain = SVector(0.0, 0.0, -π)
max_domain = SVector(1.0, 1.0, π)
domain = (min_domain, max_domain)

# Create some random static obstacles
static_obstacles = [DynamicRRT.DubinsDynamicPathRRT.random_static_obstacle() for i=1:2]

# Create some dynamic obstacles with centers that move as a function of time
dynamic_obstacles = [
    DynamicRRT.DubinsDynamicPathRRT.DynamicCircleObstacle(
        t -> SVector(0.5 + 0.5 * sin(-t), 0.3), 
        0.1), 
    DynamicRRT.DubinsDynamicPathRRT.DynamicCircleObstacle(
        t -> SVector(sin(t + π/4), 0.6), 
        0.1), 
    DynamicRRT.DubinsDynamicPathRRT.DynamicCircleObstacle(
        t -> SVector(sin(t - π/4), 0.9), 
        0.1)
    ]

# Define the goal state and vehicle's turning radius
goal_state = SVector(1.0, 1.0, 0.0)
turning_radius = 0.1

# Create the RRT problem definition
rrt_problem = DynamicRRT.DubinsDynamicPathRRT.DubinsDynamicRRTProblem(domain, turning_radius, static_obstacles, dynamic_obstacles, goal_state)

# Define the start state (x, y, θ, t)
start_state = SVector(0.0, 0.0, 0.0, 0.0)

# Set up the RRT* solver
# The widths define the cell size for the spatial hash map (only for spatial dimensions)
widths = SVector(0.1, 0.1) 
solution = DynamicRRT.RRTStar.setup(rrt_problem, start_state, widths)

# Solve the RRT* problem
DynamicRRT.RRTStar.solve!(rrt_problem, solution; 
    max_iterations=1000, 
    max_time_seconds=1.0, 
    do_rewire=true, 
    early_exit=true,
    goal_bias=0.1);

# The best path can be accessed via solution.best_path
println("Best path cost: ", solution.best_path_cost)
println("Path found: ", !isnothing(solution.best_path))

```
