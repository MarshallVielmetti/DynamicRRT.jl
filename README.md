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
