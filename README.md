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

Since this is a Julia package, you can install it by adding it to your Julia environment. First, make sure you have Julia installed (version 1.6 or higher recommended).

### From a local clone:

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd dynamic_rrt
   ```

2. Start Julia in the project directory and activate the environment:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```
