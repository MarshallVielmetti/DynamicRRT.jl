"""
    DubinsDynamicPathRRT

This module provides a concrete implementation of the `RRTStar.AbstractProblem` for a Dubins car model operating in an environment with both static and dynamic obstacles.

The state is time-augmented, represented as `SVector{4,F}` for `(x, y, θ, t)`, where `t` is the time elapsed along the path from the start. This allows for planning in dynamic environments where obstacles move over time.

This module defines:
- Obstacle types: `CircleObstacle` (static) and `DynamicCircleObstacle`.
- The `DubinsDynamicRRTProblem` struct to define the planning problem.
- Implementations for all required functions from the `RRTStar` interface, such as `steer`, `collision_free`, `nearest`, etc., tailored for Dubins dynamics and time-varying collision checking.
- A suite of plotting utilities for visualizing the RRT* tree, paths, and obstacles.
"""
module DubinsDynamicPathRRT

using ..RRTStar
using ..SpatialHashMaps
using StaticArrays, LinearAlgebra
using Dubins

"""
Abstract base type for all obstacles.
"""
abstract type Obstacle{F} end

"""
    CircleObstacle{F}

Represents a static circular obstacle in the 2D plane.

# Fields
- `center::SVector{2,F}`: The center coordinates of the circle.
- `radius::F`: The radius of the circle.
"""
struct CircleObstacle{F} <: Obstacle{F}
    center::SVector{2,F}
    radius::F
end

"""
    DynamicCircleObstacle{C,F,T}

Represents a dynamic circular obstacle whose center moves over time.

# Fields
- `center::C`: A function `t -> SVector{2,F}` that returns the center of the obstacle at a given time `t`.
- `radius::F`: The radius of the circle.
- `trange::T`: A tuple `(t_start, t_end)` representing the time interval during which the obstacle exists.
"""
struct DynamicCircleObstacle{C,F,T} <: Obstacle{F}
    center::C
    radius::F
    trange::T
end
DynamicCircleObstacle(c, r) = DynamicCircleObstacle(c, r, (-Inf, Inf))

"""
    random_static_obstacle(radius=0.2)

Generates a `CircleObstacle` with a random center and radius within the unit square.
"""
function random_static_obstacle(radius=0.2)
    c = @SVector rand(2)
    r = rand() * radius
    return CircleObstacle(c, r)
end

"""
    DubinsDynamicRRTProblem{F,SO,DO}

Defines the RRT* problem for a Dubins vehicle in a dynamic environment. This is a concrete implementation of `RRTStar.AbstractProblem`.

The state for this problem is a 4D vector `SVector{4,F}` representing `(x, y, θ, t)`, where `t` is the time.

# Fields
- `domain::Tuple{SVector{3,F},SVector{3,F}}`: The configuration space domain as `(min_bounds, max_bounds)` for `(x, y, θ)`.
- `turning_radius::F`: The minimum turning radius of the Dubins vehicle.
- `static_obstacles::Vector{SO}`: A vector of static obstacles.
- `dynamic_obstacles::Vector{DO}`: A vector of dynamic obstacles.
- `goal_state::Union{Nothing,SVector{3,F}}`: The goal configuration `(x, y, θ)`.
"""
struct DubinsDynamicRRTProblem{F,SO,DO} <: RRTStar.AbstractProblem{SVector{4,F}}
    domain::Tuple{SVector{3,F},SVector{3,F}}
    turning_radius::F
    static_obstacles::Vector{SO}
    dynamic_obstacles::Vector{DO}
    goal_state::Union{Nothing,SVector{3,F}}
end

DynamicRRTProblem(domain, ρ, static_obs, dynamic_obs) = DubinsDynamicRRTProblem(domain, ρ, static_obs, dynamic_obs, nothing)

"""
    sample_domain(P::DubinsDynamicRRTProblem)

Samples a random configuration `(x, y, θ)` from the problem's domain.
"""
function sample_domain(P::DubinsDynamicRRTProblem)
    v = @SVector rand(3)
    q = P.domain[1] + (P.domain[2] - P.domain[1]) .* v
end

"""
    is_colliding(o::CircleObstacle, q::SVector)

Checks if a 2D point `q` collides with a static `CircleObstacle`.
"""
function is_colliding(o::CircleObstacle, q::SVector)
    # this does not need the time or angle check
    return norm(o.center - q[SOneTo(2)]) <= o.radius
end

"""
    is_colliding(o::DynamicCircleObstacle, q::SVector{4,F})

Checks if a state `q = (x, y, θ, t)` collides with a `DynamicCircleObstacle` at time `t`.
"""
function is_colliding(o::DynamicCircleObstacle, q::SVector{4,F}) where {F}
    t = q[4]

    if !(o.trange[1] <= t <= o.trange[2])
        # obstacle does not exist at this time
        return false
    end

    # now check position constraints
    x = q[SOneTo(2)]
    return norm(o.center(t) - x) <= o.radius
end

"""
    is_colliding(obstacles::Vector{O}, q)

Checks if a state `q` collides with any obstacle in a given vector of obstacles.
"""
function is_colliding(obstacles::Vector{O}, q) where {O}
    for o in obstacles
        if is_colliding(o, q)
            return true
        end
    end
    return false
end

"""
    RRTStar.sample_free(problem::DubinsDynamicRRTProblem; hash_map=nothing, goal_bias=0.1)

Samples a random, collision-free configuration from the state space.
With a probability `goal_bias`, it samples the `goal_state`. Otherwise, it samples from the problem domain,
ensuring the sample is not inside a static obstacle.
"""
function RRTStar.sample_free(problem::DubinsDynamicRRTProblem; hash_map=nothing, goal_bias=0.1)

    while true

        sample_goal = (goal_bias > 0) && (rand() < goal_bias)
        if sample_goal && !isnothing(problem.goal_state)
            q = problem.goal_state
        else
            q = sample_domain(problem)
        end

        # check if q is in a static obstacle
        if !is_colliding(problem.static_obstacles, q)
            return q
        end
    end

end

"""
    generate_path(problem, q1, q2)

Generates the shortest Dubins path between two configurations `q1` and `q2`.
"""
function generate_path(problem, q1, q2)
    errcode, path = dubins_shortest_path(q1, q2, problem.turning_radius)
    @assert errcode == Dubins.EDUBOK "Dubins path generation failed with code $(errcode)"
    return path
end

"""
    dubins_dist(problem, q1, q2)

Computes the length of the shortest Dubins path between two configurations `q1` and `q2`.
"""
function dubins_dist(problem, q1, q2)
    path = generate_path(problem, q1, q2)
    L = dubins_path_length(path)
    return L
end

"""
    RRTStar.nearest(problem::DubinsDynamicRRTProblem, hash_map, xt_rand)

Finds the node in the `hash_map` that is nearest to the random state `xt_rand`
using the Dubins path length as the distance metric.

Note: This implementation performs a brute-force search over all nodes. A more
efficient version would use the `SpatialHashMap` to find a candidate set of near
nodes first, then compute the exact Dubins distance for that smaller set.
"""
function RRTStar.nearest(problem::DubinsDynamicRRTProblem, hash_map, xt_rand)

    x_rand = xt_rand[SOneTo(3)]

    i_nearest = 0
    d_nearest = Inf

    # TODO: Use spatial hash map to find a candidate set of near nodes first.
    for i in 1:length(hash_map)
        if !RRTStar.is_invalid(hash_map[i])
            x = (hash_map[i].state)[SOneTo(3)]
            d = dubins_dist(problem, x, x_rand)
            if d < d_nearest
                i_nearest = i
                d_nearest = d
            end
        end
    end
    return CartesianIndex(i_nearest)
end

"""
    RRTStar.near(problem::DubinsDynamicRRTProblem, hash_map, x_new)

Returns a set of indices of nodes in the `hash_map` that are considered "near" the new state `x_new`.

Note: This implementation considers all nodes to be "near". A more efficient implementation
would use the `SpatialHashMap` to return only nodes within a certain radius.
"""
function RRTStar.near(problem::DubinsDynamicRRTProblem, hash_map, x_new)
    # TODO: be more judicious about which nodes are nearby using the spatial hash map.
    # r = 0.2 # radius to look around in
    # i_neighbors = SpatialHashMaps.query_radius(hash_map, x_new[SOneTo(2)], r)
    # return i_neighbors
    return CartesianIndices(1:length(hash_map))
end

"""
    RRTStar.steer(problem::DubinsDynamicRRTProblem, q_nearest::SVector{4,F}, q_rand::SVector, distance_fraction=1.0) where {F}

Steers from `q_nearest` towards `q_rand` for a given `distance_fraction` of the total Dubins path length.
The state is time-augmented, so `q = (x, y, θ, t)`. The cost of the new edge is the length of the path segment,
which is equivalent to the time elapsed.

Returns `(q_new, cost)`, where `q_new` is the new state and `cost` is the path length.
"""
function RRTStar.steer(problem::DubinsDynamicRRTProblem, q_nearest::SVector{4,F}, q_rand::SVector, distance_fraction=1.0) where {F}

    t_nearest = q_nearest[4]
    x_nearest = q_nearest[SOneTo(3)]
    x_rand = q_rand[SOneTo(3)]

    # construct a path from x_nearest to x_rand
    path = generate_path(problem, x_nearest, x_rand)

    # get a subpath based on the length
    L = dubins_path_length(path)

    errcode, subpath = dubins_extract_subpath(path, distance_fraction * L)
    @assert errcode == Dubins.EDUBOK

    path = subpath
    L_new = dubins_path_length(path)

    # compute the new final state
    errcode, x_new = dubins_path_endpoint(path)

    t_new = t_nearest + L_new

    q_new = SVector(x_new..., t_new)

    # get the total path length
    return q_new, L_new
end

"""
    RRTStar.collision_free(problem, q_nearest, q_new; step_size=0.001)

Checks if the Dubins path between `q_nearest` and `q_new` is collision-free.
It samples points along the path and checks for collisions with both static and dynamic obstacles.
The state `q` is time-augmented, `q = (x, y, θ, t)`.
"""
function RRTStar.collision_free(
    problem::DubinsDynamicRRTProblem,
    q_nearest, # time indexed
    q_new; # time indexed
    step_size=0.001,
)

    # extract variables
    t_nearest = q_nearest[4]
    x_nearest = q_nearest[SOneTo(3)]
    x_new = q_new[SOneTo(3)]

    # get the path
    path = generate_path(problem, x_nearest, x_new)
    L = dubins_path_length(path)

    # Discretize the path and check for collisions at each step
    for t_path in range(0, stop=L, step=step_size)
        errcode, x = dubins_path_sample(path, t_path)
        @assert errcode == Dubins.EDUBOK

        # The state includes the time component for dynamic collision checking
        q = SVector(x..., t_nearest + t_path)
        if is_colliding(problem.static_obstacles, q)
            return false
        end
        if is_colliding(problem.dynamic_obstacles, q)
            return false
        end
    end
    return true
end

"""
    RRTStar.spatial_position(problem::DubinsDynamicRRTProblem, x::T)

Extracts the 2D spatial position `(x, y)` from a full state vector `x`.
This is used for insertion and querying in the `SpatialHashMap`.
"""
function RRTStar.spatial_position(problem::DubinsDynamicRRTProblem, x::T) where {T}
    return x[SOneTo(2)]
end

"""
    RRTStar.goal_reachable(problem::DubinsDynamicRRTProblem, hash_map, i_new)

Checks if the goal is reachable from the new node `i_new`.
For this problem, it checks if the 2D position of the new node is within a certain
tolerance of the goal's 2D position.
"""
function RRTStar.goal_reachable(problem::DubinsDynamicRRTProblem, hash_map, i_new)
    curr_state = hash_map[i_new].state
    return RRTStar.collision_free(problem, curr_state, problem.goal_state)

    if isnothing(problem.goal_state)
        return false
    end

    n_new = hash_map[i_new]
    x_new = RRTStar.spatial_position(problem, n_new.state)

    # TODO: This should also check the angle and potentially the final path segment for collisions.
    # TODO: Use a tolerance paramter?
    # TODO: This should just change...should be if there exists a collision free path from the current
    # TODO: pose to the goal imo.
    if norm(problem.goal_state[SOneTo(2)] - x_new[SOneTo(2)]) <= 0.01
        return true
    else
        return false
    end
end

"""
    RRTStar.cost_to_goal(problem::DubinsDynamicRRTProblem, node::RRTStar.Node)

Calculates the estimated cost from a node to the goal. For Dubins path, this is
the Dubins distance from the node's state to the goal state.
"""
function RRTStar.cost_to_goal(problem::DubinsDynamicRRTProblem, node::RRTStar.Node)
    if isnothing(problem.goal_state)
        return Inf
    end
    return dubins_dist(problem, node.state[SOneTo(3)], problem.goal_state)
end

end # module DubinsDynamicPathRRT