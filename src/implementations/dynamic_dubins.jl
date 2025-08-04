module DubinsDynamicPathRRT

using ..RRTStar
using ..SpatialHashMaps
using StaticArrays, LinearAlgebra
using Dubins

abstract type Obstacle{F} end

struct CircleObstacle{F} <: Obstacle{F}
    center::SVector{2,F}
    radius::F
end

struct DynamicCircleObstacle{C,F,T} <: Obstacle{F}
    center::C
    radius::F
    trange::T
end
DynamicCircleObstacle(c, r) = DynamicCircleObstacle(c, r, (-Inf, Inf))

function random_static_obstacle(radius=0.2)
    c = @SVector rand(2)
    r = rand() * radius
    return CircleObstacle(c, r)
end

struct DubinsDynamicRRTProblem{F,SO,DO} <: RRTStar.AbstractProblem{SVector{2,F}}
    domain::Tuple{SVector{3,F},SVector{3,F}}
    turning_radius::F
    static_obstacles::Vector{SO}
    dynamic_obstacles::Vector{DO}
    goal_state::Union{Nothing,SVector{3,F}}
end

DynamicRRTProblem(domain, ρ, static_obs, dynamic_obs) = DynamicRRTProblem(domain, ρ, static_obs, dynamic_obs, nothing)

function sample_domain(P::DubinsDynamicRRTProblem)
    v = @SVector rand(3)
    q = P.domain[1] + (P.domain[2] - P.domain[1]) .* v
end

function is_colliding(o::CircleObstacle, q::SVector)
    # this does not need the time or angle check
    return norm(o.center - q[SOneTo(2)]) <= o.radius
end

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

function is_colliding(obstacles::Vector{O}, q) where {O}
    for o in obstacles
        if is_colliding(o, q)
            return true
        end
    end
    return false
end

"""
    sample_free(problem::DubinsDynamicRRTProblem)

returns a random state within the problem domain that is not occupied by static obstacles
"""
function RRTStar.sample_free(problem::DubinsDynamicRRTProblem; hash_map=nothing, goal_bias=0.1)

    while true

        sample_goal = (goal_bias > 0) && (rand() < goal_bias)
        if sample_goal
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

function generate_path(problem, q1, q2)
    errcode, path = dubins_shortest_path(q1, q2, problem.turning_radius)
    @assert errcode == Dubins.EDUBOK
    return path
end

function dubins_dist(problem, q1, q2)
    path = generate_path(problem, q1, q2)
    L = dubins_path_length(path)
    return L
end

"""
    nearest(problem, nodes, x_rand)

returns the index of the nearest node within the spatial hashmap of `nodes` to the state `x_rand`, according to a dubins path distance.
"""
function RRTStar.nearest(problem::DubinsDynamicRRTProblem, hash_map, xt_rand)

    x_rand = xt_rand[SOneTo(3)]

    i_nearest = 0
    d_nearest = Inf

    # for now, just go through all the nodes
    for i in 1:length(hash_map)
        x = (hash_map[i].state)[SOneTo(3)]
        d = dubins_dist(problem, x, x_rand)
        if d < d_nearest
            i_nearest = i
            d_nearest = d
        end
    end
    # i_nearest = SpatialHashMaps.query_nearest(hash_map, x_rand[SOneTo(2)])
    # x_nearest = hash_map[i_nearest].state
    # d_nearest = dubins_dist(problem, x_nearest, x2)
    # r = 0.2 # radius to look around in

    # I_near = SpatialHashMaps.query_radius(hash_map, x_new[SOneTo(2)], r)

    # for i_near in I_near



    return CartesianIndex(i_nearest)
end

"""
        near(problem, nodes, x_new)

returns the set of nearby nodes to a state `x_new`. 
"""
function RRTStar.near(problem::DubinsDynamicRRTProblem, hash_map, x_new)
    # todo: be more judicious about which nodes are nearby
    # return hash_map

    # r = 0.2 # radius to look around in

    # i_neighbors = SpatialHashMaps.query_radius(hash_map, x_new[SOneTo(2)], r)

    # return i_neighbors
    return CartesianIndices(1:length(hash_map))
end

"""
    (x_new, stage_cost) = steer(problem, x_nearest, x_rand, distance_fraction=0.2)

returns the (state, cost) of travelling upto `distance_fraction` towards `x_rand` starting from `x_nearest`.
It is safe to assume distance fraction is in [0, 1]. 
"""
function RRTStar.steer(problem::DubinsDynamicRRTProblem, q_nearest::SVector{4,F}, q_rand::SVector, distance_fraction=0.2) where {F}

    t_nearest = q_nearest[4]
    x_nearest = q_nearest[SOneTo(3)]
    x_rand = q_rand[SOneTo(3)]

    # construct a path from x_nearest to x_rand
    path = generate_path(problem, x_nearest, x_rand)

    # get a subpath based on the length
    L = dubins_path_length(path)
    # if distance_fraction < 1
    errcode, subpath = dubins_extract_subpath(path, distance_fraction * L)
    @assert errcode == Dubins.EDUBOK
    path = subpath
    L = dubins_path_length(path)
    # end

    # compute the new final state
    errcode, x_new = dubins_path_endpoint(path)

    t_new = t_nearest + L

    q_new = SVector(x_new..., t_new)

    # get the total path length
    return q_new, L
end

"""
    collision_free(problem, x_nearest, x_new; step_size=0.001)

checks if a path from `x_nearest` to `x_new` is collision free with respect to the wezes in `problem.wezes`.
"""
function RRTStar.collision_free(
    problem::DubinsDynamicRRTProblem,
    q_nearest, # time indexed
    q_new; # time indexed
    step_size=0.001,
)

    # extract variables
    t_nearest = q_nearest[4]
    # t_new = q_new[4]

    x_nearest = q_nearest[SOneTo(3)]
    x_new = q_new[SOneTo(3)]

    # get the path 
    path = generate_path(problem, x_nearest, x_new)

    L = dubins_path_length(path)

    for t in range(0, stop=L, step=step_size)
        errcode, x = dubins_path_sample(path, t)
        @assert errcode == Dubins.EDUBOK

        q = SVector(x..., t_nearest + t)
        if is_colliding(problem.static_obstacles, q)
            return false
        end
        if is_colliding(problem.dynamic_obstacles, q)
            return false
        end
    end
    return true
end


function RRTStar.spatial_position(problem::DubinsDynamicRRTProblem, x::T) where {T}
    return x[SOneTo(2)]
end

function RRTStar.goal_reachable(problem::DubinsDynamicRRTProblem, hash_map, i_new)

    # return false # for now :D

    if isnothing(problem.goal_state)
        return false
    end

    n_new = hash_map[i_new]
    x_new = RRTStar.spatial_position(problem, n_new.state)

    if norm(problem.goal_state[SOneTo(2)] - x_new[SOneTo(2)]) <= 0.1 #TODO(dev): todo fix this
        return true
    else
        return false
    end

end

end




function plot_node!(node; kwargs...)
    x, y = node.state
    scatter!([x], [y], label=false; kwargs...)
end

function plot_tree!(problem, solution, i_root_node=1; kwargs...)

    # xs = Float64[]
    # ys = Float64[]

    queue = [i_root_node,] # should be a deque

    while length(queue) > 0
        i_node = popfirst!(queue)
        n_node = solution.hash_map[i_node]

        if RRTStar.is_invalid(n_node)
            continue
        end

        for i_child in n_node.children
            # dereference the obj
            n_child = solution.hash_map[i_child]

            if RRTStar.is_invalid(n_child)
                continue
            end

            # ok these are valid parent child pair
            # get the path between them
            x_node = n_node.state[SOneTo(3)]
            x_child = n_child.state[SOneTo(3)]
            path = DubinsDynamicPathRRT.generate_path(problem, x_node, x_child)

            # sample the path at some places
            errcode, xs = Dubins.dubins_path_sample_many(path, 0.01)

            # plot the path 
            plot!([x[1] for x in xs], [x[2] for x in xs], label=false, opacity=0.2, linecolor=:gray, markersize=0.2; kwargs...)

            # add the child to the queue
            push!(queue, i_child)
        end
    end

    # now plot all the nodes in the hashmap
    scatter!([n.state[1] for n in solution.hash_map], [n.state[2] for n in solution.hash_map], markersize=0.5, label=false)

    # plot!(xs, ys, marker=:dot, opacity=0.2, linecolor=:gray, markersize=0.2; kwargs...)

end

function plot_tree_dynamic!(problem, solution, t, i_root_node=1; kwargs...)
    # collect a list of all points on the tree at the current time
    points = SVector{2,Float64}[]

    queue = [i_root_node,] # should be a deque...

    # try to traverse the tree
    while length(queue) > 0

        i_node = popfirst!(queue)
        n_node = solution.hash_map[i_node]

        # check is valid  and see if the t is after the node's start time
        if !RRTStar.is_invalid(n_node) && n_node.state[4] <= t
            for i_child in n_node.children
                n_child = solution.hash_map[i_child]
                if !RRTStar.is_invalid(n_child)
                    # see if t is before the childs time
                    if t <= n_child.state[4]
                        # ok so a point can exist
                        # interpolate to find the robot location at t
                        # TODO(dev): make this plotting better
                        # t1 = n_node.state[4]
                        # t2 = n_child.state[4]
                        # x1 = n_node.state[SOneTo(2)]
                        # x2 = n_child.state[SOneTo(2)]

                        # x = x1 + ((t - t1) / (t2 - t1)) * (x2 - x1)

                        # generate a path between these two points
                        t_node = n_node.state[4]
                        t_child = n_child.state[4]
                        x_node = n_node.state[SOneTo(3)]
                        x_child = n_child.state[SOneTo(3)]
                        path = DubinsDynamicPathRRT.generate_path(problem, x_node, x_child)
                        L = dubins_path_length(path)

                        if !(isapprox(L, t_child - t_node; atol=1e-2))
                            @show L
                            @show t_child
                            @show t_node
                            @show t_child - t_node

                            @show x_node
                            @show x_child
                            @show path

                            println("idk what happened, skipping")
                            continue

                        end

                        # @assert isapprox(L, t_child - t_node; atol=1e-2)

                        # sample the path
                        errcode, x = dubins_path_sample(path, t - t_node)
                        push!(points, x[SOneTo(2)])
                    else
                        # maybe the children contain the target time
                        push!(queue, i_child)
                    end
                end
            end
        end
    end
    scatter!(first.(points), last.(points); label=false, kwargs...)
end

function plot_circle!(center, radius; kwargs...)
    plot!(t -> center[1] + radius * cos(t), t -> center[2] + radius * sin(t), 0, 2π; label=false, aspect_ratio=:equal, kwargs...)
end

function plot_obstacle!(obs::DubinsDynamicPathRRT.CircleObstacle; kwargs...)
    plot_circle!(obs.center, obs.radius; kwargs...)
end

function plot_dynamic_obstacle!(obs::DubinsDynamicPathRRT.DynamicCircleObstacle, t; kwargs...)
    c = obs.center(t)
    r = obs.radius
    plot_circle!(c, r; kwargs...)
end

function parent_counter(node, count=0)
    if RRTStar.has_parent(node)
        return parent_counter(node.parent[], count + 1)
    end
    return count
end


