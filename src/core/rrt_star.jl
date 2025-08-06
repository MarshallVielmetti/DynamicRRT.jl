"""
    RRTStar

This module provides a generic implementation of the RRT* (Rapidly-exploring Random Tree Star) algorithm.
It is designed to be extended for specific path-planning problems by defining a concrete `AbstractProblem` type
and implementing the required interface functions (e.g., `steer`, `collision_free`, `sample_free`).
"""
module RRTStar

# export Node, rrt_star, get_best_path

using ..SpatialHashMaps, StaticArrays

"""
    AbstractProblem{T}

An abstract type representing a path-planning problem. A concrete implementation of `AbstractProblem`
must be created for a specific use case, along with methods for the RRT* interface functions
(e.g., `steer`, `collision_free`, `sample_free`, `nearest`, `near`, `spatial_position`, `goal_reachable`).
The type parameter `T` represents the type of the state.
"""
abstract type AbstractProblem{T} end

const LinearIndex = CartesianIndex{1}

"""
    Node{T}

Represents a node in the RRT* tree.

# Fields
- `state::T`: The state associated with the node (e.g., position, orientation).
- `parent::LinearIndex`: The index of the parent node in the `SpatialHashMap`.
- `cost_so_far::Float64`: The cumulative cost from the root node to this node.
- `children::Set{LinearIndex}`: A set of indices to the children of this node.
- `is_invalid::Bool`: A flag indicating if the node is part of a valid path. This is used for dynamically marking branches as invalid if they become obstructed.
"""
mutable struct Node{T}
    state::T
    parent::LinearIndex
    cost_so_far::Float64
    children::Set{LinearIndex}
    is_invalid::Bool
end

"""
    RootNode(state::T) where {T}

Constructs the root node of the RRT* tree. The root has no parent (parent index 0) and zero cost.
"""
function RootNode(state::T) where {T}
    parent = LinearIndex(0)
    cost_so_far = 0.0
    children = Set{LinearIndex}()
    is_invalid = false
    return Node(
        state,
        parent,
        cost_so_far,
        children,
        is_invalid)
end

"""
    ChildNode(state::T, i_parent::LinearIndex, cost_so_far) where {T}

Constructs a child node with a given state, parent index, and cost from the root.
"""
function ChildNode(
    state::T,
    i_parent::LinearIndex,
    cost_so_far
) where {T}

    children = Set{LinearIndex}()
    is_invalid = false

    node = Node(
        state,
        i_parent,
        cost_so_far,
        children,
        is_invalid)

    # return the new node
    return node
end

"""
    has_parent(node)

Returns `true` if the node has a parent, `false` otherwise.
"""
function has_parent(node)
    return node.parent[1] > 0 # Root parent index is 0
end

"""
    is_invalid(node)

Returns `true` if the node has been marked as invalid, `false` otherwise.
"""
function is_invalid(node)
    return node.is_invalid
end

"""
    cost(node::Node{T}) where {T}

Returns the cost of a node. Returns `Inf` if the node is invalid.
"""
function cost(node::Node{T}) where {T}
    if is_invalid(node)
        return Inf
    end
    return node.cost_so_far
end

"""
    SolutionStatus

An enum representing the termination status of the `solve!` function.
- `GoalReachable`: The algorithm found at least one path to the goal region.
- `MaxIterations`: The algorithm terminated after reaching the maximum number of iterations.
- `MaxTime`: The algorithm terminated after reaching the maximum time limit.
- `NotSolved`: The algorithm has not been run or has not terminated yet.
"""
@enum SolutionStatus GoalReachable MaxIterations MaxTime NotSolved

"""
    AbstractSolution

Abstract type for a solution to a path-planning problem.
"""
abstract type AbstractSolution end

"""
    RRTStarSolution{T,DIM,F,S}

Contains the state of the RRT* algorithm's execution.

# Fields
- `root_node::Node{T}`: The root node of the tree.
- `hash_map::SpatialHashMap{DIM,F,Node{T}}`: The spatial hash map storing all the nodes in the tree for efficient spatial queries.
- `best_path::Union{Nothing,Vector{T}}`: The best path found so far (if any).
- `best_path_cost::Float64`: The cost of the best path found so far (`Inf` if no path is found).
- `status::S`: The `SolutionStatus` of the algorithm.
"""
mutable struct RRTStarSolution{T,DIM,F,S} <: AbstractSolution
    root_node::Node{T}
    hash_map::SpatialHashMap{DIM,F,Node{T}}
    best_path::Union{Nothing,Vector{T}}
    best_path_cost::Float64
    status::S
end

"""
    setup(problem::AbstractProblem, start_state::T, hash_map_widths::SVector{DIM,F}) where {DIM,F,T}

Initializes the RRT* solver.

# Arguments
- `problem`: The problem definition.
- `start_state`: The initial state of the agent.
- `hash_map_widths`: The cell widths for the spatial hash map.

# Returns
- An `RRTStarSolution` object initialized with the root node.
"""
function setup(problem::AbstractProblem, start_state::T, hash_map_widths::SVector{DIM,F}) where {DIM,F,T}

    # create the spatial hash map
    origin = spatial_position(problem, start_state)
    hash_map = SpatialHashMap{Node{T}}(origin, hash_map_widths)

    # create the root node
    root_node = RootNode(start_state)

    # insert root node into spatial hash
    insert!(hash_map, root_node, origin)

    best_path = nothing
    best_path_cost = Inf
    status = NotSolved

    # create the solution structure
    solution = RRTStarSolution(
        root_node,
        hash_map,
        best_path,
        best_path_cost,
        status
    )

    return solution

end

"""
    solve!(problem::AbstractProblem, solution::RRTStarSolution; kwargs...)

Runs the RRT* algorithm to find a path.

# Arguments
- `problem`: The problem definition.
- `solution`: The solution object, initialized by `setup`.

# Keyword Arguments
- `max_iterations=1000`: The maximum number of iterations to run.
- `max_time_seconds=1.0`: The maximum time in seconds to run the algorithm.
- `distance_fraction=0.2`: The fraction of the distance to steer towards the random sample.
- `do_rewire=true`: Whether to perform the RRT* rewiring step.
- `early_exit=true`: If `true`, the algorithm terminates as soon as the first path to the goal is found.
- `goal_bias=0.1`: The probability of sampling the goal state instead of a random state.

# Returns
- The updated `RRTStarSolution` object containing the final tree and solution status.
"""
function solve!(problem::AbstractProblem, solution::RRTStarSolution;
    max_iterations=1000,
    max_time_seconds=1.0,
    distance_fraction=0.2,
    do_rewire=true,
    early_exit=true,
    goal_bias=0.1)

    # initialize
    max_time = time() + max_time_seconds

    # start loop
    for iter = 1:max_iterations

        if time() > max_time
            # exited because of max time
            solution.status = MaxTime
            return solution
        end

        # add one node to the list
        i_new_node = rrt_step!(problem, solution;
            distance_fraction=distance_fraction,
            goal_bias=goal_bias
        )

        if isnothing(i_new_node)
            # no new node added, continue to next iteration
            continue
        end

        # If the new node is the first one to reach the goal region, update the status.
        # If early_exit is enabled, compute the path and terminate.
        if solution.status != GoalReachable && goal_reachable(problem, solution.hash_map, i_new_node)
            solution.status = GoalReachable
            if early_exit
                best_path!(problem, solution) # Find and store the best path
                return solution
            end
        end

        # check if the tree can be rewired
        if do_rewire
            rrt_rewire!(problem, solution, i_new_node)
        end

    end

    # exited because of max iterations
    if solution.status != GoalReachable
        solution.status = MaxIterations
    end

    # Find the best path among all goal-reaching nodes found.
    best_path!(problem, solution)

    return solution
end

"""
    rrt_step!(problem::AbstractProblem, solution::RRTStarSolution; kwargs...)

Performs a single iteration of the RRT* algorithm.
This involves sampling a random state, finding the nearest node in the tree,
steering towards the random state, and connecting the new node to the best parent in the neighborhood.

# Returns
- The `LinearIndex` of the newly added node, or `nothing` if no node was added.
"""
function rrt_step!(problem::AbstractProblem, solution::RRTStarSolution;
    distance_fraction=0.2,
    goal_bias=0.1)

    # i -> index in nodes vector
    # x -> actual state
    # n -> full node

    # grab a random state
    x_rand = sample_free(problem; goal_bias=goal_bias, hash_map=solution.hash_map)

    # find the nearest node
    i_nearest = nearest(problem, solution.hash_map, x_rand)
    if isnothing(i_nearest) || i_nearest[1] == 0
        return nothing
    end

    # check the output
    n_nearest = solution.hash_map[i_nearest]
    x_nearest = n_nearest.state
    if is_invalid(n_nearest)
        return nothing
    end

    # get the new point, if moving distance_fraction towards x_rand
    x_new, cost_nearest_new = steer(problem, x_nearest, x_rand, distance_fraction)

    # check that it is obstacle free
    if collision_free(problem, x_nearest, x_new)

        # determine the best one to connect to
        best_i_near = i_nearest # best parent node
        best_n_near = n_nearest # best parent node
        best_x_new = x_new      # best child state
        best_c_new = cost(best_n_near) + cost_nearest_new  # cost of child node

        # get the set of nearby nodes (this should return an set of nodes)
        I_near = near(problem, solution.hash_map, x_new)

        for i_near in I_near
            if i_near[1] == 0
                continue
            end
            n_near = solution.hash_map[i_near]

            if is_invalid(n_near)
                continue
            end

            # see if it is beneficial to connect to the other nearby node instead
            x_new_candidate, cost_near_to_new_candidate = steer(problem, n_near.state, x_new, 1.0)
            c_candidate = cost(n_near) + cost_near_to_new_candidate

            if (c_candidate < best_c_new) && collision_free(problem, n_near.state, x_new_candidate)
                best_i_near = i_near # update the best parent
                best_n_near = n_near # update the best parent
                best_x_new = x_new_candidate # update the resulting state
                best_c_new = c_candidate # update the cost of the new child
            end
        end

        # create child node
        n_new = ChildNode(
            best_x_new,
            best_i_near,
            best_c_new)

        # add child node to hash map
        i_new = insert!(solution.hash_map, n_new, spatial_position(problem, n_new.state))

        # push this child to the parent list
        push!(best_n_near.children, CartesianIndex(i_new))

        return CartesianIndex(i_new)
    end
    return nothing
end

"""
    rrt_rewire!(problem::AbstractProblem, solution::RRTStarSolution, i_new_node::LinearIndex)

Performs the rewiring step of the RRT* algorithm for the `i_new_node`.
It checks all neighbors of the new node and rewires their parent to be the new node if it results
in a lower-cost path.
"""
function rrt_rewire!(problem::AbstractProblem, solution::RRTStarSolution, i_new_node::LinearIndex)
    n_new = solution.hash_map[i_new_node]
    I_near = near(problem, solution.hash_map, n_new.state)

    for i_near in I_near
        if i_near[1] == 0
            continue
        end
        n_near = solution.hash_map[i_near]
        if is_invalid(n_near) || !has_parent(n_near) # skip the root node or any invalid nodes
            continue
        end
        if can_rewire(problem, solution, i_near, i_new_node)
            change_parent!(problem, solution, i_near, i_new_node)
            recalculate_tree!(problem, solution, i_new_node)
        end
    end
    return
end

"""
    can_rewire(problem, solution, i_node, i_new_parent)

Checks if a `node` can be rewired to have `new_parent` as its parent.
Rewiring is possible if the path from `new_parent` to `node` is collision-free
and the total cost through `new_parent` is less than the node's current cost.
"""
function can_rewire(problem, solution, i_node, i_new_parent)

    n_node = solution.hash_map[i_node]
    n_new_parent = solution.hash_map[i_new_parent]

    @assert has_parent(n_node)
    @assert !is_invalid(n_node)
    @assert !is_invalid(n_new_parent)

    # first compute what the new node_state would be with the new parent
    new_state, stage_cost = steer(problem, n_new_parent.state, n_node.state, 1.0)

    # now check if the cost is potentially lower
    new_cost = cost(n_new_parent) + stage_cost
    if new_cost < n_node.cost_so_far
        # now check if the path is collision free
        if collision_free(problem, n_new_parent.state, new_state)
            return true
        end
    end

    return false
end


"""
    change_parent!(problem::AbstractProblem, solution, i_node, i_new_parent)

Changes the parent of `i_node` to `i_new_parent`.
This involves updating the parent index of the node and adjusting the `children` sets of both the old and new parents.
"""
function change_parent!(problem::AbstractProblem, solution, i_node, i_new_parent)

    n_node = solution.hash_map[i_node]
    n_new_parent = solution.hash_map[i_new_parent]

    @assert has_parent(n_node)
    @assert !is_invalid(n_node)
    @assert !is_invalid(n_new_parent)

    i_old_parent = n_node.parent
    n_old_parent = solution.hash_map[i_old_parent]

    # remove the node from the old_parent's children list
    delete!(n_old_parent.children, i_node)

    # set nodes parent id to new parent id
    n_node.parent = i_new_parent

    # add this node to the parents children
    push!(n_new_parent.children, i_node)
end

"""
    recalculate_tree!(problem::AbstractProblem, solution, i_node)

Recursively recalculates the cost and state of all descendants of `i_node`.
This is called after a rewiring operation to propagate the cost changes down the tree.
If a branch becomes invalid due to a collision, it marks the branch as such.
"""
function recalculate_tree!(problem::AbstractProblem, solution, i_node)

    n_node = solution.hash_map[i_node]

    for i_child in n_node.children

        n_child = solution.hash_map[i_child]

        if is_invalid(n_child)
            # skip this child
            continue
        end

        # update the new state of the child
        # (in dynamic problems this is needed to update the time of the state)
        n_child.state, stage_cost = steer(problem, n_node.state, n_child.state, 1.0)

        # this might also require changing the node's position in the hash map
        SpatialHashMaps.change_position!(solution.hash_map, i_child, spatial_position(problem, n_child.state))

        # check for collision along path
        if collision_free(problem, n_node.state, n_child.state)

            # update the costs
            n_child.cost_so_far = n_node.cost_so_far + stage_cost

            # run this function on the child too
            recalculate_tree!(problem, solution, i_child)
        else
            # mark this child and all children as invalid
            mark_branch_as_invalid(problem, solution, i_child)
        end
    end

    return
end

"""
    mark_branch_as_invalid(problem, solution, i_node)

Recursively marks a node and all its descendants as invalid.
This does not modify the parent/child lists, only the `is_invalid` flag.
"""
function mark_branch_as_invalid(problem, solution, i_node)

    n_node = solution.hash_map[i_node]

    n_node.is_invalid = true

    # do the same to all of this nodes children
    for i_child in n_node.children
        mark_branch_as_invalid(problem, solution, i_child)
    end
    return
end

"""
    best_path!(problem::AbstractProblem, solution::RRTStarSolution)

Finds the best path to the goal among all nodes in the tree.
It iterates through all nodes, checks if they reach the goal, and if so,
updates the `best_path` and `best_path_cost` in the solution object if the new path is better.
"""
function best_path!(problem::AbstractProblem, solution::RRTStarSolution)
    # Loop through all nodes to find the one with the lowest cost that is reachable to the goal
    for (i, node) in enumerate(solution.hash_map)
        # If the goal is not reachable from this node, or the node is invalid, skip it
        if is_invalid(node) || !goal_reachable(problem, solution.hash_map, i)
            continue
        end
        # Set exit status to reachable if at least one node can reach the goal
        solution.status = GoalReachable

        # Compute the total cost to the goal using that node
        c = cost(node) + cost_to_goal(problem, node)
        if c < solution.best_path_cost
            solution.best_path_cost = c
            solution.best_path = extract_path(solution, i)
            push!(solution.best_path, SVector{4,Float64}(problem.goal_state..., 0.0)) # Append the goal state to the path
        end
    end
end

"""
    extract_path(solution::RRTStarSolution, node_i)

Extracts the path from the root to the node at index `node_i` by backtracking from child to parent.
Returns a vector of states representing the path.
"""
function extract_path(solution::RRTStarSolution, node_i)
    path = []
    current = solution.hash_map[node_i]

    push!(path, current.state)
    while !isnothing(current) && has_parent(current)
        current = solution.hash_map[current.parent]
        push!(path, current.state)
    end
    return reverse(path)
end


# -------------------------------------------------------------------------------- #
# Abstract Problem Interface
#
# The following functions must be implemented for a concrete `AbstractProblem` type
# to define the specific behavior of the RRT* algorithm.
# -------------------------------------------------------------------------------- #

"""
    sample_free(problem::P; goal_bias=0.0, hash_map=nothing)

Sample a random state from the free space.
"""
function sample_free(problem::P; goal_bias=0.0, hash_map=nothing) where {T,P<:AbstractProblem{T}}
    throw(MethodError(sample_free, (problem, goal_bias, hash_map)))
end

"""
    nearest(problem::P, hash_map, x_rand)

Find the nearest node in the `hash_map` to the state `x_rand`.
"""
function nearest(problem::P, hash_map, x_rand) where {T,P<:AbstractProblem{T}}
    throw(MethodError(nearest, (problem, hash_map, x_rand)))
end

"""
    near(problem::P, hash_map, x_new)

Find all nodes in the `hash_map` within a certain radius of the state `x_new`.
"""
function near(problem::P, hash_map, x_new) where {T,P<:AbstractProblem{T}}
    throw(MethodError(near, (problem, hash_map, x_new)))
end

"""
    steer(problem::P, x_nearest, x_rand, distance_fraction)

Steer from `x_nearest` towards `x_rand` by a `distance_fraction`.
Returns `(new_state, cost)`.
"""
function steer(problem::P, x_nearest, x_rand, distance_fraction) where {T,P<:AbstractProblem{T}}
    throw(MethodError(steer, (problem, x_nearest, x_rand, distance_fraction)))
end

"""
    collision_free(problem::P, x_from, x_to)

Check if the path from `x_from` to `x_to` is collision-free.
"""
function collision_free(problem::P, x_from, x_to) where {T,P<:AbstractProblem{T}}
    throw(MethodError(collision_free, (problem, x_from, x_to)))
end

"""
    spatial_position(problem::P, x::T)

Extract the spatial component of a state `x` for use in the `SpatialHashMap`.
"""
function spatial_position(problem::P, x::T) where {T,P<:AbstractProblem{T}}
    throw(MethodError(spatial_position, (problem, x)))
end

"""
    goal_reachable(problem::P, hash_map, i_new)

Check if the goal is reachable from the new node `i_new`.
"""
function goal_reachable(problem::P, hash_map, i_new) where {T,P<:AbstractProblem{T}}
    throw(MethodError(goal_reachable, (problem, hash_map, i_new)))
end

"""
    cost_to_goal(problem::P, node::Node{T}) where {T,P<:AbstractProblem{T}}

Calculate the cost from a given `node` to the goal. This is used to find the true best path.
"""
function cost_to_goal(problem::P, node::Node{T}) where {T,P<:AbstractProblem{T}}
    throw(MethodError(cost_to_goal, (problem, node)))
end

end