module RRTStar

# export Node, rrt_star, get_best_path

using ..SpatialHashMaps, StaticArrays

abstract type AbstractProblem{T} end

const LinearIndex = CartesianIndex{1}

"""
    Node{T}(state::T, parent_index::Int64, incremental_cost::Float64)

Defines a node in the RRT* Tree, with a defined `state`, `parent_index`, and the `incremental_cost` from the parent to the current node. 
Nodes are stored in a spatial hash map for quick insertions and neighbor lookups.

"""
mutable struct Node{T}
    state::T
    parent::LinearIndex
    cost_so_far::Float64
    children::Set{LinearIndex}
    is_invalid::Bool
end


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

function has_parent(node)
    return node.parent[1] >= 0 # check the cartesian index >= 0
end

function is_invalid(node)
    return node.is_invalid
end


function cost(node::Node{T}) where {T}
    if is_invalid(node)
        return Inf
    end
    return node.cost_so_far
end

@enum SolutionStatus GoalReachable MaxIterations MaxTime NotSolved

abstract type AbstractSolution end

mutable struct RRTStarSolution{T,DIM,F,S} <: AbstractSolution
    root_node::Node{T}
    hash_map::SpatialHashMap{DIM,F,Node{T}}
    best_path::Union{Nothing,Vector{T}}
    status::S
end

function setup(problem::AbstractProblem, start_state::T, hash_map_widths::SVector{DIM,F}) where {DIM,F,T}

    # create the spatial hash map
    origin = spatial_position(problem, start_state)
    hash_map = SpatialHashMap{Node{T}}(origin, hash_map_widths)

    # create the root node
    root_node = RootNode(start_state)

    # insert root node into spatial hash
    insert!(hash_map, root_node, origin)

    best_path = nothing
    status = NotSolved

    # create the solution structure
    solution = RRTStarSolution(
        root_node,
        hash_map,
        best_path,
        status
    )

    return solution

end

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

        if early_exit
            if goal_reachable(problem, solution.hash_map, i_new_node)
                # return the best path
                solution.status = GoalReachable
                return solution
            end
        end

        # check if the tree can be rewired
        if do_rewire
            rrt_rewire!(problem, solution, i_new_node)
        end

    end

    # exited because of max iterations
    solution.status = MaxIterations
    return solution

end



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
    @assert !isnothing(i_nearest)

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




function rrt_rewire!(problem::AbstractProblem, solution::RRTStarSolution, i_new_node::LinearIndex)
    n_new = solution.hash_map[i_new_node]
    I_near = near(problem, solution.hash_map, n_new.state)

    for i_near in I_near
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
    checks if we can rewire nodes's parent to new_parent
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
change the parent of i_node to i_new_parent
also handles updating the childrens lists and the parent id
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
Starting at node id, recalculate the cost for all the children, traversing down the tree
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

        # this might also require changing the node's position
        SpatialHashMaps.change_position!(solution.hash_map, i_child, spatial_position(problem, n_child.state))

        # check for collision along path 
        if collision_free(problem, n_node.state, n_child.state)

            # update the costs
            n_child.cost_so_far = n_node.cost_so_far + stage_cost

            # run this function on the child too
            recalculate_tree!(problem, solution, i_child)
        else
            # mark this child and all children as invalid
            mark_branch_as_invalid(problem, solution, i_node)
        end
    end

    return

end

"""
starting from this node, mark all the children associated with this branch as invalid
Does not modify the parent/child lists
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


# the following functions need to be defined for your problem
function sample_free(problem::P; goal_bias=0.0, hash_map=nothing) where {T,P<:AbstractProblem{T}}
    throw(MethodError(sample_free, (problem, goal_bias, hash_map)))
end

function nearest(problem::P, hash_map, x_rand) where {T,P<:AbstractProblem{T}}
    throw(MethodError(nearest, (problem, nodes, x_rand)))
end

function near(problem::P, hash_map, x_new) where {T,P<:AbstractProblem{T}}
    throw(MethodError(near, (problem, nodes, x_new)))
end

function steer(problem::P, x_nearest, x_rand, distance_fraction) where {T,P<:AbstractProblem{T}}
    throw(MethodError(steer, (problem, x_nearest, x_rand, distance_fraction)))
end

function collision_free(problem::P, x_nearest, x_new) where {T,P<:AbstractProblem{T}}
    throw(MethodError(collision_free, (problem, x_nearest, x_new)))
end

# function path_cost(problem::P, x_near, x_new) where {T,P<:AbstractProblem{T}}
#     throw(MethodError(path_cost, (problem, x_near, x_new)))
# end

function spatial_position(problem::P, x::T) where {T,P<:AbstractProblem{T}}
    throw(MethodError(spatial_position, (problem, x)))
end

function goal_reachable(problem::P, hash_map, i_new) where {T,P<:AbstractProblem{T}}
    throw(MethodError(goal_reachable, (problem, solution, i_new)))
end

end