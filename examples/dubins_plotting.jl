module DynamicDubinsPlotting

using DynamicRRT
using Plots, StaticArrays, Dubins, Random

export plot_node!, plot_tree!, plot_path!, plot_tree_dynamic!, plot_obstacle!, plot_dynamic_obstacle!, parent_counter

"""
    plot_node!(node; kwargs...)

Plots a single node from the RRT* tree.
"""
function plot_node!(node; kwargs...)
    x, y = node.state
    scatter!([x], [y], label=false; kwargs...)
end

"""
    plot_tree!(problem, solution, i_root_node=1; kwargs...)

Plots the entire RRT* tree by traversing from the root node.
It plots the Dubins path segments between parent and child nodes.
"""
function plot_tree!(problem, solution, i_root_node=1; kwargs...)

    queue = [i_root_node,] # should be a deque

    while length(queue) > 0
        i_node = popfirst!(queue)
        n_node = solution.hash_map[i_node]

        if RRTStar.is_invalid(n_node)
            continue
        end

        for i_child in n_node.children
            n_child = solution.hash_map[i_child]

            if RRTStar.is_invalid(n_child)
                continue
            end

            # Get the Dubins path between the valid parent-child pair
            x_node = n_node.state[SOneTo(3)]
            x_child = n_child.state[SOneTo(3)]
            path = DubinsDynamicPathRRT.generate_path(problem, x_node, x_child)

            # Sample the path for plotting
            errcode, xs = Dubins.dubins_path_sample_many(path, 0.01)

            # Plot the path segment
            plot!([x[1] for x in xs], [x[2] for x in xs], label=false, opacity=0.2, linecolor=:gray, markersize=0.2; kwargs...)

            push!(queue, i_child)
        end
    end

    # Plot all nodes in the hashmap
    scatter!([n.state[1] for n in solution.hash_map], [n.state[2] for n in solution.hash_map], markersize=0.5, label=false)
end

"""
    plot_path!(problem, path; kwargs...)

Plots a given path, which is a sequence of states.
"""
function plot_path!(problem, path; kwargs...)
    # Plot the Dubins path segments
    for i in 1:length(path)-1
        x1 = path[i][SOneTo(3)]
        x2 = path[i+1][SOneTo(3)]
        p = DubinsDynamicPathRRT.generate_path(problem, x1, x2)
        errcode, xs = Dubins.dubins_path_sample_many(p, 0.01)
        plot!([x[1] for x in xs], [x[2] for x in xs], label=false, opacity=0.9, linecolor=:blue, markersize=0.2, linewidth=3.0; kwargs...)
    end
end

"""
    plot_tree_dynamic!(problem, solution, t, i_root_node=1; kwargs...)

Plots the state of the tree at a specific time `t`. It shows the position of the vehicle
along the path segments that are "active" at that time.
"""
function plot_tree_dynamic!(problem, solution, t, i_root_node=1; kwargs...)
    points = SVector{2,Float64}[]
    queue = [i_root_node,]

    while length(queue) > 0
        i_node = popfirst!(queue)
        n_node = solution.hash_map[i_node]

        if !RRTStar.is_invalid(n_node) && n_node.state[4] <= t
            for i_child in n_node.children
                n_child = solution.hash_map[i_child]
                if !RRTStar.is_invalid(n_child)
                    if t <= n_child.state[4]
                        # The current time `t` is between the parent and child node's time.
                        # Interpolate the position along the Dubins path.
                        t_node = n_node.state[4]
                        x_node = n_node.state[SOneTo(3)]
                        x_child = n_child.state[SOneTo(3)]
                        path = DubinsDynamicPathRRT.generate_path(problem, x_node, x_child)

                        # Sample the path at the correct time offset
                        errcode, x = dubins_path_sample(path, t - t_node)
                        push!(points, x[SOneTo(2)])
                    else
                        # This branch is in the past, but its children might be in the future.
                        push!(queue, i_child)
                    end
                end
            end
        end
    end
    scatter!(first.(points), last.(points); label=false, kwargs...)
end

"""
    plot_circle!(center, radius; kwargs...)

Helper function to plot a circle.
"""
function plot_circle!(center, radius; kwargs...)
    plot!(t -> center[1] + radius * cos(t), t -> center[2] + radius * sin(t), 0, 2π; label=false, aspect_ratio=:equal, kwargs...)
end

"""
    plot_obstacle!(obs::DubinsDynamicPathRRT.CircleObstacle; kwargs...)

Plots a static `CircleObstacle`.
"""
function plot_obstacle!(obs::DubinsDynamicPathRRT.CircleObstacle; kwargs...)
    plot_circle!(obs.center, obs.radius; kwargs...)
end

"""
    plot_dynamic_obstacle!(obs::DubinsDynamicPathRRT.DynamicCircleObstacle, t; kwargs...)

Plots a `DynamicCircleObstacle` at a specific time `t`.
"""
function plot_dynamic_obstacle!(obs::DubinsDynamicPathRRT.DynamicCircleObstacle, t; kwargs...)
    c = obs.center(t)
    r = obs.radius
    plot_circle!(c, r; kwargs...)
end

"""
    parent_counter(node, count=0)

Utility function to count the number of parents up to the root.
"""
function parent_counter(node, count=0)
    if RRTStar.has_parent(node)
        return parent_counter(node.parent[], count + 1)
    end
    return count
end


end # module DynamicDubinsPlotting