
using DynamicRRT

include("dubins_plotting.jl")
using .DynamicDubinsPlotting

using StaticArrays, Dubins, Random, Plots

Random.seed!(0)

min_domain = SVector(0, 0.0, -π)
max_domain = SVector(1.0, 1.0, π)
domain = (min_domain, max_domain)

static_obstacles = [DubinsDynamicPathRRT.random_static_obstacle() for i = 1:2]
dynamic_obstacles = [
    DubinsDynamicPathRRT.DynamicCircleObstacle(
        t -> SVector(0.5 + 0.5 * sin(-t), 0.3),
        0.1), DubinsDynamicPathRRT.DynamicCircleObstacle(
        t -> SVector(sin(t + π / 4), 0.6),
        0.1), DubinsDynamicPathRRT.DynamicCircleObstacle(
        t -> SVector(sin(t - π / 4), 0.9),
        0.1)
]

goal_state = SVector(1, 1.0, 0.0)
turning_radius = 0.1
rrt_problem = DubinsDynamicPathRRT.DubinsDynamicRRTProblem(domain, 0.1, static_obstacles, dynamic_obstacles, goal_state)

start_state = SVector(0.0, 0.0, 0.0, 0.0)

Random.seed!(0)
widths = SVector(0.1, 0.1) # only spatial hashing - ignoring the temporal dimension of the state
sol = RRTStar.setup(rrt_problem, start_state, widths)

@time RRTStar.solve!(rrt_problem, sol;
    max_iterations=1000,
    max_time_seconds=1.0,
    do_rewire=true,
    early_exit=true,
    goal_bias=0.1);

plot()
for o in rrt_problem.static_obstacles
    plot_obstacle!(o, color=:black)
end
τ = 2.0
for o in rrt_problem.dynamic_obstacles
    plot_dynamic_obstacle!(o, τ, color=:red)
end
# scatter_plot_hashmap!(sol.hash_map; label=false)
plot_tree!(rrt_problem, sol)
plot_tree_dynamic!(rrt_problem, sol, τ)
plot_path!(rrt_problem, sol.best_path)
savefig("dubins_dynamic_rrt_star.png")

readme_gif = @animate for t in range(0.0, 2.0, length=60)
    plot()
    for o in rrt_problem.static_obstacles
        plot_obstacle!(o, color=:black)
    end
    for o in rrt_problem.dynamic_obstacles
        plot_dynamic_obstacle!(o, t, color=:red)
    end
    # scatter_plot_hashmap!(sol.hash_map; label=false)
    plot_tree!(rrt_problem, sol)

    plot_tree_dynamic!(rrt_problem, sol, t)

    xlims!(-0.75, 1.25)
end

gif(readme_gif, "dubins_dynamic_rrt_star.gif", fps=15)
