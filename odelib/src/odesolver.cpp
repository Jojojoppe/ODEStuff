#include "odesolver.h"

ODESolver::ODESolver(std::function<std::vector<double>(double, const std::vector<double>&)> derivatives)
    : derivatives_func(derivatives)
    , variables_func(nullptr)
{
}

ODESolver::ODESolver(std::function<std::vector<double>(double, const std::vector<double>&)> derivatives,
    std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables)
    : derivatives_func(derivatives)
    , variables_func(nonStateVariables)
{
}

std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
ODESolver::solve(const std::vector<double>& initial_conditions, double t_start, double t_end, double dt)
{
    // Calculate the number of time steps (based on the initial dt, but will adjust dynamically)
    std::vector<double> t_values;
    std::vector<std::vector<double>> y_values;
    std::vector<std::vector<double>> ns_values;
    t_values.push_back(t_start);
    y_values.push_back(initial_conditions);
    if (variables_func) {
        ns_values.push_back(variables_func(t_start, initial_conditions));
    }

    double t = t_start;
    std::vector<double> y = initial_conditions;

    while (t < t_end) {
        // Step the solver, get the new solution and suggested dt
        auto step_result = step(t, y, dt);
        std::vector<double>& y_next = step_result.y;

        // Update time and state
        t += step_result.dt_accepted;
        if (t > t_end) { // Clamp to t_end if needed.
            t = t_end;
        }
        t_values.push_back(t);
        y_values.push_back(y_next);

        if (variables_func) {
            ns_values.push_back(variables_func(t_start, y));
        }

        // Update the current state (y) with the new solution
        y = y_next;

        // Use the new dt for the next step
        dt = step_result.dt_new;
    }

    return { t_values, y_values, ns_values };
}
