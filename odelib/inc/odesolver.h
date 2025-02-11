#pragma once
#include <functional>
#include <tuple>
#include <vector>

/**
 * @brief Base class for solving systems of ordinary differential equations (ODEs).
 *
 * This abstract class defines an interface for ODE solvers of the form:
 * \f[
 * \frac{dy}{dt} = f(t, y)
 * \f]
 * where \f$ y \f$ is a vector of state variables. The solver advances the solution
 * over a specified time interval using adaptive time-stepping (if supported).
 *
 * Derived classes must implement the protected virtual method \c step() which
 * computes one integration step.
 */
class ODESolver {
public:
    /**
     * @brief Constructor.
     *
     * @param derivatives A function representing the derivative of the state, i.e. f(t,y)
     */
    ODESolver(std::function<std::vector<double>(double, const std::vector<double>&)> derivatives);
    /**
     * @brief Constructor.
     *
     * @param derivatives A function representing the derivative of the state, i.e. f(t,y)
     * @param nonStateVariables A function calculating extra (non-state) variables from the state and time (g(t, y))
     */
    ODESolver(std::function<std::vector<double>(double, const std::vector<double>&)> derivatives,
        std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables);

    /**
     * @brief Solves the ODE system over a specified time interval.
     *
     * This method integrates the ODE from \c t_start to \c t_end starting with the provided
     * initial conditions. The integration uses a timestep \c dt that may be adapted if the underlying
     * step method supports adaptive stepping.
     *
     * @param initial_conditions The initial state vector at time t_start.
     * @param t_start The starting time of integration.
     * @param t_end The final time of integration.
     * @param dt The initial timestep (which may be adjusted adaptively).
     * @return A tuple where the first element is a vector of time values, the second element is a
     *         vector of state vectors corresponding to each time and the last a vector of non-state
     *         variable vectors corresponding to each time.
     */
    std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    solve(const std::vector<double>& initial_conditions, double t_start, double t_end, double dt);

    /**
     * @brief Structure that holds the result of a single integration step.
     */
    struct StepResult {
        std::vector<double> y; ///< The accepted new state vector after the step.
        double dt_accepted; ///< The timestep that was actually used for the step.
        double dt_new; ///< The suggested new timestep for the next step.
    };

protected:
    /// The derivative function f(t, y) defining the ODE system.
    std::function<std::vector<double>(double, const std::vector<double>&)> derivatives_func;
    /// The non-state variables function g(t, y) calculating auxilary quantities
    std::function<std::vector<double>(double, const std::vector<double>&)> variables_func;

    /**
     * @brief Advances the solution by one step.
     *
     * Derived classes must implement this function to perform one integration step starting from
     * time \c t and state \c y using the timestep \c dt. The method returns a \c StepResult containing
     * the new state, the accepted timestep, and a suggested timestep for the next step.
     *
     * @param t The current time.
     * @param y The current state vector.
     * @param dt The timestep to attempt.
     * @return A StepResult struct with the new state and timestep information.
     */
    virtual StepResult step(double t, const std::vector<double>& y, double dt) = 0;
};
