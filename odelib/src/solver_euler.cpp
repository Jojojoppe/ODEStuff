#include "solver_euler.h"

/**
 * @brief Constructor.
 *
 * @param derivatives A function representing the derivative of the state, i.e., f(t, y).
 */
ForwardEuler::ForwardEuler(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives)
	: ODESolver(derivatives)
{
}

/**
 * @brief Performs one integration step using the Forward Euler method.
 *
 * @param t The current time.
 * @param y The current state vector.
 * @param dt The timestep to attempt.
 * @return A StepResult struct with the new state and timestep information.
 */
ODESolver::StepResult ForwardEuler::step(double t, const std::vector<double> &y, double dt)
{
	std::vector<double> y_new = y;
	std::vector<double> dydt = derivatives_func(t, y);
	for (size_t i = 0; i < y.size(); ++i)
	{
		y_new[i] += dt * dydt[i];
	}
	return {y_new, dt, dt};
}