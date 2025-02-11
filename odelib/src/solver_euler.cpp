#include "solver_euler.h"

ForwardEuler::ForwardEuler(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives)
	: ODESolver(derivatives)
{
}

ForwardEuler::ForwardEuler(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives,
	std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables)
	: ODESolver(derivatives, nonStateVariables)
{
}

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
