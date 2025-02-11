#include "solver_rk.h"
#include "vector_math.h"
#include <cmath>

RK4::RK4(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives)
	: ODESolver(derivatives)
{
}

RK4::RK4(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives,
	std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables)
	: ODESolver(derivatives, nonStateVariables)
{
}


ODESolver::StepResult RK4::step(double t, const std::vector<double> &y, double dt)
{
	std::vector<double> k1 = derivatives_func(t, y);
	std::vector<double> k2 = derivatives_func(t + 0.5 * dt, y + 0.5 * dt * k1);
	std::vector<double> k3 = derivatives_func(t + 0.5 * dt, y + 0.5 * dt * k2);
	std::vector<double> k4 = derivatives_func(t + dt, y + dt * k3);

	std::vector<double> y_new = y;
	for (size_t i = 0; i < y.size(); ++i)
	{
		y_new[i] += (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
	}
	return {y_new, dt, dt};
}

RK45::RK45(std::function<std::vector<double>(double, const std::vector<double> &)> f, double tol, double max_step)
	: ODESolver(f), tol{tol}, max_step{max_step}
{
}

RK45::RK45(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives,
	std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables,
	double tol, double max_step)
	: ODESolver(derivatives, nonStateVariables), tol(tol), max_step(max_step)
{
}

ODESolver::StepResult RK45::step(double t, const std::vector<double> &y, double dt)
{
	double dt_current = dt;

	while (true)
	{
		// Compute the Runge-Kutta stages using the Dormand-Prince tableau
		std::vector<double> k1 = derivatives_func(t, y);
		std::vector<double> k2 = derivatives_func(t + dt_current / 5.0, y + (dt_current / 5.0) * k1);
		std::vector<double> k3 = derivatives_func(t + 3.0 * dt_current / 10.0,
												  y + dt_current * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2));
		std::vector<double> k4 = derivatives_func(t + 4.0 * dt_current / 5.0,
												  y + dt_current * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2 + (32.0 / 9.0) * k3));
		std::vector<double> k5 = derivatives_func(t + 8.0 * dt_current / 9.0,
												  y + dt_current * ((19372.0 / 6561.0) * k1 - (25360.0 / 2187.0) * k2 + (64448.0 / 6561.0) * k3 - (212.0 / 729.0) * k4));
		std::vector<double> k6 = derivatives_func(t + dt_current,
												  y + dt_current * ((9017.0 / 3168.0) * k1 - (355.0 / 33.0) * k2 + (46732.0 / 5247.0) * k3 + (49.0 / 176.0) * k4 - (5103.0 / 18656.0) * k5));
		std::vector<double> k7 = derivatives_func(t + dt_current,
												  y + dt_current * ((35.0 / 384.0) * k1 + 0.0 * k2 + (500.0 / 1113.0) * k3 + (125.0 / 192.0) * k4 - (2187.0 / 6784.0) * k5 + (11.0 / 84.0) * k6));

		// Fifth-order solution (higher order)
		std::vector<double> y5 = y + dt_current * ((35.0 / 384.0) * k1 + (500.0 / 1113.0) * k3 + (125.0 / 192.0) * k4 - (2187.0 / 6784.0) * k5 + (11.0 / 84.0) * k6);

		// Fourth-order solution (embedded lower order)
		std::vector<double> y4 = y + dt_current * ((5179.0 / 57600.0) * k1 + (7571.0 / 16695.0) * k3 + (393.0 / 640.0) * k4 - (92097.0 / 339200.0) * k5 + (187.0 / 2100.0) * k6 + (1.0 / 40.0) * k7);

		// Estimate the error as the difference between the two solutions
		std::vector<double> err = y5 - y4;
		double err_norm = norm(err);

		// If error is acceptable, break and return the accepted step.
		if (err_norm <= tol)
		{
			// Compute dt suggestion for the next step:
			double dt_new = dt_current * std::min(max_scale,
												  std::max(min_scale, safety * std::pow((tol / (err_norm + 1e-10)), 0.2)));
			dt_new = std::min(dt_new, max_step);
			return {y5, dt_current, dt_new};
		}
		else
		{
			// Reject the step: reduce dt and try again.
			dt_current = dt_current * std::max(min_scale,
											   safety * std::pow((tol / (err_norm + 1e-10)), 0.2));
		}
	}
}
