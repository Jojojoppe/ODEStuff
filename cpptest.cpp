#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <cmath>

// Vector addition
std::vector<double> operator+(const std::vector<double> &a, const std::vector<double> &b)
{
	std::vector<double> result(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		result[i] = a[i] + b[i];
	return result;
}

// Vector subtraction
std::vector<double> operator-(const std::vector<double> &a, const std::vector<double> &b)
{
	std::vector<double> result(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		result[i] = a[i] - b[i];
	return result;
}

// Scalar multiplication
std::vector<double> operator*(double scalar, const std::vector<double> &v)
{
	std::vector<double> result(v.size());
	for (size_t i = 0; i < v.size(); ++i)
		result[i] = scalar * v[i];
	return result;
}

// Scalar multiplication (commutative)
std::vector<double> operator*(const std::vector<double> &v, double scalar)
{
	return scalar * v;
}

// Compute Euclidean norm of a vector
double norm(const std::vector<double> &v)
{
	double sum = 0.0;
	for (auto val : v)
		sum += val * val;
	return std::sqrt(sum);
}

class ODESolver
{
public:
	// Constructor that accepts the derivative function
	ODESolver(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives)
		: derivatives_func(derivatives) {}

	// Solve the ODE over a given time region with a fixed timestep
	std::pair<std::vector<double>, std::vector<std::vector<double>>> solve(
		const std::vector<double> &initial_conditions, double t_start, double t_end, double dt)
	{

		// Calculate the number of time steps (based on the initial dt, but will adjust dynamically)
		std::vector<double> t_values;
		std::vector<std::vector<double>> y_values;
		t_values.push_back(t_start);
		y_values.push_back(initial_conditions);

		double t = t_start;
		std::vector<double> y = initial_conditions;

		while (t < t_end)
		{
			// Step the solver, get the new solution and suggested dt
			auto step_result = step(t, y, dt);
			std::vector<double> &y_next = step_result.y;

			// Update time and state
			t += step_result.dt_accepted;
			if (t > t_end)
			{ // Clamp to t_end if needed.
				t = t_end;
			}
			t_values.push_back(t);
			y_values.push_back(y_next);

			// Update the current state (y) with the new solution
			y = y_next;

			// Use the new dt for the next step
			dt = step_result.dt_new;
		}

		return {t_values, y_values};
	}

	// Struct to hold the result of one adaptive step
	struct StepResult
	{
		std::vector<double> y; // the accepted new solution
		double dt_accepted;	   // the dt that was actually used for this step
		double dt_new;		   // the suggested new dt for the next step
	};

protected:
	// Virtual step function that will be implemented in derived classes
	virtual StepResult step(double t, const std::vector<double> &y, double dt) = 0;

	std::function<std::vector<double>(double, const std::vector<double> &)> derivatives_func;
};

// Forward Euler method (derived class)
class ForwardEuler : public ODESolver
{
public:
	ForwardEuler(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives)
		: ODESolver(derivatives) {}

protected:
	StepResult step(double t, const std::vector<double> &y, double dt) override
	{
		return {y + dt * derivatives_func(t, y), dt, dt};
	}
};

// Runge-Kutta 4th order method (derived class)
class RK4 : public ODESolver
{
public:
	RK4(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives)
		: ODESolver(derivatives) {}

protected:
	StepResult step(double t, const std::vector<double> &y, double dt) override
	{
		std::vector<double> k1 = derivatives_func(t, y);
		std::vector<double> k2 = derivatives_func(t + 0.5 * dt, y + 0.5 * dt * k1);
		std::vector<double> k3 = derivatives_func(t + 0.5 * dt, y + 0.5 * dt * k2);
		std::vector<double> k4 = derivatives_func(t + dt, y + dt * k3);

		return {y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), dt, dt};
	}
};

// RK45 implementation using Dormand–Prince coefficients
class RK45 : public ODESolver
{
public:
	RK45(std::function<std::vector<double>(double, const std::vector<double> &)> f, double tol = 1e-6, double max_step = 1)
		: ODESolver(f), tol{tol}, max_step{max_step}
	{
	}

	StepResult step(double t, const std::vector<double> &y, double dt) override
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

private:
	const double tol;
	const double safety = 0.9;
	const double min_scale = 0.2;
	const double max_scale = 5.0;
	const double max_step;
};

//--- RadauIIA solver class (two-stage, order 3) ---//
class RadauIIA : public ODESolver
{
public:
	RadauIIA(std::function<std::vector<double>(double, const std::vector<double> &)> f)
		: ODESolver(f) {}

protected:
	// Override step to perform an implicit Radau IIA step.
	StepResult step(double t, const std::vector<double> &y, double dt) override
	{
		int n = y.size();
		// Our unknown is X = [K1, K2] of length 2*n.
		std::vector<double> X(2 * n, 0.0);
		// Initial guess: use f evaluated at the stage times with current state.
		std::vector<double> guess1 = derivatives_func(t + dt / 3.0, y);
		std::vector<double> guess2 = derivatives_func(t + dt, y);
		for (int i = 0; i < n; i++)
		{
			X[i] = guess1[i];	  // initial guess for K1
			X[n + i] = guess2[i]; // initial guess for K2
		}

		// Newton iteration to solve F(X)=0, where F : R^(2n) -> R^(2n) is defined below.
		const int maxIter = 20;
		const double tol = 1e-8;
		for (int iter = 0; iter < maxIter; iter++)
		{
			std::vector<double> F_val = computeF(X, t, y, dt);
			double F_norm = vectorNorm(F_val);
			if (F_norm < tol)
				break;
			std::vector<std::vector<double>> J = computeJacobian(X, t, y, dt);
			std::vector<double> delta = solveLinearSystem(J, F_val);
			for (size_t i = 0; i < X.size(); i++)
			{
				X[i] -= delta[i];
			}
			if (vectorNorm(delta) < tol)
				break;
		}

		// Extract stage values from X.
		std::vector<double> K1(X.begin(), X.begin() + n);
		std::vector<double> K2(X.begin() + n, X.end());
		// Compute the new state: y_next = y + dt*(3/4*K1 + 1/4*K2)
		std::vector<double> y_next(n, 0.0);
		for (int i = 0; i < n; i++)
		{
			y_next[i] = y[i] + dt * (0.75 * K1[i] + 0.25 * K2[i]);
		}
		return {y_next, dt, dt};
	}

private:
	// Helper: Euclidean norm of a vector.
	double vectorNorm(const std::vector<double> &v)
	{
		double sum = 0.0;
		for (double x : v)
			sum += x * x;
		return std::sqrt(sum);
	}

	// Define the nonlinear function F(X) = 0 for stage values.
	// Here X = [K1, K2] and we set:
	//   F₁ = K1 - derivatives_func(t + dt/3, y + dt*(5/12*K1 - 1/12*K2))
	//   F₂ = K2 - derivatives_func(t + dt,   y + dt*(3/4*K1 + 1/4*K2))
	std::vector<double> computeF(const std::vector<double> &X, double t, const std::vector<double> &y, double dt)
	{
		int n = y.size();
		std::vector<double> F(2 * n, 0.0);
		std::vector<double> K1(X.begin(), X.begin() + n);
		std::vector<double> K2(X.begin() + n, X.end());

		double t1 = t + dt / 3.0;
		double t2 = t + dt;
		std::vector<double> g1(n, 0.0), g2(n, 0.0);
		for (int i = 0; i < n; i++)
		{
			g1[i] = y[i] + dt * ((5.0 / 12.0) * K1[i] - (1.0 / 12.0) * K2[i]);
			g2[i] = y[i] + dt * ((3.0 / 4.0) * K1[i] + (1.0 / 4.0) * K2[i]);
		}
		std::vector<double> f1 = derivatives_func(t1, g1);
		std::vector<double> f2 = derivatives_func(t2, g2);
		for (int i = 0; i < n; i++)
		{
			F[i] = K1[i] - f1[i];
			F[n + i] = K2[i] - f2[i];
		}
		return F;
	}

	// Compute the Jacobian J = DF/DX (finite differences).
	std::vector<std::vector<double>> computeJacobian(const std::vector<double> &X, double t, const std::vector<double> &y, double dt)
	{
		int m = X.size();
		std::vector<std::vector<double>> J(m, std::vector<double>(m, 0.0));
		std::vector<double> F0 = computeF(X, t, y, dt);
		double eps = 1e-8;
		for (int j = 0; j < m; j++)
		{
			std::vector<double> X_pert = X;
			X_pert[j] += eps;
			std::vector<double> F_pert = computeF(X_pert, t, y, dt);
			for (int i = 0; i < m; i++)
			{
				J[i][j] = (F_pert[i] - F0[i]) / eps;
			}
		}
		return J;
	}

	// Solve the linear system J * delta = F for delta via Gaussian elimination.
	std::vector<double> solveLinearSystem(std::vector<std::vector<double>> A, std::vector<double> b)
	{
		int n = b.size();
		std::vector<double> x(n, 0.0);
		// Augment A with b.
		for (int i = 0; i < n; i++)
		{
			A[i].push_back(b[i]);
		}
		// Gaussian elimination with partial pivoting.
		for (int i = 0; i < n; i++)
		{
			// Find pivot row.
			int pivot = i;
			for (int j = i + 1; j < n; j++)
			{
				if (std::fabs(A[j][i]) > std::fabs(A[pivot][i]))
					pivot = j;
			}
			if (std::fabs(A[pivot][i]) < 1e-12)
				continue; // singular matrix
			std::swap(A[i], A[pivot]);
			// Normalize pivot row.
			double pivotVal = A[i][i];
			for (int j = i; j <= n; j++)
			{
				A[i][j] /= pivotVal;
			}
			// Eliminate below pivot.
			for (int j = i + 1; j < n; j++)
			{
				double factor = A[j][i];
				for (int k = i; k <= n; k++)
				{
					A[j][k] -= factor * A[i][k];
				}
			}
		}
		// Back substitution.
		for (int i = n - 1; i >= 0; i--)
		{
			x[i] = A[i][n];
			for (int j = i + 1; j < n; j++)
			{
				x[i] -= A[i][j] * x[j];
			}
		}
		return x;
	}
};

// Function to write ODE results to a file
void save_results(const std::string &filename, const std::vector<double> &t, const std::vector<std::vector<double>> &y)
{
	std::ofstream file(filename);
	if (!file)
	{
		std::cerr << "Error opening file: " << filename << "\n";
		return;
	}

	for (size_t i = 0; i < t.size(); ++i)
	{
		file << t[i];
		for (size_t j = 0; j < y[i].size(); ++j)
		{
			file << " " << y[i][j];
		}
		file << "\n";
	}
	file.close();
}

int main()
{
	// // Model: dy/dt = y (simple exponential growth)
	// auto model = [](double t, const std::vector<double>& y) -> std::vector<double> {
	// 	return {y[0]}; // dy/dt = y (for a scalar ODE)
	// };
	// std::vector<double> initial_conditions = {1.0};

	// // RLC circuit model: damped oscillation
	// auto model = [](double t, const std::vector<double> &y) -> std::vector<double>
	// {
	// 	double Q = y[0]; // Charge
	// 	double I = y[1]; // Current

	// 	// Define RLC circuit parameters
	// 	double L = 0.2; // Inductance (H)
	// 	double R = 0.5; // Resistance (Ohm)
	// 	double C = 1.0; // Capacitance (F)

	// 	// Step at t=1
	// 	if (t >= 1.0)
	// 		I += 1.0;

	// 	double dQdt = I;
	// 	double dIdt = (-R / L) * I - (1.0 / (L * C)) * Q;

	// 	return {dQdt, dIdt};
	// };
	// std::vector<double> initial_conditions = {0.0, 0.0}; // Q = 1C, I = 0A

	// Stiff van der Pol oscillator model
	auto model = [](double t, const std::vector<double>& y) -> std::vector<double>
	{
		double y1 = y[0];
		double y2 = y[1];
		double mu = 10.0; // Large parameter for stiffness

		double dy1dt = y2;
		double dy2dt = mu * (1 - y1 * y1) * y2 - y1;

		return {dy1dt, dy2dt};
	};
	std::vector<double> initial_conditions = {2.0, 0.0}; 

	// Initial condition and time range
	double t_start = 0.0;
	double t_end = 30.0;
	double dt = 0.1;

	// Create solvers
	ForwardEuler solver_fe(model);
	RK4 solver_rk4(model);
	RK45 solver_rk45{model};
	RadauIIA solver_rd2a{model};

	// Solve the ODE with Forward Euler
	auto result_fe = solver_fe.solve(initial_conditions, t_start, t_end, dt);
	auto t_fe = result_fe.first;
	auto y_fe = result_fe.second;

	// Solve the ODE with Runge-Kutta 4
	auto result_rk4 = solver_rk4.solve(initial_conditions, t_start, t_end, dt);
	auto t_rk4 = result_rk4.first;
	auto y_rk4 = result_rk4.second;

	// Solve the ODE with Runge-Kutta 45
	auto result_rk45 = solver_rk45.solve(initial_conditions, t_start, t_end, dt);
	auto t_rk45 = result_rk45.first;
	auto y_rk45 = result_rk45.second;

	auto result_rda2 = solver_rd2a.solve(initial_conditions, t_start, t_end, dt);
	auto t_rda2 = result_rda2.first;
	auto y_rda2 = result_rda2.second;

	// Save results using the generalized function
	// save_results("results_fe.txt", t_fe, y_fe);
	// save_results("results_rk4.txt", t_rk4, y_rk4);
	save_results("results_rk45.txt", t_rk45, y_rk45);
	save_results("results_rda2.txt", t_rda2, y_rda2);

	return 0;
}
