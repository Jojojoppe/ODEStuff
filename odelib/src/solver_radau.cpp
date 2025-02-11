#include "solver_radau.h"
#include "vector_math.h"
#include <cmath>
#include <algorithm>

// Constructor for the RadauIIA solver.
RadauIIA::RadauIIA(std::function<std::vector<double>(double, const std::vector<double> &)> f, double local_tol, double max_step)
	: ODESolver(f), local_tol(local_tol), max_step(max_step)
{
}

RadauIIA::RadauIIA(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives,
	std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables,
	double tol, double max_step)
	: ODESolver(derivatives, nonStateVariables), local_tol(tol), max_step(max_step)
{
}

// Performs a single step of the RadauIIA integration method.
ODESolver::StepResult RadauIIA::step(double t, const std::vector<double> &y, double dt)
{
	int n = y.size();
	double dt_current = dt;
	StepResult result;

	while (true)
	{
		std::vector<double> X(2 * n, 0.0);
		std::vector<double> guess1 = derivatives_func(t + dt_current / 3.0, y);
		std::vector<double> guess2 = derivatives_func(t + dt_current, y);

		for (int i = 0; i < n; i++)
		{
			X[i] = guess1[i];
			X[n + i] = guess2[i];
		}

		const int maxIter = 20;
		const double newton_tol = 1e-8;

		for (int iter = 0; iter < maxIter; iter++)
		{
			std::vector<double> F_val = computeF(X, t, y, dt_current);
			double F_norm = vectorNorm(F_val);
			if (F_norm < newton_tol)
				break;
			std::vector<std::vector<double>> J = computeJacobian(X, t, y, dt_current);
			std::vector<double> delta = solveLinearSystem(J, F_val);
			for (size_t i = 0; i < X.size(); i++)
			{
				X[i] -= delta[i];
			}
			if (vectorNorm(delta) < newton_tol)
				break;
		}

		std::vector<double> K1(X.begin(), X.begin() + n);
		std::vector<double> K2(X.begin() + n, X.end());
		std::vector<double> y_high(n, 0.0);
		std::vector<double> y_low(n, 0.0);

		for (int i = 0; i < n; i++)
		{
			y_high[i] = y[i] + dt_current * (0.75 * K1[i] + 0.25 * K2[i]);
			y_low[i] = y[i] + dt_current * (0.5 * K1[i] + 0.5 * K2[i]);
		}

		std::vector<double> err(n, 0.0);
		for (int i = 0; i < n; i++)
		{
			err[i] = y_high[i] - y_low[i];
		}

		double err_norm = norm(err);
		double dt_new = dt_current * safety * std::pow(local_tol / (err_norm + 1e-10), 0.25);
		dt_new = std::max(dt_new, min_scale * dt_current);
		dt_new = std::min(dt_new, max_scale * dt_current);
		dt_new = std::min(dt_new, max_step);

		if (err_norm <= local_tol)
		{
			result.y = y_high;
			result.dt_accepted = dt_current;
			result.dt_new = dt_new;
			break;
		}
		else
		{
			dt_current = dt_new;
		}
	}

	return result;
}

// Computes the Euclidean norm of a vector.
double RadauIIA::vectorNorm(const std::vector<double> &v)
{
	double sum = 0.0;
	for (double x : v)
		sum += x * x;
	return std::sqrt(sum);
}

// Computes the nonlinear function F(X) = 0 for stage equations.
std::vector<double> RadauIIA::computeF(const std::vector<double> &X, double t, const std::vector<double> &y, double dt)
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

// Computes the Jacobian matrix for the nonlinear system.
std::vector<std::vector<double>> RadauIIA::computeJacobian(const std::vector<double> &X, double t, const std::vector<double> &y, double dt)
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

// Solves the linear system J * delta = F using Gaussian elimination.
std::vector<double> RadauIIA::solveLinearSystem(std::vector<std::vector<double>> A, std::vector<double> b)
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
		int pivot = i;
		for (int j = i + 1; j < n; j++)
		{
			if (std::fabs(A[j][i]) > std::fabs(A[pivot][i]))
				pivot = j;
		}
		if (std::fabs(A[pivot][i]) < 1e-12)
			continue; // Singular matrix.
		std::swap(A[i], A[pivot]);
		double pivotVal = A[i][i];
		for (int j = i; j <= n; j++)
		{
			A[i][j] /= pivotVal;
		}
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
