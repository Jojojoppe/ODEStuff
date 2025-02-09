#pragma once
#include "odesolver.h"
#include <vector>
#include <functional>

class BDF : public ODESolver
{
public:
	/**
	 * Constructor for the BDF solver.
	 * @param f Function representing the system of ODEs.
	 * @param local_tol Local tolerance for adaptive step size.
	 * @param max_step Maximum allowable step size.
	 */
	BDF(std::function<std::vector<double>(double, const std::vector<double> &)> f, double local_tol = 1e-6, double max_step = 1);

protected:
	/**
	 * Performs a single step of the BDF integration method.
	 * Iterates until the error estimate is below the specified tolerance.
	 * @param t Current time.
	 * @param y Current state vector.
	 * @param dt Current time step.
	 * @return StepResult containing the new state and step information.
	 */
	StepResult step(double t, const std::vector<double> &y, double dt) override;

private:
	const double local_tol;		  ///< The tolerance for error estimation.
	const double safety = 0.9;	  ///< Safety factor for step size adjustment.
	const double min_scale = 0.2; ///< Minimum scale factor for step size adjustment.
	const double max_scale = 5.0; ///< Maximum scale factor for step size adjustment.
	const double max_step;		  ///< The maximum allowed step size.

	/**
	 * Computes the Euclidean norm of a vector.
	 * @param v Input vector.
	 * @return Norm of the vector.
	 */
	double vectorNorm(const std::vector<double> &v);

	/**
	 * Computes the nonlinear function F(X) = 0 for stage equations.
	 * @param X Input vector [K1, K2, ..., Ks].
	 * @param t Current time.
	 * @param y Current state vector.
	 * @param dt Time step.
	 * @return The function values at the given point.
	 */
	std::vector<double> computeF(const std::vector<double> &X, double t, const std::vector<double> &y, double dt);

	/**
	 * Computes the Jacobian matrix for the nonlinear system.
	 * @param X Input vector [K1, K2, ..., Ks].
	 * @param t Current time.
	 * @param y Current state vector.
	 * @param dt Time step.
	 * @return The Jacobian matrix.
	 */
	std::vector<std::vector<double>> computeJacobian(const std::vector<double> &X, double t, const std::vector<double> &y, double dt);

	/**
	 * Solves the linear system J * delta = F using Gaussian elimination.
	 * @param A Coefficient matrix.
	 * @param b Right-hand side vector.
	 * @return Solution vector.
	 */
	std::vector<double> solveLinearSystem(std::vector<std::vector<double>> A, std::vector<double> b);
};
