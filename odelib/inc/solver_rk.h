#pragma once
#include "odesolver.h"
#include <functional>
#include <vector>

/**
 * @brief Runge-Kutta 4th order method for solving ordinary differential equations.
 *
 * This class implements the Runge-Kutta 4th order (RK4) method, a widely used
 * explicit method for numerically integrating ordinary differential equations.
 * It is derived from the ODESolver base class and overrides the step method
 * to perform one integration step using the RK4 scheme.
 */
class RK4 : public ODESolver
{
public:
	/**
	 * @brief Constructor.
	 *
	 * @param derivatives A function representing the derivative of the state, i.e., f(t, y).
	 */
	RK4(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives);
    /**
     * @brief Constructor.
     *
     * @param derivatives A function representing the derivative of the state, i.e. f(t,y)
     * @param nonStateVariables A function calculating extra (non-state) variables from the state and time (g(t, y))
     */
    RK4(std::function<std::vector<double>(double, const std::vector<double>&)> derivatives,
        std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables);

protected:
	/**
	 * @brief Performs one integration step using the Runge-Kutta 4th order method.
	 *
	 * @param t The current time.
	 * @param y The current state vector.
	 * @param dt The timestep to attempt.
	 * @return A StepResult struct with the new state and timestep information.
	 */
	StepResult step(double t, const std::vector<double> &y, double dt) override;
};

/**
 * @brief Runge-Kutta 4th and 5th order method for solving ordinary differential equations (Dormand-Prince method).
 *
 * This class implements the adaptive Runge-Kutta method known as the Dormand-Prince method (RK45).
 * It adapts the step size to meet a user-defined tolerance and uses a pair of solutions of different orders
 * (4th and 5th order) to estimate the error at each step.
 */
class RK45 : public ODESolver
{
public:
	/**
	 * @brief Constructor.
	 *
	 * @param f A function representing the derivative of the state, i.e., f(t, y).
	 * @param tol The tolerance for the error estimation (default is 1e-6).
	 * @param max_step The maximum allowed step size (default is 1).
	 */
	RK45(std::function<std::vector<double>(double, const std::vector<double> &)> f, double tol = 1e-6, double max_step = 1);
    /**
     * @brief Constructor.
     *
     * @param derivatives A function representing the derivative of the state, i.e. f(t,y)
     * @param nonStateVariables A function calculating extra (non-state) variables from the state and time (g(t, y))
	 * @param tol The tolerance for the error estimation (default is 1e-6).
	 * @param max_step The maximum allowed step size (default is 1).
     */
    RK45(std::function<std::vector<double>(double, const std::vector<double>&)> derivatives,
        std::function<std::vector<double>(double, const std::vector<double>&)> nonStateVariables, 
		double tol=1e-6, double max_step = 1);

protected:
	/**
	 * @brief Performs one integration step using the Runge-Kutta 4th and 5th order method (Dormand-Prince).
	 *
	 * This method computes both the 4th and 5th order solutions and estimates the error between them.
	 * The step size is adjusted dynamically based on the error to ensure that the solution meets the
	 * specified tolerance.
	 *
	 * @param t The current time.
	 * @param y The current state vector.
	 * @param dt The timestep to attempt.
	 * @return A StepResult struct with the new state and timestep information.
	 */
	StepResult step(double t, const std::vector<double> &y, double dt) override;

private:
	const double tol;			  ///< The tolerance for error estimation.
	const double safety = 0.9;	  ///< Safety factor for step size adjustment.
	const double min_scale = 0.2; ///< Minimum scale factor for step size adjustment.
	const double max_scale = 5.0; ///< Maximum scale factor for step size adjustment.
	const double max_step;		  ///< The maximum allowed step size.
};
