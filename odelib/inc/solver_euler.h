#pragma once
#include "odesolver.h"
#include <functional>
#include <vector>

/**
 * @brief Forward Euler method for solving ordinary differential equations.
 *
 * This class implements the Forward Euler method, a simple explicit method for
 * numerically integrating ordinary differential equations. It is derived from
 * the ODESolver base class and overrides the step method to perform one
 * integration step using the Forward Euler scheme.
 */
class ForwardEuler : public ODESolver
{
public:
    /**
     * @brief Constructor.
     *
     * @param derivatives A function representing the derivative of the state, i.e., f(t, y).
     */
    ForwardEuler(std::function<std::vector<double>(double, const std::vector<double> &)> derivatives);

protected:
    /**
     * @brief Performs one integration step using the Forward Euler method.
     *
     * @param t The current time.
     * @param y The current state vector.
     * @param dt The timestep to attempt.
     * @return A StepResult struct with the new state and timestep information.
     */
    StepResult step(double t, const std::vector<double> &y, double dt) override;
};
