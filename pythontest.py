import numpy as np

class ODESolver:
	def __init__(self, derivatives):
		"""
		Initialize the ODE solver with a derivatives function.
		
		Parameters:
		-----------
		derivatives : callable
			A function that takes the current state and time, and returns the derivatives.
		"""
		self.derivatives = lambda t, y: np.asarray(derivatives(t, y), float)

	def solve(self, initial_conditions, t_span, dt=1e-2):
		"""
		Solve the ODE over a given time region with a fixed timestep.
		
		Parameters:
		-----------
		initial_conditions : array_like
			The initial state of the system.
		t_span : tuple
			A tuple (t_start, t_end) specifying the start and end times.
		dt : float
			The fixed timestep for the solver.
		
		Returns:
		--------
		t : numpy.ndarray
			Array of time points.
		y : numpy.ndarray
			Array of solution values at each time point.
		"""	
		t_start, t_end = t_span
		t = np.arange(t_start, t_end + dt, dt)
		y = np.zeros((len(t), len(initial_conditions)))
		y[0] = initial_conditions

		for i in range(1, len(t)):
			y[i] = self.step(t[i-1], y[i-1], dt)

		return t, y

	def step(self, t, y, dt):
		"""
		Perform a single step of the ODE solver.
		
		Parameters:
		-----------
		t : float
			The current time.
		y : array_like
			The current state of the system.
		dt : float
			The timestep.
		
		Returns:
		--------
		y_new : array_like
			The new state of the system after the step.
		"""
		raise NotImplementedError("Step function must be implemented in derived class.")


class ForwardEuler(ODESolver):
	def step(self, t, y, dt):
		dydt = self.derivatives(t, y)
		y_new = y + dt * dydt
		return y_new
	
class RK4(ODESolver):
    def step(self, t, y, dt):
        k1 = self.derivatives(t, y)
        k2 = self.derivatives(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self.derivatives(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self.derivatives(t + dt, y + dt * k3)
        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_new


import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
if __name__ == "__main__":

	class Model:
		def __call__(self, t, y):
			return y

	plt.figure()

	model = Model()
	solver = ForwardEuler(model)
	t, y = solver.solve(initial_conditions=[1], t_span=(0, 5), dt=0.1)
	plt.plot(t, y, label='FE')

	model = Model()
	solver = RK4(model)
	t, y = solver.solve(initial_conditions=[1], t_span=(0, 5), dt=0.1)
	plt.plot(t, y, label='RK4')

	sol = solve_ivp(model, t_span=(0, 5), y0=[1], max_step=0.1)
	plt.plot(sol.t, sol.y[0], '.')
	
	plt.legend()
	plt.show()