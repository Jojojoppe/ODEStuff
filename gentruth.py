from scipy.integrate import solve_ivp

# class Model:
#   def __call__(self, t, y):
#     return y

# class Model:
#     def __call__(self, t, y):
#         Q, I = y  # Charge and Current

#         # Define RLC circuit parameters
#         L = 0.2  # Inductance (H)
#         R = 0.5  # Resistance (Ohm)
#         C = 1.0  # Capacitance (F)

#         if t>= 1.0:
#            I += 1.0

#         dQdt = I
#         dIdt = (-R / L) * I - (1.0 / (L * C)) * Q

#         return [dQdt, dIdt]

class Model:
    def __call__(self, t, y):
        y1, y2 = y  # state variables

        # Stiff van der Pol parameter (μ large makes the problem stiff)
        mu = 10.0

        # van der Pol equations:
        #   y1' = y2
        #   y2' = μ*(1 - y1^2)*y2 - y1
        dy1dt = y2
        dy2dt = mu * (1 - y1 * y1) * y2 - y1

        return [dy1dt, dy2dt]
  
model = Model()
sol = solve_ivp(model, t_span=(0, 30), y0=[2.0, 0], max_step=0.1, method='Radau')

with open('results_truth.txt', 'w') as f:
  for i, t in enumerate(sol.t):
    f.write(f'{t}')
    for yv in sol.y:
      f.write(f' {yv[i]}')
    f.write('\r\n')