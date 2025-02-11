#include <ginac/basic.h>
#include <ginac/ex.h>
#include <ginac/function.h>
#include <iostream>
#include <fstream>
#include <map>
#include <ostream>
#include <ranges>
#include <stdexcept>
#include <string>

#include <ginac/ginac.h>

#include <odesolver.h>
#include <solver_rk.h>

namespace ginac = GiNaC;

DECLARE_FUNCTION_1P(Ddt);
REGISTER_FUNCTION(Ddt, dummy());

DECLARE_FUNCTION_1P(Int);
REGISTER_FUNCTION(Int, dummy());

DECLARE_FUNCTION_3P(if_else);
static ginac::ex if_else_evalf(const ginac::ex &cond, const ginac::ex &tbranch, const ginac::ex &fbranch)
{
	ginac::ex cond_eval = cond.evalf();
	if (ginac::is_a<ginac::relational>(cond_eval))
	{
		const ginac::relational &rel = ginac::ex_to<ginac::relational>(cond_eval);
		return (rel) ? tbranch : fbranch;
	}
	else if (ginac::is_a<ginac::numeric>(cond_eval))
	{
		double val = ginac::ex_to<ginac::numeric>(cond_eval).to_double();
		return (val != 0.0) ? tbranch : fbranch;
	}
	else
	{
		return if_else(cond, tbranch, fbranch).hold();
	}
}
REGISTER_FUNCTION(if_else, evalf_func(if_else_evalf));

class ODESystem
{
public:
	ODESystem(const ginac::lst &diff_eqs, const ginac::lst &init_cnds,
			  const ginac::lst &params, const ginac::lst &variables, const ginac::symbol &t)
		: m_diffEqs(diff_eqs), m_initCnds(init_cnds), m_params(params), m_variables(variables), m_t(t)
	{
		mapItems(m_initCnds, m_initCndsMap);
		mapItems(m_params, m_paramsMap);
		mapItems(m_variables, m_variablesMap);
		getStateVariables();
	}

	std::vector<double> getInitialState() const
	{
		std::vector<double> initialStates(m_stateVariables.size());
		for (const auto &ic : m_initCnds)
		{
			ginac::symbol s = ginac::ex_to<ginac::symbol>(ic.lhs());
			ginac::ex eval = ic.rhs().evalf();
			if (ginac::is_a<ginac::numeric>(eval))
			{
				double v = ginac::ex_to<ginac::numeric>(eval).to_double();
				if (m_stateVariables.count(s.gethash()) > 0)
				{
					auto idx = m_stateVariables.at(s.gethash());
					initialStates[m_stateVariables.at(s.gethash())] = v;
				}
				else
				{
					std::cerr << "ERROR: " << ic << std::endl;
					throw std::runtime_error("Initial state variable not used in system");
				}
			}
			else
			{
				std::cerr << "ERROR: " << ic << std::endl;
				throw std::runtime_error("Right hand side of initial condition could not be evaluated until a numeric value");
			}
		}
		return initialStates;
	}

	std::vector<double> operator()(double time, const std::vector<double> &state)
	{
		std::vector<double> newState(state.size());
		std::vector<double> variables(m_updatedVariables.size());

		for (size_t i = 0; i < state.size(); i++)
		{
			newState[i] = state[i];
		}
		for (size_t i = 0; i < variables.size(); i++)
		{
			variables[i] = 0;
		}

		// for each solve equation
		for (const auto eq : m_diffEqs)
		{
			ginac::lst subs = m_params;
			// substitute t with time
			subs.append(m_t == time);
			// substitute variables
			size_t vI = 0;
			for (const auto &v : m_variables)
			{
				subs.append(v == variables[vI]);
				vI++;
			}
			// substitute state variables with state[i]
			for (const auto &s : m_initCnds)
			{
				subs.append(s.lhs() == state[m_stateVariables.at(s.lhs().gethash())]);
			}

			// solve
			ginac::ex toSolve = eq.rhs().subs(subs);
			if (eq.lhs().match(Ddt(ginac::wild())))
			{
				newState[m_stateVariables[eq.lhs().op(0).gethash()]] = ginac::ex_to<ginac::numeric>(toSolve.evalf()).to_double();
			}
			else
			{
				variables[m_updatedVariables[eq.lhs().gethash()]] = ginac::ex_to<ginac::numeric>(toSolve.evalf()).to_double();
			}
		}

		// return state variables
		return newState;
	}

private:
	const ginac::lst m_diffEqs;
	const ginac::lst m_initCnds;
	const ginac::lst m_params;
	const ginac::lst m_variables;
	const ginac::symbol m_t;
	std::map<unsigned int, size_t> m_initCndsMap;
	std::map<unsigned int, size_t> m_paramsMap;
	std::map<unsigned int, size_t> m_variablesMap;
	std::map<unsigned int, size_t> m_stateVariables;
	std::map<unsigned int, size_t> m_updatedVariables;

	void mapItems(const ginac::lst &list, std::map<unsigned int, size_t> &map)
	{
		size_t i = 0;
		for (const auto &s : list)
		{
			// Check if s is a symbol
			if (ginac::is_a<ginac::symbol>(s))
			{
				map[ginac::ex_to<ginac::symbol>(s).gethash()] = i;
			}
			// Check if s is an equation, if so if the lhs is a symbol
			else if (ginac::is_a<ginac::relational>(s) && ginac::is_a<ginac::symbol>(s.lhs()))
			{
				map[ginac::ex_to<ginac::symbol>(s.lhs()).gethash()] = i;
			}
			else
			{
				std::cerr << "ERROR: " << s << std::endl;
				throw std::runtime_error("Given item is not a symbol or symbol relation");
			}
		}
	}

	void getStateVariables()
	{
		size_t stateVariables = 0;
		size_t updatedVariables = 0;
		for (const auto &eq : m_diffEqs)
		{
			// If an equation starts with a Ddt its content is a state variable
			if (ginac::is_a<ginac::relational>(eq))
			{
				if (eq.lhs().match(Ddt(ginac::wild())))
				{
					// Ddt(..) == ...
					if (ginac::is_a<ginac::symbol>(eq.lhs().op(0)))
					{
						// State variable found
						m_stateVariables[eq.lhs().op(0).gethash()] = stateVariables++;
					}
					else
					{
						std::cerr << "ERROR: " << eq.lhs() << " " << eq.lhs().op(0) << std::endl;
						throw std::runtime_error("Given derivative has more than one variable in it or is not a symb");
					}
				}
				else
				{
					// ... == ...
					// This only is possible if no state variable functions are in yet
					if (stateVariables > 0)
					{
						std::cout << "ERROR: '" << eq << std::endl;
						throw std::runtime_error("State variable update is given before variable updates");
					}
					if (ginac::is_a<ginac::symbol>(eq.lhs()))
					{
						// Updated variable found
						m_updatedVariables[eq.lhs().gethash()] = updatedVariables++;
					}
					else
					{
						std::cerr << "ERROR: " << eq.lhs() << std::endl;
						throw std::runtime_error("Given variable assignment does not assign to a variable");
					}
				}
			}
			else
			{
				std::cerr << "ERROR: " << eq << std::endl;
				throw std::runtime_error("Given equation is not a relation (must contain ==)");
			}
		}
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

int main(int argc, char **argv)
{
	ginac::symbol I("I"), Q("Q");		  // State variables
	ginac::symbol L("L"), R("R"), C("C"); // Parameters
	ginac::symbol t("t"), x("x");
	ginac::lst diff_eqs = {
		x == I + if_else(t > 1.0, 1.0, 0.0),
		Ddt(Q) == x,
		Ddt(I) == ((-1 * R / L) * x) - ((1.0 / (L * C)) * Q),
	};
	ginac::lst init_cnds = {I == 0.0, Q == 0};
	ginac::lst params = {R == 0.1, L == 0.5, C == 0.5};
	ginac::lst variables = {x};

	std::cout << "diff_eqs:\t" << diff_eqs << std::endl;
	std::cout << "init_cnds:\t" << init_cnds << std::endl;

	ODESystem system{diff_eqs, init_cnds, params, variables, t};
	RK45 solver(system);

	// system(0, system.getInitialState());
	auto result = solver.solve(system.getInitialState(), 0, 10, 0.1);
	save_results("results.txt", result.first, result.second);

	return 0;
}
