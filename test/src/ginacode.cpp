#include <ginac/basic.h>
#include <ginac/ex.h>
#include <iostream>
#include <fstream>
#include <map>
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

class ODESystem
{
public:
	ODESystem(const ginac::lst &diff_eqs, const ginac::lst &init_cnds,
			  const ginac::lst &params, const ginac::lst &variables, const ginac::symbol &t)
		: m_diffEqs(diff_eqs)
		, m_initCnds(init_cnds)
		, m_params(params)
		, m_variables(variables) 
		, m_t(t)
	{
		mapItems(m_initCnds, m_initCndsMap);
		mapItems(m_params, m_paramsMap);
		mapItems(m_variables, m_variablesMap);
		getStateVariables();
	}

	std::vector<double> getInitialState() const
	{
		std::vector<double> initialStates(m_stateVariables.size());
		for(const auto& ic : m_initCnds)
		{
			ginac::symbol s = ginac::ex_to<ginac::symbol>(ic.lhs());
			ginac::ex eval = ic.rhs().evalf();
			std::cout << s << " ";
			if(ginac::is_a<ginac::numeric>(eval))
			{
				double v = ginac::ex_to<ginac::numeric>(eval).to_double();
				if(m_stateVariables.count(s.gethash())>0)
				{
					auto idx = m_stateVariables.at(s.gethash());
					std::cout << "idx " << idx << " value " << v << std::endl;
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
		return {};
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

	void mapItems(const ginac::lst& list, std::map<unsigned int, size_t>& map)
	{
		size_t i = 0;
		for(const auto& s : list)
		{
			// Check if s is a symbol
			if(ginac::is_a<ginac::symbol>(s))
			{
				map[ginac::ex_to<ginac::symbol>(s).gethash()] = i;
			}
			// Check if s is an equation, if so if the lhs is a symbol
			else if(ginac::is_a<ginac::relational>(s) && ginac::is_a<ginac::symbol>(s.lhs()))
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
		for(const auto& eq : m_diffEqs)
		{
			// If an equation starts with a Ddt its content is a state variable
			if(ginac::is_a<ginac::relational>(eq))
			{
				if (eq.lhs().match(Ddt(ginac::wild())))
				{
					// Ddt(..) == ...
					if(ginac::is_a<ginac::symbol>(eq.lhs().op(0)))
					{
						// State variable found
						std::cout << "state variable: " << eq.lhs().op(0) << " idx " << stateVariables << std::endl;
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
					if(eq.lhs().nops() == 1)
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
	ginac::symbol t("t");
	ginac::lst diff_eqs = {
		Ddt(Q) == I,
		Ddt(I) == (-R / L) * I - (1.0 / (L * C)) * Q,
	};
	ginac::lst init_cnds = {I == 1.0, Q == 0};
	ginac::lst params = {R == 0.1, L == 0.2, C == 0.4};
	ginac::lst variables = {};

	std::cout << "diff_eqs:\t" << diff_eqs << std::endl;
	std::cout << "init_cnds:\t" << init_cnds << std::endl;

	ODESystem system{diff_eqs, init_cnds, params, variables, t};
	RK45 solver(system);

	auto is = system.getInitialState();

	// auto result = solver.solve(system.getInitialState(), 0, 15, 0.1);
	// save_results("results.txt", result.first, result.second);

	return 0;
}
