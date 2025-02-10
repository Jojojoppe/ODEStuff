#include <iostream>
#include <fstream>
#include <map>
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
    ODESystem(const ginac::lst &diff_eqs, const ginac::lst &init_cnds, const ginac::lst &params, const ginac::symbol &t)
        : m_diff_eqs(diff_eqs), m_init_cnds(init_cnds), m_params(params), m_t(t)
    {
        // Extract variables from initial conditions
        for (const auto &ic : m_init_cnds)
        {
            ginac::symbol var = ginac::ex_to<ginac::symbol>(ic.lhs());
            m_variables.push_back(var);
            m_var_index[var.gethash()] = m_variables.size() - 1;
        }
        // Determine the state variable of each equation
        for (const auto &de : m_diff_eqs)
        {
            ginac::ex lhs = de.lhs();
            ginac::symbol svar;
            // Double check if it only contains one thing and is or contains a symbol
            if (ginac::is_a<ginac::symbol>(lhs))
            {
                svar = ginac::ex_to<ginac::symbol>(lhs);
            }
            else if (lhs.nops() == 1 && ginac::is_a<ginac::symbol>(lhs.op(0)))
            {
                svar = ginac::ex_to<ginac::symbol>(lhs.op(0));
            }
            else
            {
                std::cerr << "lhs: " << lhs << std::endl;
                throw std::runtime_error("(Differential) equation lhs is not in the form of Ddt(x) or x");
            }
            // Check if svar in map
            if (m_var_index.count(svar.gethash()) > 0)
            {
                m_eq_to_variable.push_back(svar.gethash());
            }
            else
            {
                throw std::runtime_error("Symbol " + svar.get_name() + " is not in state variable list");
            }
        }
    }

    std::vector<double> get_initial_state() const
    {
        std::vector<double> initial_state(m_variables.size());
        for (const auto &ic : m_init_cnds)
        {
            ginac::symbol var = ginac::ex_to<ginac::symbol>(ic.lhs());
            ginac::ex rhs_evaluated = ic.rhs().evalf();
            if (ginac::is_a<ginac::numeric>(rhs_evaluated))
            {
                double value = ginac::ex_to<ginac::numeric>(rhs_evaluated).to_double();
                auto idx = m_var_index.at(var.gethash());
                initial_state[idx] = value;
            }
            else
            {
                throw std::runtime_error("Initial condition for variable " + var.get_name() + " did not evaluate to a numeric value.");
            }
        }
        return initial_state;
    }

    std::vector<double> operator()(double time, const std::vector<double> &state)
    {
        std::vector<double> derivatives(state.size());

        // Create a substitution list for the current state and time
        ginac::lst substitutions;
        for (const auto &var : m_variables)
        {
            substitutions.append(var == state[m_var_index[var.gethash()]]);
        }
        substitutions.append(m_t == time);
        for (const auto &p : m_params)
        {
            substitutions.append(p);
        }

        // Evaluate each differential equation
        for (size_t i = 0; i < m_diff_eqs.nops(); ++i)
        {
            GiNaC::ex eq = m_diff_eqs.op(i);
            GiNaC::ex evaluated_derivative = eq.rhs().subs(substitutions);
            size_t idx = m_var_index[m_eq_to_variable[i]];
            derivatives[idx] = GiNaC::ex_to<GiNaC::numeric>(evaluated_derivative.evalf()).to_double();
        }

        return derivatives;
    }

private:
    ginac::lst m_diff_eqs;
    ginac::lst m_init_cnds;
    ginac::lst m_params;
    std::vector<ginac::symbol> m_variables;
    std::vector<unsigned int> m_eq_to_variable;
    std::map<unsigned int, size_t> m_var_index;
    ginac::symbol m_t;
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
    ginac::symbol I("I"), Q("Q");         // State variables
    ginac::symbol L("L"), R("R"), C("C"); // Parameters
    ginac::symbol t("t");
    ginac::lst diff_eqs = {
        Ddt(Q) == I,
        Ddt(I) == (-R / L) * I - (1.0 / (L * C)) * Q,
    };
    ginac::lst init_cnds = {I == 1.0, Q == 0};
    ginac::lst params = {R == 0.1, L == 0.2, C == 0.4};

    std::cout << "diff_eqs:\t" << diff_eqs << std::endl;
    std::cout << "init_cnds:\t" << init_cnds << std::endl;

    ODESystem system{diff_eqs, init_cnds, params, t};
    RK45 solver(system);
    auto result = solver.solve(system.get_initial_state(), 0, 15, 0.1);
    save_results("results.txt", result.first, result.second);

    return 0;
}