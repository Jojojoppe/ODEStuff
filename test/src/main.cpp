#include "odesolver.h"
#include "solver_euler.h"
#include "solver_rk.h"
#include "solver_radau.h"
#include "solver_bdf.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <string>

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

class Timer
{
public:
	Timer(const std::string &timer_name) : name(timer_name)
	{
		running = true;
		start_time = std::chrono::high_resolution_clock::now();
	}

	~Timer()
	{
		stop();
	}

	void stop()
	{
		if (running)
		{
			auto end_time = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = end_time - start_time;
			std::cout << name << " elapsed time: " << elapsed.count() << " seconds\n";
			running = false;
		}
	}

private:
	bool running{false};
	std::string name;
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

int main(int argc, char **argv)
{
	auto model = [](double t, const std::vector<double> &y) -> std::vector<double>
	{
		double Q = y[0]; // Charge
		double I = y[1]; // Current

		// Define RLC circuit parameters
		double L = 0.1; // Inductance (H)
		double R = 0.1; // Resistance (Ohm)
		double C = 0.3; // Capacitance (F)

		// Step at t=1
		if (t >= 1.0)
			I += 1.0;

		double dQdt = I;
		double dIdt = (-R / L) * I - (1.0 / (L * C)) * Q;

		return {dQdt, dIdt};
	};
	const std::vector<double> initial_conditions = {0.0, 0.0}; // Q = 1C, I = 0A
	const double t_start = 0.0;
	const double t_end = 15.0;
	const double dt = 0.1;

	{
		ForwardEuler solver{model};
		Timer t{"fe"};
		auto res = solver.solve(initial_conditions, t_start, t_end, dt);
		t.stop();
		// save_results("results_fe.txt", res.first, res.second);
	}

	{
		RK4 solver{model};
		Timer t{"rk4"};
		auto res = solver.solve(initial_conditions, t_start, t_end, dt);
		t.stop();
		save_results("results_rk4.txt", res.first, res.second);
	}

	{
		RK45 solver{model};
		Timer t{"rk45"};
		auto res = solver.solve(initial_conditions, t_start, t_end, dt);
		t.stop();
		save_results("results_rk45.txt", res.first, res.second);
	}

	{
		RadauIIA solver{model, 1e-3};
		Timer t{"rda2"};
		auto res = solver.solve(initial_conditions, t_start, t_end, dt);
		t.stop();
		save_results("results_rda2.txt", res.first, res.second);
	}

	{
		BDF solver{model, 1e-3};
		Timer t{"bdf"};
		auto res = solver.solve(initial_conditions, t_start, t_end, dt);
		t.stop();
		save_results("results_bdf.txt", res.first, res.second);
	}

	return 0;
}