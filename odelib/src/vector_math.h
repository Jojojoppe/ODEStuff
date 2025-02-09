#pragma once
#include <vector>
#include <cmath>

// Vector addition
inline std::vector<double> operator+(const std::vector<double> &a, const std::vector<double> &b)
{
	std::vector<double> result(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		result[i] = a[i] + b[i];
	return result;
}

// Vector subtraction
inline std::vector<double> operator-(const std::vector<double> &a, const std::vector<double> &b)
{
	std::vector<double> result(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		result[i] = a[i] - b[i];
	return result;
}

// Scalar multiplication
inline std::vector<double> operator*(double scalar, const std::vector<double> &v)
{
	std::vector<double> result(v.size());
	for (size_t i = 0; i < v.size(); ++i)
		result[i] = scalar * v[i];
	return result;
}

// Scalar multiplication (commutative)
inline std::vector<double> operator*(const std::vector<double> &v, double scalar)
{
	return scalar * v;
}

// Compute Euclidean norm of a vector
inline double norm(const std::vector<double> &v)
{
	double sum = 0.0;
	for (auto val : v)
		sum += val * val;
	return std::sqrt(sum);
}

