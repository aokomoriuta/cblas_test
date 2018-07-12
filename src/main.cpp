#include <iostream>
#include <memory>
#include <random>
#include <chrono>

#include <cblas.h>

// C = AB
int main()
{
	constexpr auto N = std::size_t(100000);
	constexpr auto M = std::size_t(10000);
	constexpr auto L = std::size_t(1000);

	constexpr auto N_A = N;
	constexpr auto M_A = L;
	constexpr auto N_B = L;
	constexpr auto M_B = M;
	constexpr auto N_C = N;
	constexpr auto M_C = M;

	auto A = std::make_unique<double[]>(N_A * M_A);
	auto B = std::make_unique<double[]>(N_B * M_B);
	auto C = std::make_unique<double[]>(N_C * M_C);

	auto generator = std::mt19937(12345);
	auto distribution = std::uniform_real_distribution<double>(-10, 10);

	for(auto i = decltype(N_A*M_A)(0); i < N_A*M_A; ++i)
	{
		A[i] = distribution(generator);
	}
	for(auto i = decltype(N_B*M_B)(0); i < N_B*M_B; ++i)
	{
		B[i] = distribution(generator);
	}
	std::fill_n(C.get(), N_C*M_C, 0);

	constexpr auto ORDER = ::CblasRowMajor;
	constexpr auto TRANS_A = ::CblasNoTrans;
	constexpr auto TRANS_B = ::CblasNoTrans;
	constexpr auto ALPHA = double(1);
	constexpr auto BETA = double(0);

	std::cout << N_A << "x" << M_A << " * " << N_B << "x" << M_B << std::endl;
	const auto begin = std::chrono::system_clock::now();
	::cblas_dgemm(ORDER, TRANS_A, TRANS_B, N, M, L, ALPHA,
		A.get(), M_A,
		B.get(), M_B, BETA,
		C.get(), M_C);
	const auto end = std::chrono::system_clock::now();
	const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	const auto gflops = N*(2*M - 1)*L / us * 1e-3;
	std::cout << us*1e-3 << "[ms]" << std::endl;
	std::cout << gflops << "[GFLOP/s]" << std::endl;

	return 0;
}
