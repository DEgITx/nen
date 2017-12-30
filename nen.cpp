#include "nen.hpp"

int main()
{
	//srand(time(NULL));
	NEN::NeuronNetwork n(2, 1, 1, 3, NEN::StochasticGradient);
	n.rate = 0.2;
	n.momentum = 0.7;
	//NeuronNetwork n(2, 1, 25, 25, Adagrad);
	
	//n.gpu = true;

	/*
	std::ofstream f;
	f.open("mul.data");
	for (double i = 2; i <= 10; i++)
		for (double j = 2; j <= 10; j++)
		{
			double x = log(i) / log(100);
			double y = log(j) / log(100);
			double xy = log(i * j) / log(100);
			f << x << " " << y << " " << xy << "\n";
			std::cout << exp(xy * log(100)) << "\n";
		}
	f.close();
	*/

	/*
	std::ofstream f;
	f.open("add.data");
	for (double i = 0; i <= 25; i++)
	{
		for (double j = 0; j <= 25; j++)
		{
			double x = i / 50;
			double y = j / 50;
			double xy = (i + j) / 50;
			f << x << " " << y << " " << xy << "\n";
			std::cout << xy * 50 << "\n";
		}
	}
	f.close();
	*/

	/*
	auto start = std::chrono::high_resolution_clock::now();
	n.trainWhileError({
		normalizeInput({ log(1), log(3) }, 0, 10),
		normalizeInput({ log(2), log(7) }, 0, 10),
		normalizeInput({ log(6), log(5) }, 0, 10),
		normalizeInput({ log(5), log(5) }, 0, 10),
		normalizeInput({ log(2), log(3) }, 0, 10),
		normalizeInput({ log(1), log(8) }, 0, 10),
		normalizeInput({ log(7), log(7) }, 0, 10),
		normalizeInput({ log(3), log(6) }, 0, 10),
		normalizeInput({ log(6), log(6) }, 0, 10),
		normalizeInput({ log(8), log(4) }, 0, 10),
		normalizeInput({ log(10), log(5) }, 0, 10),
		normalizeInput({ log(6), log(7) }, 0, 10),
		normalizeInput({ log(8), log(8) }, 0, 10),
		normalizeInput({ log(9), log(9) }, 0, 10),
		normalizeInput({ log(5), log(8) }, 0, 10),
		normalizeInput({ log(5), log(7) }, 0, 10),
	}, {
		normalizeInput(std::vector<double>{ log(3) }, 0, 10),
		normalizeInput(std::vector<double>{ log(14) }, 0, 10),
		normalizeInput(std::vector<double>{ log(30) }, 0, 10),
		normalizeInput(std::vector<double>{ log(25) }, 0, 10),
		normalizeInput(std::vector<double>{ log(6) }, 0, 10),
		normalizeInput(std::vector<double>{ log(8) }, 0, 10),
		normalizeInput(std::vector<double>{ log(49) }, 0, 10),
		normalizeInput(std::vector<double>{ log(18) }, 0, 10),
		normalizeInput(std::vector<double>{ log(36) }, 0, 10),
		normalizeInput(std::vector<double>{ log(32) }, 0, 10),
		normalizeInput(std::vector<double>{ log(50) }, 0, 10),
		normalizeInput(std::vector<double>{ log(42) }, 0, 10),
		normalizeInput(std::vector<double>{ log(64) }, 0, 10),
		normalizeInput(std::vector<double>{ log(81) }, 0, 10),
		normalizeInput(std::vector<double>{ log(40) }, 0, 10),
		normalizeInput(std::vector<double>{ log(35) }, 0, 10),
	}, 0, 1);
	//}, 0, 0.1);
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	std::cout << "time: " << diff / (1000 * 1000) << " ms" << std::endl;

	n.forward(normalizeInput({ log(8), log(8) }, 0, 10));
	for (auto& o : n.output())
		std::cout << "out " << exp(deNormalizeOutput(o, 0, 10)) << std::endl;
	//n.saveFile("mul.ner");

	//n.setAutoSaveFile("add.ner");
	//n.loadTrainData("add.data");
	//n.trainWhileError(0, 0.5);
	//auto result = n.get({ log(2) / log(100), log(2) / log(100) });
	//std::cout << "out " << exp(result[0] * log(100)) << std::endl;
	*/

	auto start = std::chrono::high_resolution_clock::now();
	auto a = std::vector<std::vector<double>>{ { 0, 0 },{ 1, 0 },{ 0, 1 },{ 1, 1 }};
	auto b = std::vector<std::vector<double>>{ { 0 },{ 1 },{ 1 },{ 0 } };
	n.trainWhileError(a, b, 0, 0.5);
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
	std::cout << "time: " << diff << " ms" << std::endl;
	std::cout << n.iterations << " it\n";
	std::cout << n.get({ 0, 1 })[0] << "\n";
	std::cout << n.get({ 1, 1 })[0] << "\n";

    return 0;
}
