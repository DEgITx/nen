#include "nen.hpp"
#if defined(_WIN32) || defined(WIN32)
#include <windows.h>
#endif

int main(int argc, char* argv[])
{
	srand(argc > 1 ? atoi(argv[1]) : 1);
	bool* best = new bool[10];
	for (int i = 0; i < 10; i++)
		best[i] = rand() % 2;

	int prise[10] = {
		10,
		5,
		21,
		2,
		8,
		32,
		2,
		4,
		11,
		16
	};
	std::string names[10] = {
		"stol",
		"kolonka",
		"holodilnic",
		"pepelnica",
		"dver",
		"televizor",
		"sigareti",
		"eda",
		"telefon",
		"noutbuk"
	};
	int w[10] = {
		12,
		4,
		20,
		2,
		13,
		18,
		1,
		5,
		4,
		9
	};

	auto f = [&](bool* a, bool* b)
	{
		int prize1 = 0, prize2 = 0, w1 = 0, w2 = 0;
		for (int i = 0; i < 10; i++)
		{
			if (a[i])
			{
				prize1 += prise[i];
				w1 += w[i];
			}
			if (b[i])
			{
				prize2 += prise[i];
				w2 += w[i];
			}
		}

		if (w1 > 45)
			return w1 < w2;


		return prize1 > prize2;
	};

	std::vector<bool*> genetic_population;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; true; i++)
	{
		NEN::genetic<bool>(f, genetic_population, best, 10, [](bool prev, bool initial) -> bool {
			return rand() % 2;
		});
		int prize1 = 0, w1 = 0;
		for (int i = 0; i < 10; i++)
		{
			if (best[i])
			{
				std::cout << names[i] << " ";
				prize1 += prise[i];
				w1 += w[i];
			}
		}
		std::cout << "\npr = " << prize1 << " w = " << w1 << "\n";
		if (w1 == 43) {
			std::cout << "ended at i = " << i << std::endl;
			break;
		}
		//Sleep(50);
	}
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	std::cout << "time: " << diff / (1000 * 1000) << " ms" << std::endl;


	//srand(time(NULL));
	NEN::NeuronNetwork n(2, 1, 4, 20, NEN::Adam);
	n.genetic_population_size = 90;
	n.genetic_elite_part = 10;
	n.rate = 0.17;
	//n.genetic_elite_part = 5;
	//n.rate = 0.1;
	//n.rate = 0.02;
	//n.d_epsilon = 0.000000001;
	//n.activation = NEN::Sigmoid;

	//n.setMultiThreads(false);
	//n.enable_shuffle = false;

	//n.momentum = 0.7;
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
	
	auto a = std::vector<std::vector<double>>{
		NEN::normalizeInput({ log(1), log(3) }, 0, 10),
		NEN::normalizeInput({ log(2), log(7) }, 0, 10),
		NEN::normalizeInput({ log(6), log(5) }, 0, 10),
		NEN::normalizeInput({ log(5), log(5) }, 0, 10),
		NEN::normalizeInput({ log(2), log(3) }, 0, 10),
		NEN::normalizeInput({ log(1), log(8) }, 0, 10),
		NEN::normalizeInput({ log(7), log(7) }, 0, 10),
		NEN::normalizeInput({ log(3), log(6) }, 0, 10),
		NEN::normalizeInput({ log(6), log(6) }, 0, 10),
		NEN::normalizeInput({ log(8), log(4) }, 0, 10),
		NEN::normalizeInput({ log(10), log(5) }, 0, 10),
		NEN::normalizeInput({ log(6), log(7) }, 0, 10),
		NEN::normalizeInput({ log(8), log(8) }, 0, 10),
		NEN::normalizeInput({ log(9), log(9) }, 0, 10),
		NEN::normalizeInput({ log(5), log(8) }, 0, 10),
		NEN::normalizeInput({ log(5), log(7) }, 0, 10),
	};
	auto b = std::vector<std::vector<double>>{
		NEN::normalizeInput(std::vector<double>{ log(3) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(14) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(30) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(25) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(6) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(8) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(49) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(18) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(36) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(32) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(50) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(42) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(64) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(81) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(40) }, 0, 10),
		NEN::normalizeInput(std::vector<double>{ log(35) }, 0, 10),
	};
	
	
	auto start = std::chrono::high_resolution_clock::now();
	n.train(a, b, 0.01);
	//}, 0, 0.1);
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	std::cout << "time: " << diff / (1000 * 1000) << " ms" << std::endl;

	n.forward(NEN::normalizeInput({ log(8), log(8) }, 0, 10));
	for (auto& o : n.output())
		std::cout << "out " << exp(NEN::deNormalizeOutput(o, 0, 10)) << std::endl;
	n.forward(NEN::normalizeInput({ log(4), log(5) }, 0, 10));
	for (auto& o : n.output())
		std::cout << "out " << exp(NEN::deNormalizeOutput(o, 0, 10)) << std::endl;
	//n.saveFile("mul.ner");
	
	*/
	//n.setAutoSaveFile("add.ner");
	//n.loadTrainData("add.data");
	//n.trainWhileError(0, 0.5);
	//auto result = n.get({ log(2) / log(100), log(2) / log(100) });
	//std::cout << "out " << exp(result[0] * log(100)) << std::endl;
	
/*
	
	auto a = std::vector<std::vector<double>>{ { 0, 0 },{ 1, 0 },{ 0, 1 },{ 1, 1 }};
	auto b = std::vector<std::vector<double>>{ { 0 },{ 1 },{ 1 },{ 0 } };
	
	auto start = std::chrono::high_resolution_clock::now();
	n.train(a, b, 0.5);
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
	std::cout << "time: " << diff << " ms" << std::endl;
	std::cout << n.iterations << " it\n";
	std::cout << n.get({ 0, 1 })[0] << "\n";
	std::cout << n.get({ 1, 1 })[0] << "\n";

	*/

	

#if 0
	double error = 1;
	auto start = std::chrono::high_resolution_clock::now();
	unsigned itlimit = 10000;
	while (error > 0.005)
	//for(int k = 0; k < 500; k++)
	{
		if (itlimit > 0 && n.iterations > itlimit)
			break;

		error = 0;
		/*
		for (int j = 0; j < a.size(); j++)
		{
			//n.forward(a[j]);
			//error += n.backPropagate(b[j]);
			//error += n.train(a[j], b[j]);
			n.genetic([&](double * c, double * d) {
				n.forward(a[j], c);
				double error1 = n.getError(b[j]);

				n.forward(a[j], d);
				double error2 = n.getError(b[j]);

				return error1 < error2;
			});

			n.forward(a[j], n.genetic_population[0]);
			memcpy(n.neuron_targets, b[j].data(), sizeof(double) * n.outputs);
			error += NEN::error(n.neuron_outputs, n.neuron_targets, n.outputs, n.outputs_offset_neurons);
		}
		error /= a.size();
		*/
		
		/*
		std::cout << error * 100 << "%";

		
		std::cout << " ( ";
		for (int i = 0; i < 10; i++)
		{
			n.forward(a[0], n.genetic_population[i]);
			memcpy(n.neuron_targets, b[0].data(), sizeof(double) * n.outputs);
			std::cout << NEN::error(n.neuron_outputs, n.neuron_targets, n.outputs, n.outputs_offset_neurons) * 100 << "% ";
		}
		std::cout << " )";
		
		std::cout << "\n";
		*/
		//Sleep(1000);
	}
#endif
#if 0

	auto a = std::vector<std::vector<double>>{ { 0, 0 },{ 1, 0 },{ 0, 1 },{ 1, 1 } };
	auto b = std::vector<std::vector<double>>{ { 0 },{ 1 },{ 1 },{ 0 } };

	auto start = std::chrono::high_resolution_clock::now();
	double e = 1;
	std::vector<double> errs = {1, 1, 1, 1};
	auto fitness = [&n, &a, &b, &errs, &e](unsigned long long iteration) -> std::pair<std::function<bool(double*, double*)>, std::function<double()>> {
		unsigned i = iteration % a.size();
		std::vector<double> input = a[i];
		return std::pair<std::function<bool(double*, double*)>, std::function<double()>>(
		[input, &n, &b, i](double* c, double* d) -> bool {
			n.forward(input, c);
			double error1 = n.getError(b[i]);

			n.forward(input, d);
			double error2 = n.getError(b[i]);

			return error1 < error2;
		}, [input, &n, &b, i, &errs, &e]() -> double {
			n.forward(input);
			errs[i] = n.getError(b[i]);
			if (i == 3)
			{
				e = 0;
				for (double x : errs)
					e += x;
				e /= 4;
			}
			return e;
		});
	};
	//n.iterations_limit = 100;
	n.train(a, b, 0.2, fitness);
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
	std::cout << "it: " << n.iterations << "\n";
	std::cout << "time: " << diff << " ms" << std::endl;
	std::cout << n.get({ 0, 1 })[0] << "\n";
	std::cout << n.get({ 1, 1 })[0] << "\n";
	std::cout << "w: ";
	for (int i = 0; i < n.neuron_weigths_size; i++)
		std::cout << n.neuron_weigths[i] << " ";

	//n.forward(NEN::normalizeInput({ log(2), log(8) }, 0, 10));
	//for (auto& o : n.output())
	//	std::cout << "out " << exp(NEN::deNormalizeOutput(o, 0, 10)) << std::endl;	
#endif

    return 0;
}
