
#pragma once
#if defined(__NVCC__) || defined(__CUDACC__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>
#include <functional>
#include <algorithm>
#include <unordered_set>

#if !defined(__NVCC__) && !defined(__CUDACC__)
template<typename Type>
void cudaMallocManaged(Type** devPtr, size_t size)
{
	*devPtr = (Type*)calloc(1, size);
}

void cudaFree(void* devPtr)
{
	free(devPtr);
}
#endif

namespace NEN
{

	enum TrainingAlgorithm {
		StochasticGradient = 0,
		Adagrad,
		RMSProp,
		Adam
	};

	enum ActivationFunction {
		Sigmoid = 0,
		TanH,
		Identity,
		ReLU
	};

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
		double transferFunction(double x, const ActivationFunction& activation = Sigmoid)
	{
		switch (activation)
		{
		case Sigmoid:
			return 1.0 / (1.0 + std::exp(-x));
		case TanH:
			return (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
		case Identity:
			return x;
		case ReLU:
			return x > 0 ? x : 0;
		}
		return 0;
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
		double transferFunctionDerivative(double x, const ActivationFunction& activation = Sigmoid)
	{
		switch (activation)
		{
		case Sigmoid:
			return (1.0 - x) * x;
		case TanH:
			return 1.0 - x * x;
		case Identity:
			return 1;
		case ReLU:
			return x > 0 ? 1 : 0;
		}

		return 0;
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
		void forwardKernel(int i, double *outputs, double *weightes, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons, const ActivationFunction& activation)
	{
		unsigned neurons_size = layer == layers + 1 ? outputs_size : neurons;
		unsigned offset_neuron = inputs + 1 + (layer - 1) * (neurons + 1);
		unsigned prev_layer_size = (layer == 1 ? inputs : neurons) + 1;
		unsigned prev_layer_offset_neuron = offset_neuron - prev_layer_size;
		unsigned prev_layer_weight_offset = (layer == 1 ? 0 : (inputs + 1) * neurons);
		if (layer > 1)
			prev_layer_weight_offset += (neurons + 1) * neurons * (layer - 2);

		// prev layer
		double sum = 0;
		for (unsigned j = 0; j < prev_layer_size; ++j)
		{
			sum += outputs[prev_layer_offset_neuron + j] * weightes[prev_layer_weight_offset + j * neurons_size + i];
		}
		outputs[offset_neuron + i] = transferFunction(sum, activation);
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__global__ void forwardKernelGPU(double *outputs, double *weightes, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons, ActivationFunction activation)
	{
		int i = threadIdx.x;
		forwardKernel(i, outputs, weightes, layer, inputs, outputs_size, layers, neurons, activation);
	}
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
		void calculateOutputDelta(int i, double *outputs, double *delta, double *targets, const unsigned outputs_offset, const ActivationFunction& activation)
	{
		double delta_ = targets[i] - outputs[i + outputs_offset];
		delta[i + outputs_offset] = delta_ * transferFunctionDerivative(outputs[i + outputs_offset], activation);
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__global__ void calculateOutputDeltaGPU(double *outputs, double *delta, double *targets, const unsigned outputs_offset, ActivationFunction activation)
	{
		int i = threadIdx.x;
		calculateOutputDelta(i, outputs, delta, targets, outputs_offset, activation);
	}
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
		void calculateHiddensDelta(int i, double *outputs, double *weightes, double *delta, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons, const ActivationFunction& activation)
	{
		unsigned neurons_size = neurons + 1;
		unsigned offset_neuron = inputs + 1 + (layer - 1) * (neurons + 1);
		unsigned weight_offset = (inputs + 1) * neurons + (neurons + 1) * neurons * (layer - 1);
		unsigned next_layer_size = (layer == layers ? outputs_size : neurons);
		unsigned next_layer_offset_neuron = offset_neuron + neurons_size;

		double dow = 0.;
		for (unsigned n = 0; n < next_layer_size; ++n)
		{
			dow += weightes[weight_offset + i * next_layer_size + n] * delta[next_layer_offset_neuron + n];
		}
		delta[i + offset_neuron] = dow * transferFunctionDerivative(outputs[i + offset_neuron], activation);
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__global__ void calculateHiddensDeltaGPU(double *outputs, double *weightes, double *delta, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons, ActivationFunction activation)
	{
		int i = threadIdx.x;
		calculateHiddensDelta(i, outputs, weightes, delta, layer, inputs, outputs_size, layers, neurons, activation);
	}
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
	double fastPow(double a, double b) {
		// calculate approximation with fraction of the exponent
		int e = (int)b;
		union {
			double d;
			int x[2];
		} u = { a };
		u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
		u.x[0] = 0;

		// exponentiation by squaring with the exponent's integer part
		// double r = u.d makes everything much slower, not sure why
		double r = 1.0;
		while (e) {
			if (e & 1) {
				r *= a;
			}
			a *= a;
			e >>= 1;
		}

		return r * u.d;
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
		void updateInputWeights(
			int i,

			double *outputs,
			double *weightes,
			double *delta,
			double *delta_weight,

			const unsigned layer,
			const unsigned inputs,
			const unsigned outputs_size,
			const unsigned layers,
			const unsigned neurons,

			TrainingAlgorithm algorithm,

			double *algorithm_e,
			double *algorithm_m,
			double *algorithm_v,
			double *algorithm_t,

			double rate = 0.02,
			double momentum = 0.3,
			double beta1 = 0.9,
			double beta2 = 0.999,
			double d_epsilon = 0.000000001
		)
	{
		unsigned neurons_size = layer == layers + 1 ? outputs_size : neurons;
		unsigned offset_neuron = inputs + 1 + (layer - 1) * (neurons + 1);
		unsigned prev_layer_size = (layer == 1 ? inputs : neurons) + 1;
		unsigned prev_layer_offset_neuron = offset_neuron - prev_layer_size;
		unsigned prev_layer_weight_offset = (layer == 1 ? 0 : (inputs + 1) * neurons);
		if (layer > 1)
			prev_layer_weight_offset += (neurons + 1) * neurons * (layer - 2);

		for (unsigned j = 0; j < prev_layer_size; ++j)
		{
			unsigned prev_layer_weight_index = prev_layer_weight_offset + j * neurons_size + i;
			double oldDeltaWeight = delta_weight[prev_layer_weight_index];
			double gradient = outputs[prev_layer_offset_neuron + j] * delta[offset_neuron + i];
			double newDeltaWeight;

			switch (algorithm)
			{
			case StochasticGradient:
			{
				newDeltaWeight = rate * gradient + momentum * oldDeltaWeight;

				break;
			}
			case Adagrad:
			{
				double& e = algorithm_e[prev_layer_weight_index];

				e = e + gradient * gradient;
				newDeltaWeight = rate * gradient / sqrt(e + d_epsilon);

				break;
			}
			case RMSProp:
			{
				double& e = algorithm_e[prev_layer_weight_index];

				e = momentum * e + (1 - momentum) * gradient * gradient;
				newDeltaWeight = rate * gradient / sqrt(e + d_epsilon);

				break;
			}
			case Adam:
			{
				double& m = algorithm_m[prev_layer_weight_index];
				double& v = algorithm_v[prev_layer_weight_index];
				double& t = algorithm_t[prev_layer_weight_index];

				m = beta1 * m + (1 - beta1) * gradient;
				v = beta2 * v + (1 - beta2) * gradient * gradient;

				double mt = m / (1 - fastPow(beta1, t));
				double mv = v / (1 - fastPow(beta2, t));
				t++;

				newDeltaWeight = rate * mt / sqrt(mv + d_epsilon);

				break;
			}
			default:
				break;
			}

			delta_weight[prev_layer_weight_index] = newDeltaWeight;
			weightes[prev_layer_weight_index] += newDeltaWeight;
		}
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__global__ void updateInputWeightsGPU(
		double *outputs,
		double *weightes,
		double *delta,
		double *delta_weight,

		const unsigned layer,
		const unsigned inputs,
		const unsigned outputs_size,
		const unsigned layers,
		const unsigned neurons,

		TrainingAlgorithm algorithm,

		double *algorithm_e,
		double *algorithm_m,
		double *algorithm_v,
		double *algorithm_t,

		double rate,
		double momentum,
		double beta1,
		double beta2,
		double d_epsilon
	)
	{
		int i = threadIdx.x;
		updateInputWeights(
			i,

			outputs,
			weightes,
			delta,
			delta_weight,

			layer,
			inputs,
			outputs_size,
			layers,
			neurons,

			algorithm,

			algorithm_e,
			algorithm_m,
			algorithm_v,
			algorithm_t,

			rate,
			momentum,
			beta1,
			beta2,
			d_epsilon
		);
	}
#endif

	void forwardInput(double* neuron_outputs, double* neuron_weigths, unsigned inputs, unsigned outputs, unsigned layers, unsigned neurons, const ActivationFunction& activation, bool gpu)
	{
		// forward
		for (unsigned layer = 1; layer <= layers + 1; ++layer)
		{
			unsigned threads = (layer == layers + 1) ? outputs : neurons;
#if defined(__NVCC__) || defined(__CUDACC__)
			if (gpu)
			{
				forwardKernelGPU << <1, threads >> > (neuron_outputs, neuron_weigths, layer, inputs, outputs, layers, neurons, activation);
				cudaDeviceSynchronize();
			}
			else
#endif
			{
				for (int i = 0; i < threads; ++i)
					forwardKernel(i, neuron_outputs, neuron_weigths, layer, inputs, outputs, layers, neurons, activation);
			}
		}

	}

	double error(double* neuron_outputs, double* neuron_targets, unsigned outputs, unsigned outputs_offset_neurons)
	{
		double error = 0.0;
		for (unsigned i = 0; i < outputs; ++i)
		{
			double delta = neuron_targets[i] - neuron_outputs[outputs_offset_neurons + i];
			error += delta * delta;
		}
		error /= outputs;
		return error;
	}

	void backPropagation(
		double* neuron_outputs,
		double* neuron_weigths,
		double* neuron_delta,
		double* neuron_delta_weight,
		double* neuron_targets,

		unsigned inputs,
		unsigned outputs,
		unsigned layers,
		unsigned neurons,
		unsigned outputs_offset_neurons,

		TrainingAlgorithm algorithm,
		const ActivationFunction& activation,

		bool gpu,

		double *algorithm_e,
		double *algorithm_m,
		double *algorithm_v,
		double *algorithm_t,

		double rate,
		double momentum,
		double beta1,
		double beta2,
		double d_epsilon
	)
	{
		// calculate output delta
#if defined(__NVCC__) || defined(__CUDACC__)
		if (gpu)
		{
			calculateOutputDeltaGPU << <1, outputs >> > (neuron_outputs, neuron_delta, neuron_targets, outputs_offset_neurons, activation);
			cudaDeviceSynchronize();
		}
		else
#endif
		{
			for (int i = 0; i < outputs; ++i)
				calculateOutputDelta(i, neuron_outputs, neuron_delta, neuron_targets, outputs_offset_neurons, activation);
		}

		// calculate hidden deltas
		for (unsigned layer = layers; layer > 0; --layer)
		{
#if defined(__NVCC__) || defined(__CUDACC__)
			if (gpu)
			{
				calculateHiddensDeltaGPU << <1, neurons + 1 >> > (neuron_outputs, neuron_weigths, neuron_delta, layer, inputs, outputs, layers, neurons, activation);
				cudaDeviceSynchronize();
			}
			else
#endif
			{
				for (int i = 0; i < neurons + 1; ++i)
					calculateHiddensDelta(i, neuron_outputs, neuron_weigths, neuron_delta, layer, inputs, outputs, layers, neurons, activation);
			}
		}

		// update weights
		for (unsigned layer = layers + 1; layer > 0; --layer)
		{
			unsigned threads = (layer == layers + 1) ? outputs : neurons;
#if defined(__NVCC__) || defined(__CUDACC__)
			if (gpu)
			{
				updateInputWeightsGPU << <1, threads >> > (
					neuron_outputs,
					neuron_weigths,
					neuron_delta,
					neuron_delta_weight,

					layer,
					inputs,
					outputs,
					layers,
					neurons,

					algorithm,

					algorithm_e,
					algorithm_m,
					algorithm_v,
					algorithm_t,

					rate,
					momentum,
					beta1,
					beta2,
					d_epsilon
					);
				cudaDeviceSynchronize();
			}
			else
#endif
			{
				for (int i = 0; i < threads; ++i)
				{
					updateInputWeights(
						i,

						neuron_outputs,
						neuron_weigths,
						neuron_delta,
						neuron_delta_weight,

						layer,
						inputs,
						outputs,
						layers,
						neurons,

						algorithm,

						algorithm_e,
						algorithm_m,
						algorithm_v,
						algorithm_t,

						rate,
						momentum,
						beta1,
						beta2,
						d_epsilon
					);
				}
			}
		}
	}

	double randomWeight(void) { return rand() / double(RAND_MAX); }

	struct NeuronNetwork
	{
		unsigned inputs;
		unsigned outputs;
		unsigned layers;
		unsigned neurons;
		TrainingAlgorithm algorithm = Adam;
		ActivationFunction activation = Sigmoid;

		double* neuron_outputs;
		double* neuron_delta;
		double* neuron_weigths;
		double* neuron_delta_weight;
		double* neuron_targets;

		double* algorithm_e;
		double* algorithm_m;
		double* algorithm_v;
		double* algorithm_t;

		unsigned neurons_size;
		unsigned hidden_offset_neurons;
		unsigned outputs_offset_neurons;
		unsigned neuron_weigths_size;

		double rate = 0.02;
		double momentum = 0.3;
		double beta1 = 0.9;
		double beta2 = 0.999;
		double d_epsilon = 0.000000001;

		bool gpu = false;

		std::vector<std::vector<double>> train_data_inputs;
		std::vector<std::vector<double>> train_data_outputs;
		std::string auto_save_file;
		unsigned long long iterations = 0;
		unsigned long long iterations_limit = 0;
		std::function<void(unsigned long long, double)> iteration_callback;

		std::vector<double*> genetic_population;
		std::unordered_set<double*> genetic_population_allowed;
		int genetic_population_size = 10;
		int genetic_elite_part = 3;
		int genetic_max_weight = 40;
		bool genetic_populate = true;

		NeuronNetwork(unsigned inputs_, unsigned outputs_, unsigned layers_, unsigned neurons_, TrainingAlgorithm algorithm_ = Adam)
		{
			algorithm = algorithm_;

			inputs = inputs_;
			outputs = outputs_;
			layers = layers_;
			neurons = neurons_;

			init();
		}

		void init()
		{
			neurons_size = inputs + 1 + outputs + (neurons + 1) * layers;
			hidden_offset_neurons = inputs + 1;
			outputs_offset_neurons = hidden_offset_neurons + (neurons + 1) * layers;
			neuron_weigths_size = ((neurons + 1) * neurons) * (layers - 1) + ((inputs + 1) * neurons) + (outputs * (neurons + 1));

			cudaMallocManaged(&neuron_outputs, neurons_size * sizeof(double));
			cudaMallocManaged(&neuron_delta, neurons_size * sizeof(double));
			cudaMallocManaged(&neuron_weigths, neuron_weigths_size * sizeof(double));
			cudaMallocManaged(&neuron_delta_weight, neuron_weigths_size * sizeof(double));
			cudaMallocManaged(&neuron_targets, outputs * sizeof(double));
			cudaMallocManaged(&algorithm_e, neuron_weigths_size * sizeof(double));
			cudaMallocManaged(&algorithm_m, neuron_weigths_size * sizeof(double));
			cudaMallocManaged(&algorithm_v, neuron_weigths_size * sizeof(double));
			cudaMallocManaged(&algorithm_t, neuron_weigths_size * sizeof(double));

			// bias neurons
			for (unsigned layer = 0, i = 0; layer < layers + 1; ++layer)
			{
				unsigned layer_size = layer == 0 ? inputs + 1 : neurons + 1;
				i += layer_size;
				neuron_outputs[i - 1] = 1;
			}

			// first t = 1
			for (unsigned i = 0; i < neuron_weigths_size; ++i)
			{
				algorithm_t[i] = 1;
			}

			// random weightes
			for (unsigned i = 0; i < neuron_weigths_size; ++i)
			{
				neuron_weigths[i] = randomWeight();
			}

			// clear train data from prev inits
			clearTrainData();

			iterations = 0;
		}

		void free()
		{
			cudaFree(neuron_outputs);
			cudaFree(neuron_delta);
			cudaFree(neuron_weigths);
			cudaFree(neuron_delta_weight);
			cudaFree(neuron_targets);

			cudaFree(algorithm_e);
			cudaFree(algorithm_m);
			cudaFree(algorithm_v);
			cudaFree(algorithm_t);
		}

		~NeuronNetwork()
		{
			free();
			// cudaDeviceReset must be called before exiting in order for profiling and
			// tracing tools such as Nsight and Visual Profiler to show complete traces.
#if defined(__NVCC__) || defined(__CUDACC__)
			cudaDeviceReset();
#endif
		}

		void forward(const std::vector<double> &i)
		{
			memcpy(neuron_outputs, i.data(), sizeof(double) * inputs);
			forwardInput(neuron_outputs, neuron_weigths, inputs, outputs, layers, neurons, activation, gpu);
		}

		void forward(const std::vector<double> &i, double* weigths)
		{
#ifndef _DEBUG
			if (genetic_population_allowed.find(weigths) == genetic_population_allowed.end())
				return;
#endif

			memcpy(neuron_outputs, i.data(), sizeof(double) * inputs);
			forwardInput(neuron_outputs, weigths, inputs, outputs, layers, neurons, activation, gpu);
		}

		std::vector<double> output() const
		{
			std::vector<double> out;
			for (unsigned n = 0; n < outputs; ++n)
			{
				out.push_back(neuron_outputs[outputs_offset_neurons + n]);
			}
			return out;
		}

		std::vector<double> get(const std::vector<double> &i)
		{
			forward(i);
			return output();
		}

		double getError(const double* o)
		{
			memcpy(neuron_targets, o, sizeof(double) * outputs);
			return error(neuron_outputs, neuron_targets, outputs, outputs_offset_neurons);
		}

		double getError(const std::vector<double>& o)
		{
			return getError(o.data());
		}

		double backPropagate(const std::vector<double>& o)
		{
			return backPropagate(o.data());
		}

		double backPropagate(const double* o)
		{
			memcpy(neuron_targets, o, sizeof(double) * outputs);
			double err = error(neuron_outputs, neuron_targets, outputs, outputs_offset_neurons);
			backPropagation(
				neuron_outputs,
				neuron_weigths,
				neuron_delta,
				neuron_delta_weight,
				neuron_targets,

				inputs,
				outputs,
				layers,
				neurons,
				outputs_offset_neurons,

				algorithm,
				activation,

				gpu,

				algorithm_e,
				algorithm_m,
				algorithm_v,
				algorithm_t,

				rate,
				momentum,
				beta1,
				beta2,
				d_epsilon
			);
			return err;
		}

		double train(
			const double* i, 
			const double* o, 
			const std::function<bool(double*, double*)>& fitness = std::function<bool(double*, double*)>(), 
			const std::function<double()>& error_check = std::function<double()>()
		)
		{
			if (fitness)
			{
				genetic(fitness);
				double err = error_check();
				if (iteration_callback)
					iteration_callback(iterations, err);
				return err;
			}
			else
			{
				iterations++;
				memcpy(neuron_outputs, i, sizeof(double) * inputs);
				forwardInput(neuron_outputs, neuron_weigths, inputs, outputs, layers, neurons, activation, gpu);
				return backPropagate(o);
			}
			return 1;
		}

		double train(
			const std::vector<double>& i, 
			const std::vector<double>& o, 
			const std::function<bool(double*, double*)>& fitness = std::function<bool(double*, double*)>(), 
			const std::function<double()>& error_check = std::function<double()>()
		)
		{
			return train(i.data(), o.data(), fitness, error_check);
		}

		std::vector<double> train(
			const std::vector<std::vector<double>> &i, 
			const std::vector<std::vector<double>> &o,
			const std::function<
				std::pair<std::function<bool(double*, double*)>, 
				std::function<double()>
			>(unsigned long long)>& fitness = std::function<std::pair<std::function<bool(double*, double*)>, std::function<double()>>(unsigned long long)>()
		)
		{
			std::vector<double> errors;
			if (fitness)
			{
				auto ft = fitness(iterations);
				std::vector<double> nil;
				errors.push_back(train(nil, nil, ft.first, ft.second));
			}
			else
			{
				for (unsigned n = 0; n < i.size(); ++n)
				{
					errors.push_back(train(i[n], o[n]));
				}
			}
			
			// shuffle
			/*
			for (auto s = i.size() - 1; s > 0; --s)
			{
				auto r = rand() % (s + 1);
				auto i_back = i[r];
				auto o_back = o[r];
				i[r] = i[s];
				o[r] = o[s];
				i[s] = i_back;
				o[s] = o_back;
			}
			*/
			// print
			static auto start = std::chrono::high_resolution_clock::now();
			auto finish = std::chrono::high_resolution_clock::now();
			auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
			if (diff > 1000 * 1000 * 90)
			{
				start = finish;
				printStatistic(errors);
			}
			static auto start2 = std::chrono::high_resolution_clock::now();
			auto diff2 = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start2).count();
			if (diff2 > 1000 * 1000 * 600)
			{
				start2 = finish;

				if (!auto_save_file.empty())
				{
					saveFile(auto_save_file);
					std::cout << "sv" << std::endl;
				}
			}


			return errors;
		}

		std::vector<double> train(
			const std::vector<std::vector<double>> &i, 
			const std::vector<std::vector<double>> &o, 
			double error_ = 0,
			const std::function<
				std::pair<std::function<bool(double*, double*)>, 
				std::function<double()>
			>(unsigned long long)>& fitness = std::function<std::pair<std::function<bool(double*, double*)>, std::function<double()>>(unsigned long long)>()
		)
		{
			unsigned long long real_iterations = 0;
			std::vector<double> errors;
			do {
				errors = train(i, o, fitness);
				real_iterations++;
			} while (([&]() {
				if (error_ == 0)
					return false;

				if (iterations_limit > 0 && real_iterations >= iterations_limit)
					return false;

				double errorAvrg = 0;
				for (auto error : errors)
				{
					errorAvrg += error;
				}
				errorAvrg /= errors.size();

				if (iteration_callback && !fitness)
					iteration_callback(iterations, errorAvrg);

				if (errorAvrg * 100 > error_)
					return true;

				return false;
			})());
			printStatistic(errors);
			return errors;
		}

		void train(double error_)
		{
			if (train_data_inputs.size() == 0 || train_data_outputs.size() == 0)
				return;
			train(train_data_inputs, train_data_outputs, error_);
		}

		void printStatistic(const std::vector<double>& errors)
		{
#ifdef _DEBUG
			system("cls");

			std::cout << "neurons = " << neurons_size << " w = " << neuron_weigths_size << " run on = " << (gpu ? "GPU" : "CPU") << " algorithm = " << algorithm << "\n";
			std::cout << "r = " << rate << " m = " << momentum << " b1 = " << beta1 << " b2 = " << beta2 << " eps = " << d_epsilon << "\n";
			std::cout << "iterations: " << iterations << "\n";
			double avrg = 0;
			for (double error : errors)
			{
				std::cout << error * 100 << "%" << std::endl;
				avrg += error;
			}
			std::cout << "avrg error = " << (avrg / errors.size()) * 100 << "%" << std::endl;
#endif
		}

		void saveFile(const std::string& file)
		{
			std::ofstream f;
			f.open(file);
			f << inputs << "\n";
			f << outputs << "\n";
			f << layers << "\n";
			f << neurons << "\n";
			f << activation << "\n";
			for (unsigned i = 0; i < neuron_weigths_size; ++i)
				f << neuron_weigths[i] << "\n";
			f.close();
		}

		void loadFile(const std::string& file)
		{
			std::ifstream f;
			f.open(file);
			if (!f.is_open())
				return;
			int activation_value;
			f >> inputs >> outputs >> layers >> neurons >> activation_value;
			activation = (ActivationFunction)activation_value;
			free();
			init();
			double weight;
			for (unsigned i = 0; i < neuron_weigths_size; ++i)
			{
				f >> weight;
				neuron_weigths[i] = weight;
			}
			f.close();
		}

		void loadTrainData(const std::string& file)
		{
			std::string line;
			std::ifstream f(file);
			if (!f.is_open())
				return;
			while (getline(f, line))
			{
				std::stringstream s(line);
				std::string arg;
				std::vector<double> i;
				std::vector<double> o;
				while (getline(s, arg, ' ')) {
					if (i.size() < inputs)
						i.push_back(stod(arg));
					else
						o.push_back(stod(arg));
				}
				if (i.size() == inputs && o.size() == outputs)
				{
					train_data_inputs.push_back(i);
					train_data_outputs.push_back(o);
				}
				else
					assert(false);
			}
			f.close();
		}

		void clearTrainData()
		{
			train_data_inputs.clear();
			train_data_outputs.clear();
		}

		void setAutoSaveFile(const std::string& file)
		{
			auto_save_file = file;
			loadFile(file);
		}

		// genetic algorithm
		void genetic(const std::function<bool(double*, double*)>& genetic_fitness)
		{
			// copy population
			if (genetic_population.size() != genetic_population_size)
			{
				for (double* entity : genetic_population)
					cudaFree(entity);
				genetic_population_allowed.clear();

				double* entity;
				cudaMallocManaged(&entity, neuron_weigths_size * sizeof(double));
				genetic_population_allowed.insert(entity);
				memcpy(entity, neuron_weigths, neuron_weigths_size * sizeof(double));
				genetic_population.push_back(entity);

				for (int i = 1; i < genetic_population_size; ++i)
				{
					double* entity;
					cudaMallocManaged(&entity, neuron_weigths_size * sizeof(double));
					genetic_population_allowed.insert(entity);
					for (unsigned j = 0; j < neuron_weigths_size; j++)
						entity[j] = ((double)rand() / (RAND_MAX));
					genetic_population.push_back(entity);
				}
			}
			// sort population
			std::sort(genetic_population.begin(), genetic_population.end(), genetic_fitness);
			unsigned elite = genetic_population_size / genetic_elite_part;

			// out best result
			memcpy(neuron_weigths, genetic_population[0], neuron_weigths_size * sizeof(double));

			// populate
			if (genetic_populate)
			{
				for (int i = 1; i < genetic_population_size / elite; i++)
					for (int j = 0; j < elite; j++)
						memcpy(genetic_population[i * elite + j], genetic_population[j], neuron_weigths_size * sizeof(double));
			}

			for (int i = elite; i < genetic_population_size; ++i)
			{
				double* bad_entity = genetic_population[i];

				// crossing over
				double* random_elite = genetic_population[rand() % elite];
				for (unsigned gen = 0; gen < neuron_weigths_size; ++gen)
					if (rand() % 2 == 1)
						bad_entity[gen] = random_elite[gen];

				// mutation
				//for (unsigned gen = 0; gen < neuron_weigths_size; ++gen)
				//	if (rand() % 4 == 1)
				//		bad_entity[gen] = ((double)rand() / (RAND_MAX));
				for (unsigned gen = 0; gen < neuron_weigths_size; ++gen)
				{
					if (rand() % 4 == 1)
					{
						if (bad_entity[gen] > genetic_max_weight || bad_entity[gen] < -genetic_max_weight)
							bad_entity[gen] = (((double)rand() / (RAND_MAX)) * 2 - 1);
						bad_entity[gen] += (((double)rand() / (RAND_MAX)) * 2 - 1) * rate;
					}
				}
			}
			iterations++;
		}
	};


	double normalizeInput(double x, double max, double min)
	{
		return (x - min) / (max - min);
	}

	std::vector<double> normalizeInput(const std::vector<double> &xArray, double max, double min)
	{
		std::vector<double> xSes;
		for (double x : xArray)
			xSes.push_back(normalizeInput(x, max, min));
		return xSes;
	}

	double deNormalizeOutput(double y, double max, double min)
	{
		return min + y * (max - min);
	}

	std::vector<double> deNormalizeOutput(const std::vector<double> &yArray, double max, double min)
	{
		std::vector<double> ySes;
		for (double y : yArray)
			ySes.push_back(deNormalizeOutput(y, max, min));
		return ySes;
	}

}
