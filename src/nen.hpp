
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
#include <omp.h>

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
	void forwardKernel(
		double *outputs, 
		double *weightes, 
		
		const unsigned inputs, 
		const unsigned outputs_size,
		const unsigned layers, 
		const unsigned neurons, 
		
		const unsigned neurons_size,

		const ActivationFunction& activation,
		const unsigned threadId = 0
	)
	{
		for (unsigned layer = 1; layer <= layers + 1; ++layer)
		{
			unsigned current_layer_neurons_size = layer == layers + 1 ? outputs_size : neurons;
			unsigned current_layer_offset_neuron = inputs + 1 + (layer - 1) * (neurons + 1);
			unsigned prev_layer_size = (layer == 1 ? inputs : neurons) + 1;
			unsigned prev_layer_offset_neuron = current_layer_offset_neuron - prev_layer_size;
			unsigned prev_layer_weight_offset = (layer == 1 ? 0 : (inputs + 1) * neurons);
			if (layer > 1)
				prev_layer_weight_offset += (neurons + 1) * neurons * (layer - 2);

			for (int i = 0; i < current_layer_neurons_size; ++i)
			{
				double sum = 0;
				for (unsigned j = 0; j < prev_layer_size; ++j)
				{
					sum += outputs[threadId * neurons_size + prev_layer_offset_neuron + j] * weightes[prev_layer_weight_offset + j * current_layer_neurons_size + i];
				}
				outputs[threadId * neurons_size + current_layer_offset_neuron + i] = transferFunction(sum, activation);
			}
		}
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
	void calculateOutputDelta(
		double *outputs, 
		double *delta, 
		double *targets,

		const unsigned outputs_offset, 
		const unsigned outputs_size,
		const unsigned neurons_size,
		
		const ActivationFunction& activation,
		const unsigned threadId = 0
	)
	{
		for (int i = 0; i < outputs_size; ++i)
		{
			double delta_ = targets[threadId * outputs_size + i] - outputs[threadId * neurons_size + i + outputs_offset];
			delta[threadId * neurons_size + i + outputs_offset] = delta_ * transferFunctionDerivative(outputs[threadId * neurons_size + i + outputs_offset], activation);
		}
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
	void calculateHiddensDelta(
		double *outputs,
		double *weightes, 
		double *delta, 
		
		const unsigned inputs, 
		const unsigned outputs_size, 
		const unsigned layers,
		const unsigned neurons,

		const unsigned neurons_size,
		
		const ActivationFunction& activation,
		const unsigned threadId = 0
	)
	{
		for (unsigned layer = layers; layer > 0; --layer)
		{
			unsigned current_layer_neurons_size = neurons + 1;
			unsigned current_layer_offset_neuron = inputs + 1 + (layer - 1) * (neurons + 1);
			unsigned current_layer_weight_offset = (inputs + 1) * neurons + (neurons + 1) * neurons * (layer - 1);
			unsigned next_layer_size = (layer == layers ? outputs_size : neurons);
			unsigned next_layer_offset_neuron = current_layer_offset_neuron + current_layer_neurons_size;

			for (int i = 0; i < neurons + 1; ++i)
			{
				double dow = 0.;
				for (unsigned n = 0; n < next_layer_size; ++n)
				{
					dow += weightes[current_layer_weight_offset + i * next_layer_size + n] * delta[threadId * neurons_size + next_layer_offset_neuron + n];
				}
				delta[threadId * neurons_size + i + current_layer_offset_neuron] = dow * transferFunctionDerivative(outputs[threadId * neurons_size + i + current_layer_offset_neuron], activation);
			}
		}
	}

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

    static unsigned int g_seed[8];
    int fastRand(const int& tid = 0) {
        g_seed[tid] = (214013*g_seed[tid]+2531011);
        return (g_seed[tid]>>16)&0x7FFF;
    }

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
		void updateInputWeights(
			double *outputs,
			double *weightes,
			double *delta,
			double *delta_weight,

			const unsigned inputs,
			const unsigned outputs_size,
			const unsigned layers,
			const unsigned neurons,
			const unsigned neurons_size,
			const unsigned neuron_weigths_size,

			TrainingAlgorithm algorithm,

			double *algorithm_e,
			double *algorithm_m,
			double *algorithm_v,
			double *algorithm_t,

			double rate = 0.02,
			double momentum = 0.3,
			double beta1 = 0.9,
			double beta2 = 0.999,
			double d_epsilon = 0.000000001,

			const unsigned threadId = 0
		)
	{
		for (unsigned layer = layers + 1; layer > 0; --layer)
		{
			unsigned current_layer_neurons_size = layer == layers + 1 ? outputs_size : neurons;
			unsigned current_layer_offset_neuron = inputs + 1 + (layer - 1) * (neurons + 1) + threadId * neurons_size;
			unsigned prev_layer_size = (layer == 1 ? inputs : neurons) + 1;
			unsigned prev_layer_offset_neuron = current_layer_offset_neuron - prev_layer_size;
			unsigned prev_layer_weight_offset = (layer == 1 ? 0 : (inputs + 1) * neurons) + threadId * neuron_weigths_size;
			if (layer > 1)
				prev_layer_weight_offset += (neurons + 1) * neurons * (layer - 2);

			for (int i = 0; i < current_layer_neurons_size; ++i)
			{
				for (unsigned j = 0; j < prev_layer_size; ++j)
				{
					unsigned prev_layer_weight_index = prev_layer_weight_offset + j * current_layer_neurons_size + i;
					double oldDeltaWeight = delta_weight[prev_layer_weight_index];
					double gradient = outputs[prev_layer_offset_neuron + j] * delta[current_layer_offset_neuron + i];
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

						newDeltaWeight = rate * (m / (1 - fastPow(beta1, t))) / sqrt((v / (1 - fastPow(beta2, t))) + d_epsilon);
						t++;

						break;
					}
					default:
						break;
					}

					delta_weight[prev_layer_weight_index] = newDeltaWeight;
					//weightes[prev_layer_weight_index] += newDeltaWeight;
				}
			}
		}
	}

#if defined(__NVCC__) || defined(__CUDACC__)
	__host__ __device__
#endif
	double error(
		double* neuron_outputs,
		double* neuron_targets, 
		
		unsigned outputs, 
		unsigned outputs_offset_neurons, 
		const unsigned neurons_size,
		
		unsigned threadId = 0
	)
	{
		double error = 0.0;
		for (unsigned i = 0; i < outputs; ++i)
		{
			double delta = neuron_targets[threadId * outputs + i] - neuron_outputs[threadId * neurons_size + outputs_offset_neurons + i];
			error += delta * delta;
		}
		error /= outputs;
		return error;
	}

	double randomWeight(void) { return rand() / double(RAND_MAX); }
	
	static int threads_max = 1;

	template<typename T>
    void genetic(
        int tid,
		const std::function<bool(T*, T*)>& genetic_fitness,
		std::vector<T*>& genetic_population,
		T* neuron_weigths,
		const size_t neuron_weigths_size,
		const std::function<T(T prev, bool initial)>& random,
		unsigned genetic_population_size = 10,
		unsigned genetic_elite_part = 3,
		bool genetic_populate = true,
		std::unordered_set<T*>* genetic_population_allowed = nullptr
	)
	{
		// sort population
        std::sort(genetic_population.begin() + tid * genetic_population_size, genetic_population.begin() + (tid + 1) * genetic_population_size, genetic_fitness);
		unsigned elite = genetic_population_size / genetic_elite_part;


		// out best result
        if(tid == 0) {
            //memcpy(neuron_weigths, genetic_population[0], neuron_weigths_size * sizeof(T));
            for (size_t i = 0; i < neuron_weigths_size; ++i )
                neuron_weigths[i] = genetic_population[0][i];
        }

		// populate
		if (genetic_populate)
		{
            for (int i = 1; i < genetic_population_size / elite; i++)
			{
                for (int j = 0; j < elite; j++) {
                    //memcpy(genetic_population[tid * genetic_population_size + i * elite + j], genetic_population[tid * genetic_population_size + j], neuron_weigths_size * sizeof(T));
                    for (size_t k = 0; k < neuron_weigths_size; ++k )
                        genetic_population[tid * genetic_population_size + i * elite + j][k] = genetic_population[tid * genetic_population_size + j][k];
                }
			}
		}

		for (int i = elite; i < genetic_population_size; ++i)
		{
            T* bad_entity = genetic_population[tid * genetic_population_size + i];

			if(i >= genetic_population_size - 1)
            {
                for (unsigned gen = 0; gen < neuron_weigths_size; ++gen)
                    bad_entity[gen] = random(bad_entity[gen], false);

                continue;
            }

			// crossing over
            T* random_elite = genetic_population[tid * genetic_population_size + rand() % elite];
			for (unsigned gen = 0; gen < neuron_weigths_size; ++gen)
            {
                if (fastRand(tid) % 2 == 1) {
                    bad_entity[gen] = random_elite[gen];
                }
            }

            // mutation
            for (unsigned gen = 0; gen < neuron_weigths_size; ++gen)
			{
                if (fastRand(tid) % 8 == 1)
				{
                    bad_entity[gen] = random(bad_entity[gen], false);
				}
            }
        }
        //iterations++;
    }

    template<typename T>
    void genetic_async(
        int& iteration,
        const std::function<bool(T*, T*)>& genetic_fitness,
        std::vector<T*>& genetic_population,
        T* neuron_weigths,
        const size_t neuron_weigths_size,
        const std::function<T(T prev, bool initial)>& random,
        const std::function<bool()>& condition,
        unsigned genetic_population_size = 10,
        unsigned genetic_elite_part = 3,
        bool genetic_populate = true,
        std::unordered_set<T*>* genetic_population_allowed = nullptr
    )
    {
        // copy population
        int threads = omp_get_max_threads();
        if (genetic_population.size() != genetic_population_size * threads)
        {
            std::cout << "gen";
            for (T* entity : genetic_population)
                cudaFree(entity);
            if (genetic_population_allowed)
                genetic_population_allowed->clear();

            T* entity;
            cudaMallocManaged(&entity, neuron_weigths_size * sizeof(T));
            if (genetic_population_allowed)
                genetic_population_allowed->insert(entity);
            memcpy(entity, neuron_weigths, neuron_weigths_size * sizeof(T));
            genetic_population.push_back(entity);

            for (int i = 1; i < genetic_population_size * threads; ++i)
            {
                T* entity;
                cudaMallocManaged(&entity, neuron_weigths_size * sizeof(T));
                if (genetic_population_allowed)
                    genetic_population_allowed->insert(entity);
                for (unsigned j = 0; j < neuron_weigths_size; j++)
                    entity[j] = random(0, true);
                genetic_population.push_back(entity);
            }
        }

		// copy elite to all threaded generations, for try it suvive ability
		unsigned elite = genetic_population_size / genetic_elite_part;

        bool check = true;
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
			while(check)
            {
                genetic<T>(tid, genetic_fitness, genetic_population, neuron_weigths, neuron_weigths_size, random, genetic_population_size, genetic_elite_part, genetic_populate, genetic_population_allowed);

#pragma omp barrier
				{
					for (int i = 0; i < threads; i++) {
						for (int j = 0; j < elite; j++)
						{
							for (int k = 0; k < threads; k++) {
								if (k == i)
									continue;
								if (genetic_population_size - (i * elite) - j - 1 < elite) {
									continue;
								}
								memcpy(genetic_population[k * genetic_population_size + genetic_population_size - (i * elite) - j - 1], genetic_population[i * genetic_population_size + j], neuron_weigths_size * sizeof(T));
							}
						}
					}
				}

                #pragma omp single
                {
                    iteration++;
                    check = condition();
                }

                //if(iteration > 60)
                //    break;
            }
        }
    }

	struct NeuronNetwork
	{
		int threads = 1;
		bool enable_shuffle = true;

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
		double d_epsilon = 0.00000000001;

		bool gpu = false;

		std::vector<std::vector<double>> train_data_inputs;
		std::vector<std::vector<double>> train_data_outputs;
		std::string auto_save_file;
		unsigned long long iterations = 0;
		unsigned long long epoch_iterations = 0;
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
			int new_threads_count = omp_get_max_threads();
			if(new_threads_count > threads_max)
				threads_max = new_threads_count;
			
			algorithm = algorithm_;
			inputs = inputs_;
			outputs = outputs_;
			layers = layers_;
			neurons = neurons_;
			init();
		}

		void init()
		{
			threads = threads_max;
			if (threads <= 0)
				threads = 1;

			neurons_size = inputs + 1 + outputs + (neurons + 1) * layers;
			hidden_offset_neurons = inputs + 1;
			outputs_offset_neurons = hidden_offset_neurons + (neurons + 1) * layers;
			neuron_weigths_size = ((neurons + 1) * neurons) * (layers - 1) + ((inputs + 1) * neurons) + (outputs * (neurons + 1));

			cudaMallocManaged(&neuron_outputs, neurons_size * sizeof(double) * threads);
			cudaMallocManaged(&neuron_delta, neurons_size * sizeof(double) * threads);
			cudaMallocManaged(&neuron_weigths, neuron_weigths_size * sizeof(double));
			cudaMallocManaged(&neuron_delta_weight, neuron_weigths_size * sizeof(double) * threads);
			cudaMallocManaged(&neuron_targets, outputs * sizeof(double) * threads);
			cudaMallocManaged(&algorithm_e, neuron_weigths_size * sizeof(double) * threads);
			cudaMallocManaged(&algorithm_m, neuron_weigths_size * sizeof(double) * threads);
			cudaMallocManaged(&algorithm_v, neuron_weigths_size * sizeof(double) * threads);
			cudaMallocManaged(&algorithm_t, neuron_weigths_size * sizeof(double) * threads);

			// bias neurons
			for (int j = 0; j < threads; j++)
			{
				for (unsigned layer = 0, i = 0; layer < layers + 1; ++layer)
				{
					unsigned layer_size = layer == 0 ? inputs + 1 : neurons + 1;
					i += layer_size;
					neuron_outputs[j * neurons_size + i - 1] = 1;
				}
			}

			// first t = 1
			for (int j = 0; j < threads; j++)
			{
				for (unsigned i = 0; i < neuron_weigths_size; ++i)
				{
					algorithm_t[j * neuron_weigths_size + i] = 1;
				}
			}

			// random weightes
			for (unsigned i = 0; i < neuron_weigths_size; ++i)
			{
				neuron_weigths[i] = randomWeight();
			}

			// clear train data from prev inits
			clearTrainData();

			iterations = 0;
			epoch_iterations = 0;
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

		void setMultiThreads(bool multi)
		{
			if (multi)
			{
				threads = threads_max;
				omp_set_num_threads(threads_max);
			}
			else
			{
				threads = 1;
				omp_set_num_threads(1);
			}
		}

		void forward(const std::vector<double> &i)
		{
			memcpy(neuron_outputs, i.data(), sizeof(double) * inputs);
			forwardKernel(
				neuron_outputs, 
				neuron_weigths, 
				
				inputs, 
				outputs, 
				layers, 
				neurons, 
				neurons_size, 
				
				activation
			);
		}

		void forward(const std::vector<double> &i, double* weigths)
		{
#ifndef _DEBUG
			if (genetic_population_allowed.find(weigths) == genetic_population_allowed.end())
				return;
#endif

			memcpy(neuron_outputs, i.data(), sizeof(double) * inputs);
			forwardKernel(
				neuron_outputs,
				weigths,

				inputs,
				outputs,
				layers,
				neurons,
				neurons_size,

				activation
			);
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
			return error(neuron_outputs, neuron_targets, outputs, outputs_offset_neurons, neurons_size);
		}

		double getError(const std::vector<double>& o)
		{
			return getError(o.data());
		}

		double backPropagate(const std::vector<double>& o, unsigned threadId = 0)
		{
			return backPropagate(o.data(), threadId);
		}

		double backPropagate(const double* o, unsigned threadId = 0)
		{
			memcpy(neuron_targets + threadId * outputs, o, sizeof(double) * outputs);
			double err = error(neuron_outputs, neuron_targets, outputs, outputs_offset_neurons, neurons_size, threadId);
			// calculate output delta
			calculateOutputDelta(
				neuron_outputs,
				neuron_delta,
				neuron_targets,

				outputs_offset_neurons,
				outputs,
				neurons_size,

				activation,
				threadId
			);

			// calculate hidden deltas
			calculateHiddensDelta(
				neuron_outputs,
				neuron_weigths,
				neuron_delta,

				inputs,
				outputs,
				layers,
				neurons,

				neurons_size,

				activation,
				threadId
			);

			// update weights
			updateInputWeights(
				neuron_outputs,
				neuron_weigths,
				neuron_delta,
				neuron_delta_weight,

				inputs,
				outputs,
				layers,
				neurons,
				neurons_size,
				neuron_weigths_size,

				algorithm,

				algorithm_e,
				algorithm_m,
				algorithm_v,
				algorithm_t,

				rate,
				momentum,
				beta1,
				beta2,
				d_epsilon,

				threadId
			);
			return err;
		}

		double train(
			const double* i, 
			const double* o, 
			const std::function<bool(double*, double*)>& fitness = std::function<bool(double*, double*)>(), 
			const std::function<double()>& error_check = std::function<double()>(),
			unsigned threadId = 0
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
#pragma omp atomic
				iterations++;
				memcpy(neuron_outputs + threadId * neurons_size, i, sizeof(double) * inputs);
				forwardKernel(
					neuron_outputs,
					neuron_weigths,

					inputs,
					outputs,
					layers,
					neurons,
					neurons_size,

					activation,
					threadId
				);
				return backPropagate(o, threadId);
			}
			return 1;
		}

		double train(
			const std::vector<double>& i, 
			const std::vector<double>& o, 
			const std::function<bool(double*, double*)>& fitness = std::function<bool(double*, double*)>(), 
			const std::function<double()>& error_check = std::function<double()>(),
			unsigned threadId = 0
		)
		{
			return train(i.data(), o.data(), fitness, error_check, threadId);
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
				size_t inputs_size = i.size();
				errors.resize(inputs_size);
				for (int j = 0; j < (int)ceil((double)inputs_size / threads); ++j)
				{
#pragma omp parallel for
					for (int t = 0; t < threads; ++t)
					{
						int tid = omp_get_thread_num();
						unsigned n = j * threads + t;
						if(n >= inputs_size)
							continue;

						errors[n] = train(i[n], o[n], std::function<bool(double*, double*)>(), std::function<double()>(), tid);
						
						//#pragma omp critical
						//{
						//	for (int j = 0; j < neuron_weigths_size; j++)
						//		neuron_weigths[j] += neuron_delta_weight[tid * neuron_weigths_size + j];
						//}
					}

					for (int i = 0; i < threads; i++)
					{
						if (j * threads + i >= inputs_size)
							break;

						for (int j = 0; j < neuron_weigths_size; j++)
						{
							neuron_weigths[j] += neuron_delta_weight[i * neuron_weigths_size + j];
						}
					}
				}
			}
			
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
			const std::vector<std::vector<double>> &inputs,
			const std::vector<std::vector<double>> &outputs,
			double error_ = 0,
			const std::function<
				std::pair<std::function<bool(double*, double*)>, 
				std::function<double()>
			>(unsigned long long)>& fitness = std::function<std::pair<std::function<bool(double*, double*)>, std::function<double()>>(unsigned long long)>()
		)
		{
			auto& i = (inputs.size() == 0 && train_data_inputs.size() > 0) ? train_data_inputs :  inputs;
			auto& o = (outputs.size() == 0 && train_data_outputs.size() > 0) ? train_data_outputs : outputs;

			std::vector<double> errors;
			
			std::vector<std::vector<double>> input_shuffle;
			std::vector<std::vector<double>> output_shuffle;
			auto& input_real = enable_shuffle ? input_shuffle : i;
			auto& output_real = enable_shuffle ? output_shuffle : o;

			if (enable_shuffle && i.size() != 0)
			{
				input_shuffle = i;
				output_shuffle = o;
			}

			do {
				if (enable_shuffle && i.size() != 0)
				{
					for (auto s = i.size() - 1; s > 0; --s)
					{
						auto r = rand() % (s + 1);
						auto i_back = input_shuffle[r];
						auto o_back = output_shuffle[r];
						input_shuffle[r] = input_shuffle[s];
						output_shuffle[r] = output_shuffle[s];
						input_shuffle[s] = i_back;
						output_shuffle[s] = o_back;
					}
				}

				errors = train(input_real, output_real, fitness);
				epoch_iterations++;
			} while (([&]() {
				double errorAvrg = 0;
				for (auto error : errors)
				{
					errorAvrg += error;
				}
				errorAvrg /= errors.size();

				if (iteration_callback && !fitness)
					iteration_callback(iterations, errorAvrg);

				if (error_ == 0 && iterations_limit == 0)
					return false;

				if (iterations_limit > 0 && epoch_iterations >= iterations_limit)
					return false;

				if (errorAvrg * 100 <= error_)
					return false;

				return true;
			})());
			printStatistic(errors);
			return errors;
		}

		void printStatistic(const std::vector<double>& errors)
		{
#ifdef _DEBUG
			system("cls");

			std::cout << "neurons = " << neurons_size << " w = " << neuron_weigths_size << " run on = " << (gpu ? "GPU" : "CPU") << " threads = " << threads << " algorithm = " << algorithm << "\n";
			std::cout << "shuffle = " << enable_shuffle << " r = " << rate << " m = " << momentum << " b1 = " << beta1 << " b2 = " << beta2 << " eps = " << d_epsilon << "\n";
			std::cout << "iterations: " << iterations << "\n";
			double avrg = 0;
			int errors_max = 0;
			for (double error : errors)
			{
				avrg += error;
				if (++errors_max >= 25)
					continue;
				std::cout << error * 100 << "%" << std::endl;
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
            NEN::genetic<double>(
                0,
				genetic_fitness,
				genetic_population,
				neuron_weigths,
				neuron_weigths_size,
				[&](double prev, bool initial) {
					if (initial) {
						return (double)rand() / (RAND_MAX);
					}

					double ret;
					if (prev > genetic_max_weight || prev < -genetic_max_weight)
						ret = (((double)rand() / (RAND_MAX)) * 2 - 1);
					ret = prev + (((double)rand() / (RAND_MAX)) * 2 - 1) * rate;
					return ret;
				},
				genetic_population_size,
				genetic_elite_part,
				genetic_populate,
				&genetic_population_allowed
			);
			iterations++;
		}
	};


	double normalizeInput(double x, double min, double max)
	{
		return (x - min) / (max - min);
	}

	std::vector<double> normalizeInput(const std::vector<double> &xArray, double min, double max)
	{
		std::vector<double> xSes;
		for (double x : xArray)
			xSes.push_back(normalizeInput(x, min, max));
		return xSes;
	}

	double deNormalizeOutput(double y, double min, double max)
	{
		return min + y * (max - min);
	}

	std::vector<double> deNormalizeOutput(const std::vector<double> &yArray, double min, double max)
	{
		std::vector<double> ySes;
		for (double y : yArray)
			ySes.push_back(deNormalizeOutput(y, min, max));
		return ySes;
	}

}
