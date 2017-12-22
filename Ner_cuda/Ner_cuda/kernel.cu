
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
//#include "../../core/network.hpp"
#include <iostream>
#include <vector>

/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
*/

enum TrainingAlgorithm {
	StochasticGradient = 0,
	Adagrad,
	RMSProp,
	Adam
};

__device__ double transferFunction(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

__device__ double transferFunctionDerivative(double x)
{
	return (1.0 - x) * x;
}

__global__ void forwardKernel(double *outputs, double *weightes, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons)
{
	int i = threadIdx.x;

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
	outputs[offset_neuron + i] = transferFunction(sum);

	printf("out = %f\n", outputs[i + offset_neuron]);
}

__global__ void calculateOutputDelta(double *outputs, double *delta, double *targets, const unsigned outputs_offset)
{
	int i = threadIdx.x;

	double delta_ = targets[i] - outputs[i + outputs_offset];
	delta[i + outputs_offset] = delta_ * transferFunctionDerivative(outputs[i + outputs_offset]);
}


__global__ void calculateHiddensDelta(double *outputs, double *weightes, double *delta, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons)
{
	int i = threadIdx.x;

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
	delta[i + offset_neuron] = dow * transferFunctionDerivative(outputs[i + offset_neuron]);
}

__constant__ const double rate = 0.7;
__constant__ const double momentum = 0.3;
__constant__ const double beta1 = 0.9;
__constant__ const double beta2 = 0.999;
__constant__ const double d_epsilon = 0.0000001;

__global__ void updateInputWeights(
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
	double *algorithm_t
)
{
	int i = threadIdx.x;

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

			e = e + pow(gradient, 2);
			newDeltaWeight = rate * gradient / sqrt(e + d_epsilon);

			break;
		}
		case RMSProp:
		{
			double& e = algorithm_e[prev_layer_weight_index];

			e = momentum * e + (1 - momentum) * pow(gradient, 2);
			newDeltaWeight = rate * gradient / sqrt(e + d_epsilon);

			break;
		}
		case Adam:
		{
			double& m = algorithm_m[prev_layer_weight_index];
			double& v = algorithm_v[prev_layer_weight_index];
			double& t = algorithm_t[prev_layer_weight_index];

			m = beta1 * m + (1 - beta1) * gradient;
			v = beta2 * v + (1 - beta2) * pow(gradient, 2);

			double mt = m / (1 - pow(beta1, t));
			double mv = v / (1 - pow(beta2, t));
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

void forward(double* neuron_outputs, double* neuron_weigths, unsigned inputs, unsigned outputs, unsigned layers, unsigned neurons)
{
	// forward
	for (unsigned layer = 1; layer <= layers + 1; ++layer)
	{
		unsigned threads = (layer == layers + 1) ? outputs : neurons;
		forwardKernel << <1, threads >> >(neuron_outputs, neuron_weigths, layer, inputs, outputs, layers, neurons);
		cudaDeviceSynchronize();
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
	std::cout << error * 100 << '%' << std::endl;
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

	double *algorithm_e,
	double *algorithm_m,
	double *algorithm_v,
	double *algorithm_t
)
{
	// calculate output delta
	calculateOutputDelta << <1, outputs >> >(neuron_outputs, neuron_delta, neuron_targets, outputs_offset_neurons);
	cudaDeviceSynchronize();

	// calculate hidden deltas
	for (unsigned layer = layers; layer > 0; --layer)
	{
		calculateHiddensDelta << <1, neurons + 1 >> >(neuron_outputs, neuron_weigths, neuron_delta, layer, inputs, outputs, layers, neurons);
		cudaDeviceSynchronize();
	}

	// update weights
	for (unsigned layer = layers + 1; layer > 0; --layer)
	{
		unsigned threads = (layer == layers + 1) ? outputs : neurons;
		updateInputWeights << <1, threads >> >(
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
			algorithm_t
		);
		cudaDeviceSynchronize();
	}
}

double randomWeight(void) { return rand() / double(RAND_MAX); }

struct NeuronNetwork
{
	int inputs;
	int outputs;
	int layers;
	int neurons;
	TrainingAlgorithm algorithm = StochasticGradient;

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

	NeuronNetwork(int inputs_, int outputs_, int layers_, int neurons_, TrainingAlgorithm algorithm_ = StochasticGradient)
	{
		algorithm = algorithm_;

		inputs = inputs_;
		outputs = outputs_;
		layers = layers_;
		neurons = neurons_;

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
	}

	~NeuronNetwork() 
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

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaDeviceReset();
	}

	void train(double* i, double* o)
	{
		memcpy(neuron_outputs, i, sizeof(double) * inputs);
		memcpy(neuron_targets, o, sizeof(double) * outputs);
		forward(neuron_outputs, neuron_weigths, inputs, outputs, layers, neurons);
		error(neuron_outputs, neuron_targets, outputs, outputs_offset_neurons);
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

			algorithm_e,
			algorithm_m,
			algorithm_v,
			algorithm_t
		);
	}

	void train(std::vector<double> i, std::vector<double> o)
	{
		train(i.data(), o.data());
	}
};

int main()
{
	NeuronNetwork n(2, 1, 1, 2, StochasticGradient);

	n.neuron_weigths[0] = 0.45;
	n.neuron_weigths[1] = 0.78;
	n.neuron_weigths[2] = -0.12;
	n.neuron_weigths[3] = 0.13;
	n.neuron_weigths[6] = 1.5;
	n.neuron_weigths[7] = -2.3;

	n.train(std::vector<double>{ 1, 0 }, std::vector<double>{ 1 });
	n.train(std::vector<double>{ 1, 0 }, std::vector<double>{ 1 });
	

    return 0;
}
