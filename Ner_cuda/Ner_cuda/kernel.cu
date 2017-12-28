
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>

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

__host__ __device__ double transferFunction(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

__host__ __device__ double transferFunctionDerivative(double x)
{
	return (1.0 - x) * x;
}

__host__ __device__ void forwardKernel(int i, double *outputs, double *weightes, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons)
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
	outputs[offset_neuron + i] = transferFunction(sum);
}

__global__ void forwardKernelGPU(double *outputs, double *weightes, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons)
{
	int i = threadIdx.x;
	forwardKernel(i, outputs, weightes, layer, inputs, outputs_size, layers, neurons);
}

__host__ __device__ void calculateOutputDelta(int i, double *outputs, double *delta, double *targets, const unsigned outputs_offset)
{
	double delta_ = targets[i] - outputs[i + outputs_offset];
	delta[i + outputs_offset] = delta_ * transferFunctionDerivative(outputs[i + outputs_offset]);
}

__global__ void calculateOutputDeltaGPU(double *outputs, double *delta, double *targets, const unsigned outputs_offset)
{
	int i = threadIdx.x;
	calculateOutputDelta(i, outputs, delta, targets, outputs_offset);
}

__host__ __device__ void calculateHiddensDelta(int i, double *outputs, double *weightes, double *delta, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons)
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
	delta[i + offset_neuron] = dow * transferFunctionDerivative(outputs[i + offset_neuron]);
}

__global__ void calculateHiddensDeltaGPU(double *outputs, double *weightes, double *delta, const unsigned layer, const unsigned inputs, const unsigned outputs_size, const unsigned layers, const unsigned neurons)
{
	int i = threadIdx.x;
	calculateHiddensDelta(i, outputs, weightes, delta, layer, inputs, outputs_size, layers, neurons);
}

__host__ __device__ void updateInputWeights(
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

void forwardInput(double* neuron_outputs, double* neuron_weigths, unsigned inputs, unsigned outputs, unsigned layers, unsigned neurons, bool gpu)
{
	// forward
	for (unsigned layer = 1; layer <= layers + 1; ++layer)
	{
		unsigned threads = (layer == layers + 1) ? outputs : neurons;
		if (gpu)
		{
			forwardKernelGPU << <1, threads >> >(neuron_outputs, neuron_weigths, layer, inputs, outputs, layers, neurons);
			cudaDeviceSynchronize();
		}
		else
		{
			for(int i = 0; i < threads; ++i)
				forwardKernel(i, neuron_outputs, neuron_weigths, layer, inputs, outputs, layers, neurons);
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
	if (gpu)
	{
		calculateOutputDeltaGPU << <1, outputs >> > (neuron_outputs, neuron_delta, neuron_targets, outputs_offset_neurons);
		cudaDeviceSynchronize();
	}
	else
	{
		for (int i = 0; i < outputs; ++i)
			calculateOutputDelta(i, neuron_outputs, neuron_delta, neuron_targets, outputs_offset_neurons);
	}

	// calculate hidden deltas
	for (unsigned layer = layers; layer > 0; --layer)
	{
		if (gpu)
		{
			calculateHiddensDeltaGPU << <1, neurons + 1 >> >(neuron_outputs, neuron_weigths, neuron_delta, layer, inputs, outputs, layers, neurons);
			cudaDeviceSynchronize();
		}
		else
		{
			for (int i = 0; i < neurons + 1; ++i)
				calculateHiddensDelta(i, neuron_outputs, neuron_weigths, neuron_delta, layer, inputs, outputs, layers, neurons);
		}
	}

	// update weights
	for (unsigned layer = layers + 1; layer > 0; --layer)
	{
		unsigned threads = (layer == layers + 1) ? outputs : neurons;
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

	NeuronNetwork(unsigned inputs_, unsigned outputs_, unsigned layers_, unsigned neurons_, TrainingAlgorithm algorithm_ = StochasticGradient)
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
		cudaDeviceReset();
	}

	void forward(const std::vector<double> &i)
	{
		memcpy(neuron_outputs, i.data(), sizeof(double) * inputs);
		forwardInput(neuron_outputs, neuron_weigths, inputs, outputs, layers, neurons, gpu);
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

	double train(const double* i, const double* o)
	{
		memcpy(neuron_outputs, i, sizeof(double) * inputs);
		memcpy(neuron_targets, o, sizeof(double) * outputs);
		forwardInput(neuron_outputs, neuron_weigths, inputs, outputs, layers, neurons, gpu);
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
		iterations++;
		return err;
	}

	double train(const std::vector<double>& i, const std::vector<double>& o)
	{
		return train(i.data(), o.data());
	}

	std::vector<double> train(const std::vector<std::vector<double>> &i, const std::vector<std::vector<double>> &o)
	{
		assert(i.size() == o.size());
		std::vector<double> errors;
		for (unsigned n = 0; n < i.size(); ++n)
		{
			errors.push_back(train(i[n], o[n]));
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

	void trainWhileError(const std::vector<std::vector<double>> &i, const std::vector<std::vector<double>> &o, double errorPercent, double errorPercentAvrg)
	{
		std::vector<double> errors;
		do {
			errors = train(i, o);
		} while (([&]() {
			double errorAvrg = 0;
			for (auto error : errors)
			{
				errorAvrg += error;
				if (errorPercent > 0 && error * 100 >= errorPercent)
				{
					return true;
				}
			}
			errorAvrg /= errors.size();
			if (errorAvrg * 100 > errorPercentAvrg)
				return true;

			return false;
		})());
		printStatistic(errors);
	}

	void trainWhileError(double errorPercent, double errorPercentAvrg)
	{
		if (train_data_inputs.size() == 0 || train_data_outputs.size() == 0)
			return;
		trainWhileError(train_data_inputs, train_data_outputs, errorPercent, errorPercentAvrg);
	}

	void printStatistic(const std::vector<double>& errors)
	{
		system("cls");

		std::cout << "neurons = " << neurons_size << " w = " << neuron_weigths_size << " run on = " << (gpu ? "GPU" : "CPU") << "\n";
		std::cout << "r = " << rate << " m = " << momentum << " b1 = " << beta1 << " b2 = " << beta2 << " eps = " << d_epsilon << "\n";
		std::cout << "iterations: " << iterations << "\n";
		double avrg = 0;
		for (double error : errors)
		{
			std::cout << error * 100 << "%" << std::endl;
			avrg += error;
		}
		std::cout << "avrg error = " << (avrg / errors.size()) * 100 << "%" << std::endl;
	}

	void saveFile(const std::string& file)
	{
		std::ofstream f;
		f.open(file);
		f << inputs << "\n";
		f << outputs << "\n";
		f << layers << "\n";
		f << neurons << "\n";
		for(unsigned i = 0; i < neuron_weigths_size; ++i)
			f << neuron_weigths[i] << "\n";
		f.close();
	}

	void loadFile(const std::string& file)
	{
		std::ifstream f;
		f.open(file);
		if (!f.is_open())
			return;
		f >> inputs >> outputs >> layers >> neurons;
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

int main()
{
	//srand(time(NULL));
	//NeuronNetwork n(2, 1, 1, 3, StochasticGradient);
	NeuronNetwork n(2, 1, 25, 25, Adagrad);
	
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

	//auto a = std::vector<std::vector<double>>{ { 0, 1 },{ 1, 0 },{ 0, 0 },{ 1, 1 } };
	//auto b = std::vector<std::vector<double>>{ { 1 },{ 1 },{ 0 },{ 0 } };
	//n.trainWhileError(a, b, 0, 0.5);

    return 0;
}
