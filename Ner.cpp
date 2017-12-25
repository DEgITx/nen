#include "stdafx.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>

enum TrainingAlgorithm {
	StochasticGradient = 0,
	Adagrad,
	RMSProp,
	Adam
};

class Neuron
{
public:
	struct Connection
	{
		double weight;
		double deltaWeight = 0.0;
		double e = 0.0;
		double m = 0.0;
		double v = 0.0;
		double t = 1;
	};
	Neuron(unsigned index, TrainingAlgorithm algorithm = StochasticGradient);
	Neuron(unsigned index, unsigned numWeights, TrainingAlgorithm algorithm = StochasticGradient);
	Neuron(unsigned index, const std::vector<double>& weights, TrainingAlgorithm algorithm = StochasticGradient);
	Neuron(unsigned index, const std::vector<Connection>& weights, TrainingAlgorithm algorithm = StochasticGradient);
	typedef std::vector<Neuron> Layer;
	void forward(const Layer &prevLayer); // calculate value throwth neuron network
	void setOutputValue(double val) { m_output = val; }
	double outputValue(void) const { return m_output; }
	void calculateOutputGradients(double targetVals);
	void calculateHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	const std::vector<Connection>& connections() const { return m_outputWeights; };
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	void setTrainingAlgorithm(TrainingAlgorithm algorithm) {
		m_algorithm = algorithm;
	};
protected:
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
private:
	double sumDOW(const Layer &nextLayer) const;

	unsigned m_index;
	double m_delta;
	double m_output;
	std::vector<Connection> m_outputWeights;
	TrainingAlgorithm m_algorithm;
	// learnings vars
	static double rate;
	static double momentum;
	static double beta1;
	static double beta2;
	static double d_epsilon;
};

double Neuron::beta1 = 0.9;
double Neuron::beta2 = 0.999;
double Neuron::rate = 0.01;
double Neuron::momentum = 0.3;
double Neuron::d_epsilon = 0.0000001;

double Neuron::transferFunction(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

double Neuron::transferFunctionDerivative(double x)
{
	return (1.0 - x) * x;
}

Neuron::Neuron(unsigned index, TrainingAlgorithm algorithm) : m_index(index), m_algorithm(algorithm)
{

}

Neuron::Neuron(unsigned index, unsigned numWeights, TrainingAlgorithm algorithm) : m_index(index), m_algorithm(algorithm)
{
	for (unsigned i = 0; i < numWeights; ++i)
	{
		Connection connection;
		connection.weight = randomWeight();
		m_outputWeights.push_back(connection);
	}
}

Neuron::Neuron(unsigned index, const std::vector<double>& weights, TrainingAlgorithm algorithm) : m_index(index), m_algorithm(algorithm)
{
	for (double w : weights)
	{
		Connection connection;
		connection.weight = w;
		m_outputWeights.push_back(connection);
	}
}

Neuron::Neuron(unsigned index, const std::vector<Connection>& weights, TrainingAlgorithm algorithm) : m_index(index), m_algorithm(algorithm)
{
	for (const Connection& w : weights)
	{
		m_outputWeights.push_back(w);
	}
}

void Neuron::forward(const Layer &prevLayer)
{
	double sum = 0.0;
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].m_output * prevLayer[n].m_outputWeights[m_index].weight;
	}
	m_output = transferFunction(sum);
}

void Neuron::calculateOutputGradients(double targetVals)
{
	double delta = targetVals - m_output;
	m_delta = delta * transferFunctionDerivative(m_output);
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_delta = dow * transferFunctionDerivative(m_output);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_delta;
	}
	return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;
		double gradient = neuron.m_output * m_delta;
		double newDeltaWeight;

		switch (m_algorithm)
		{
		case StochasticGradient:
		{
			newDeltaWeight = rate * gradient + momentum * oldDeltaWeight;
			
			break;
		}
		case Adagrad:
		{
			double& e = neuron.m_outputWeights[m_index].e;
			e = e + pow(gradient, 2);
			newDeltaWeight = rate * gradient / sqrt(e + d_epsilon);

			break;
		}
		case RMSProp:
		{
			double& e = neuron.m_outputWeights[m_index].e;
			e = momentum * e + (1 - momentum) * pow(gradient, 2);
			newDeltaWeight = rate * gradient / sqrt(e + d_epsilon);
			
			break;
		}
		case Adam:
		{
			double& m = neuron.m_outputWeights[m_index].m;
			double& v = neuron.m_outputWeights[m_index].v;
			double& t = neuron.m_outputWeights[m_index].t;

			m = beta1 * m + (1 - beta1) * gradient;
			v = beta2 * v + (1 - beta2) * pow(gradient, 2);

			double mt = m / (1 - pow(beta1, t));
			double mv = v / (1 - pow(beta2, t));
			t++;

			newDeltaWeight = rate * mt / sqrt(mv + d_epsilon);

			break;
		}
		default:
			assert(false);
			break;
		}

		neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_index].weight += newDeltaWeight;
	}
}

class NeuronNetwork
{
public:
	NeuronNetwork(unsigned inputs, unsigned outputs, unsigned layers, unsigned sizeNeurons, std::vector<double> weights = std::vector<double>(), bool with_bias = true);
	NeuronNetwork();
	void forward(const std::vector<double> &inputVals);
	void backPropagation(const std::vector<double> &targetVals);
	std::vector<double> output() const;
	double train(const std::vector<double> &inputVals, const std::vector<double> &targetVals);
	std::vector<double> train(const std::vector<std::vector<double>> &inputVals, const std::vector<std::vector<double>> &targetVals);
	void trainWhileError(const std::vector<std::vector<double>> &inputVals, const std::vector<std::vector<double>> &targetVals, double errorPercent, double errorPercentAvrg = 0);
	void saveFile(const std::string& file);
	void loadFile(const std::string& file);
	void setWithBias(bool wb) { m_with_bias = wb; };
	void setTrainingAlgorithm(TrainingAlgorithm algo);
private:
	std::vector<Neuron::Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
	bool m_with_bias;
	unsigned error_line = 0;
	TrainingAlgorithm m_algorithm = StochasticGradient;
};

NeuronNetwork::NeuronNetwork() : m_with_bias(true)
{

}

NeuronNetwork::NeuronNetwork(unsigned inputs, unsigned outputs, unsigned layers, unsigned sizeNeurons, std::vector<double> weights, bool with_bias) : m_with_bias(with_bias)
{
	Neuron::Layer inputLayer;
	unsigned w = 0;
	for (unsigned i = 0; i < inputs; i++)
	{
		std::vector<double> ws;
		if (weights.size() > 0)
		{
			for (unsigned n = 0; n < sizeNeurons; n++)
			{
				ws.push_back(weights[w++]);
			}
			inputLayer.push_back(Neuron(i, ws, m_algorithm));
		}
		else
		{
			inputLayer.push_back(Neuron(i, sizeNeurons, m_algorithm));
		}
		if (i == inputs - 1)
		{
			if (weights.size() > 0)
			{
				Neuron neur = Neuron(i + 1, ws, m_algorithm);
				neur.setOutputValue(m_with_bias ? 1 : 0);
				inputLayer.push_back(neur);
			}
			else
			{
				Neuron neur = Neuron(i + 1, sizeNeurons, m_algorithm);
				neur.setOutputValue(m_with_bias ? 1 : 0);
				inputLayer.push_back(neur);
			}
		}
	}
	m_layers.push_back(inputLayer);

	for (unsigned layer = 0; layer < layers; layer++)
	{
		Neuron::Layer newLayer;
		for (unsigned neurons = 0; neurons < sizeNeurons; neurons++)
		{
			std::vector<double> ws;
			unsigned connections = (layer < layers - 1) ? sizeNeurons : outputs;
			if (weights.size() > 0)
			{
				for (unsigned n = 0; n < connections; n++)
				{
					ws.push_back(weights[w++]);
				}
				newLayer.push_back(Neuron(neurons, ws, m_algorithm));
			}
			else
			{
				newLayer.push_back(Neuron(neurons, connections, m_algorithm));
			}
			if (neurons == sizeNeurons - 1)
			{
				if (weights.size() > 0)
				{
					Neuron neur = Neuron(neurons + 1, ws, m_algorithm);
					neur.setOutputValue(m_with_bias ? 1 : 0);
					newLayer.push_back(neur);
				}
				else
				{
					Neuron neur = Neuron(neurons + 1, connections, m_algorithm);
					neur.setOutputValue(m_with_bias ? 1 : 0);
					newLayer.push_back(neur);
				}
			}
		}
		m_layers.push_back(newLayer);
	}

	Neuron::Layer outLayer;
	for (unsigned i = 0; i < outputs; i++)
	{
		outLayer.push_back(Neuron(i, m_algorithm));
	}
	outLayer.push_back(Neuron(outputs, m_algorithm));
	m_layers.push_back(outLayer);
}

void NeuronNetwork::forward(const std::vector<double> &inputVals)
{
	// Check the num of inputVals euqal to neuronnum expect bias
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputValue(inputVals[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Neuron::Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].forward(prevLayer);
		}
	}
}

void NeuronNetwork::backPropagation(const std::vector<double> &targetVals)
{
	// Calculate overal net error (RMS of output neuron errors)
	Neuron::Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].outputValue();
		m_error += delta *delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	
	error_line++;
	//std::cout << ++error_line << ": " << m_error * 100 << std::fixed << std::endl;
	//m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:
	//m_recentAverageError =
	//	(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
	//	/ (m_recentAverageSmoothingFactor + 1.0);
	
	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calculateOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Neuron::Layer &hiddenLayer = m_layers[layerNum];
		Neuron::Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Neuron::Layer &layer = m_layers[layerNum];
		Neuron::Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

std::vector<double> NeuronNetwork::output() const
{
	std::vector<double> out;
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		out.push_back(m_layers.back()[n].outputValue());
	}
	return out;
}

double NeuronNetwork::train(const std::vector<double> &inputVals, const std::vector<double> &targetVals)
{
	forward(inputVals);
	backPropagation(targetVals);
	return m_error;
}

std::vector<double> NeuronNetwork::train(const std::vector<std::vector<double>> &inputVals, const std::vector<std::vector<double>> &targetVals)
{
	assert(inputVals.size() == targetVals.size());
	std::vector<double> errors;
	for (unsigned n = 0; n < inputVals.size(); ++n)
	{
		errors.push_back(train(inputVals[n], targetVals[n]));
	}
	// print
	static auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	if (diff > 1000 * 1000 * 56)
	{
		start = finish;
		system("cls");
		double avrg = 0;
		for (double error : errors)
		{
			std::cout << error * 100 << "%" << std::endl;
			avrg += error;
		}
		std::cout << "avrg " << (avrg / errors.size()) * 100 << "%" << std::endl;
	}
	
	return errors;
}

void NeuronNetwork::trainWhileError(const std::vector<std::vector<double>> &inputVals, const std::vector<std::vector<double>> &targetVals, double errorPercent, double errorPercentAvrg)
{
	std::vector<double> errors;
	do {
		errors = train(inputVals, targetVals);
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
}

void NeuronNetwork::saveFile(const std::string& file)
{
	std::ofstream f;
	f.open(file);
	f << m_layers[0].size() - 1 << "\n";
	f << m_layers.back().size() - 1 << "\n";
	f << m_layers.size() - 2 << "\n";
	f << m_layers[1].size() - 1 << "\n";
	for (const Neuron::Layer& layer : m_layers)
	{
		for (const Neuron& neuron : layer)
		{
			for (const Neuron::Connection& connection : neuron.connections())
			{
				f << connection.weight << "\n";
			}
		}
	}
	f.close();
}

void NeuronNetwork::loadFile(const std::string& file)
{
	std::ifstream f;
	f.open(file);
	m_layers.clear();
	unsigned inputs, outputs, layers, neurons;
	f >> inputs >> outputs >> layers >> neurons;
	inputs++;
	outputs++;
	neurons++;
	layers += 2;
	m_layers.push_back(Neuron::Layer());
	double weight, delta;
	for (unsigned i = 0; i < inputs; ++i)
	{
		std::vector<Neuron::Connection> connections;
		for (unsigned c = 0; c < neurons - 1; c++)
		{
			Neuron::Connection connection;
			f >> connection.weight;
			connections.push_back(connection);
		}
		Neuron neur = Neuron(i, connections);
		if (i == inputs - 1)
			neur.setOutputValue(m_with_bias ? 1 : 0);
		m_layers.back().push_back(neur);
	}
	for (unsigned j = 0; j < layers - 2; ++j)
	{
		m_layers.push_back(Neuron::Layer());
		for (unsigned i = 0; i < neurons; ++i)
		{
			std::vector<Neuron::Connection> connections;
			for (unsigned c = 0; c < ((j == layers - 3) ? outputs - 1 : neurons - 1); c++)
			{
				Neuron::Connection connection;
				f >> connection.weight;
				connections.push_back(connection);
			}
			Neuron neur = Neuron(i, connections);
			if (i == neurons - 1)
				neur.setOutputValue(m_with_bias ? 1 : 0);
			m_layers.back().push_back(neur);
		}
	}
	m_layers.push_back(Neuron::Layer());
	for (unsigned i = 0; i < outputs; ++i)
	{
		m_layers.back().push_back(Neuron(i));
	}
	f.close();
}

void NeuronNetwork::setTrainingAlgorithm(TrainingAlgorithm algo)
{
	m_algorithm = algo;
	for (Neuron::Layer& layer : m_layers)
	{
		for (Neuron& neuron : layer)
		{
			neuron.setTrainingAlgorithm(algo);
		}
	}
}

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
	std::cout.precision(4);
	NeuronNetwork n1(2, 1, 19, 19);
	n1.setTrainingAlgorithm(Adagrad);
	
	/*
	n1.trainWhileError({
			{1, 0},
			{0, 1},
			{0, 0},
			{1, 1},
		}, { 
			{ 1 },
			{ 1 },
			{ 0 },
			{ 0 },
		}, 0.6);
	
	n1.forward({ 1, 0 });
	for (auto& o : n1.output())
		std::cout << "out " << o << std::endl;

	*/
	
	/*

	n1.trainWhileError({
		normalizeInput({ 3, 3 }, 0, 10),
		normalizeInput({ 2, 7 }, 0, 10),
		normalizeInput({ 6, 1 }, 0, 10),
		normalizeInput({ 4, 0 }, 0, 10),
		normalizeInput({ 5, 5 }, 0, 10),
		normalizeInput({ 2, 3 }, 0, 10),
		normalizeInput({ 0, 0 }, 0, 10),
		normalizeInput({ 1, 8 }, 0, 10),
	}, {
		normalizeInput(std::vector<double>{ 6 }, 0, 10),
		normalizeInput(std::vector<double>{ 9 }, 0, 10),
		normalizeInput(std::vector<double>{ 7 }, 0, 10),
		normalizeInput(std::vector<double>{ 4 }, 0, 10),
		normalizeInput(std::vector<double>{ 10 }, 0, 10),
		normalizeInput(std::vector<double>{ 5 }, 0, 10),
		normalizeInput(std::vector<double>{ 0 }, 0, 10),
		normalizeInput(std::vector<double>{ 9 }, 0, 10),
	}, 0.6);

	n1.forward(normalizeInput({ 2, 2 }, 0, 10));
	for (auto& o : n1.output())
		std::cout << "out " << deNormalizeOutput(o, 0, 10) << std::endl;

	n1.saveFile("example.txt");

	NeuronNetwork n2;
	n2.loadFile("example.txt");
	
	n2.forward(normalizeInput({ 0, 6 }, 0, 10));
	for (auto& o : n2.output())
		std::cout << "out " << deNormalizeOutput(o, 0, 10) << std::endl;

	*/
	/*
	auto start = std::chrono::high_resolution_clock::now();
	n1.trainWhileError({
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
	auto finish = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	std::cout << "time: " << diff / (1000 * 1000) << " ms" << std::endl;

	n1.forward(normalizeInput({ log(8), log(8) }, 0, 10));
	for (auto& o : n1.output())
		std::cout << "out " << exp(deNormalizeOutput(o, 0, 10)) << std::endl;
	*/

	NeuronNetwork n2;
	n2.loadFile("mul.ner");
	n2.forward(normalizeInput({ log(8), log(8) }, 0, 10));
	for (auto& o : n2.output())
		std::cout << "out " << exp(deNormalizeOutput(o, 0, 10)) << std::endl;

    return 0;
}