NEN - Fast and Light Neural Network Library
==============

Neural Network library with with native CPU/GPU bindings and learning algorithms choice. 

## Instalation

``` sh
npm install nen
```

## Quick start

``` js
const NEuralNetwork = require('nen');

// create neural network 2x4 size
const network = NEuralNetwork(
	2, // with 2 inputs
	1, // 1 output
	2, // 2 hidden perceptron layers (width)
	4 // 4 perceptrons per layer (height)
);
// new NEuralNetwork(...) also acceptable

// set learning rate of the network
network.setRate(0.02); // default is 0.02

// now let input XOR learning data

// learning set from 4 lines
const inputData = [
	[0, 1], 
	[1, 1], 
	[0, 0], 
	[1, 0]
]

const outputData = [
	[1], // 0 xor 1 = 1
	[0], // 1 xor 1 = 0
	[0], // 0 xor 0 = 0
	[1]  // 1 xor 0 = 1
]

// start learning process until reaching average 0.5% errors (0.005)
const errors = network.train(
	inputData, 
	outputData, 
	{ error: 0.5, sync: true }
);

// lets try now our data through learned neural network
const output = network.forward([0, 1]);

console.log(output) // [ 0.9451534677309323 ]
// pretty close to 1 - good :)

```

## Performance

hidden layers size: *width* x *height*

rate: *0.02*

error: *0.005*

algorithm: *3* (default)

network size  | iterations | time
------------- | ---------- | ---------
1x4           | 3132       |  5.319ms
2x4           | 1512       |  5.163ms
4x4           | 1968       |  13.963ms
1x50          | 1584       |  29.877ms
3x12          | 1000       |  30.376ms


## API

### train( *[inputData]*, *[outputData]*, *options* )

train network with input and output data

returns Promise by default or errors array if sync setted to true in options

options:
* sync - is synchronously call (default: false) [or return Promise]
* error - min percent of errors until finish (default: 0 - one operation)
* iterations - iteration limitation (full cycle) (default: 0 - unlimit)
* iteration : Function - callback that called per full iteration cycle

#### train( fitness: Function, error: Function, *options* )

train using fitness function

### forward( *[inputData]* )

forward input data through network and return output result in array

#### forward( *[inputData]*, wId = undefined )

forward input data through wId network (userfull for fitness function learning)

### setRate( *rate* )

set learting rate
default: 0.02

### setAlgorithm( *algorithm* )

set learning algorithm
default: 3

### setActivation( *activationType* )

set actionvation function

* 0 - Logistic
* 1 - Hyperbolic tangent
* 2 - Identity
* 3 - Rectified linear unit

default: 0

### error( *[outputData]* )

get errors array from last forwarding

#### error( *[outputData]*, *[inputData]*, wId = undefined )

forward input data through the network and return errors

### iterations()

get number of iterations from last learning process

### setMoment( *moment* )

set moment factor

default: 0.3

### setDEpsilon( *dEspsilon* )

set divider epsilon

default: 10^-11

### setPopulation( *populationSize* )

population size for genetic algorithm

default: 10

### setElitePart( *populationPart* )

population part as elite part

default: 3 (means 1/3 of population)

### setElitePart( *shuffle* )

shuffle data between each iteration

default: true

### setMultiThreads( *multi* )

Enable multithread mini-batch learning. For more stable but slower learning disable this option.

default: true

## Fitness Function learning example

Sometimes we don't know exact input/output values, but know that neural network A better then neural network B. For such cases we can compare two neural networks for best results using genetic algorithms and fitness compare function

``` js
network.setRate(0.3) // set learning rate

let errors = [] // aray represents all errors of input data
let error = 1.0 // average error of input data

network.train(
	// compare network "a" with network "b" for set element number "i"
	(a, b, iteration) => {
		// get input data index from current iteration
		const i = iteration % inputData.length;
		// forward inputData through "a" network and get error for outputData
		const errorA = network.error(outputData[i], inputData[i], a);
		// forward inputData through "b" network and get error for outputData
		const errorB = network.error(outputData[i], inputData[i], b);
		// network A must be BETTER then network B
		return errorA < errorB;
	}, 
	// to know when to stop learning process send a callback
	// with represents average error from iteration
	(iteration) => {
		// get input data index from current iteration
		const i = iteration % inputData.length;
		// forward input values and get results
		const values = network.forward(inputData[i]);
		// get errors from last forwarding
		errors[i] = network.error(outputData[i]);
		// calculate average error
		if(i == errors.length - 1) // calucate error only on last input/output element
		{
			for(const e of errors)
				error += e
			error /= errors.length
		}
		// return average error to know when to stop learning process
		return error
	},
	{
		error: 0.5, // learn process until reaching 0.5% errors
		sync: true // function call synchronously
	}
)

console.log(network.forward([0, 1])) // [ 0.9999999999942144 ]
console.log(network.forward([0, 0])) // [ 9.965492560315975e-10 ]
```