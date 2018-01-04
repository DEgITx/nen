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

### error( *[outputData]* )

get errors array from last forwarding

#### error( *[outputData]*, *[inputData]*, wId = undefined )

forward input data through the network and return errors

### iterations()

get number of iterations from last learning process


## Fitness Function learning example

Sometimes we don't know exact output values. For such cases we can compare two neural networks for best results using genetic algorithms and fitness function

``` js
network.setRate(0.3) // set learning rate

const errors = network.train(
	inputData, // input data set, contains 4 learning cases
	{
		fitness: (a, b, i) => { // compare network "a" with network "b" for set element number "i"
      		const errorA = network.error(outputData[i], inputData[i], a); // forward inputData through "a" network and get error for outputData
      		const errorB = network.error(outputData[i], inputData[i], b); // forward inputData through "b" network and get error for outputData
      		return errorA < errorB; // network A must be BETTER then network B
  		}, 
  		error: (i) => { // to know when to stop learning process send a callback with represents error from input "i"
      		const values = network.forward(inputData[i]); // forward input values and get results
      		return network.error(outputData[i]); // get errors from last forwarding
  		}
	}, 
	{ 
		error: 0.5, // learn process until reaching 0.5% errors
		sync: true // function call synchronously and returns value
	}
)

console.log(network.forward([0, 1])) // [ 0.999999999993257 ]
console.log(network.forward([0, 0])) // [ 9.793498051406418e-10 ]
```