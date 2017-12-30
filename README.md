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
const errors = network.train(inputData, outputData, { error: 0.5 });

// lets try now our data through learned neural network
const output = network.forward([0, 1]);

console.log(output) // [ 0.9223849393847503 ]

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

### forward( *[inputData]* )

forward input data throuth network and return output result in array

### setRate( *rate* )

set learting rate
default: 0.01

### setAlgorithm( *algorithm* )

set learning algorithm
default: 3