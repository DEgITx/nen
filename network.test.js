const nen = require(`./build/Release/nen`)

  const inputData = [
    [0, 1], 
    [1, 1], 
    [0, 0], 
    [1, 0]
  ]
 
  const outputData = [
    [1],
    [0], 
    [0],
    [1] 
  ]

test('basic xor network', () => {
  const network = nen.NeuralNetwork(2, 1, 2, 4);
  network.setRate(0.02);
  const errors = network.train(inputData, outputData, { error: 0.5, sync: true });
  expect(errors.length).toBe(4);
  expect(errors[0]).toBeLessThan(0.01);
});

test('xor network forward', () => {
  const network = nen.NeuralNetwork(2, 1, 1, 8);
  network.setRate(0.02);
  network.train(inputData, outputData, { error: 0.5, sync: true });
  const output1 = network.forward([0, 1])[0];
  const output2 = network.forward([1, 1])[0];
  expect(output1).toBeGreaterThan(0.9);
  expect(output2).toBeLessThan(0.1);
});

test('async xor network', async () => {
  const network = nen.NeuralNetwork(2, 1, 2, 4);
  network.setRate(0.02);
  await network.train(inputData, outputData, { error: 0.5 });
  const output1 = network.forward([1, 0])[0];
  const output2 = network.forward([0, 0])[0];
  expect(output1).toBeGreaterThan(0.9);
  expect(output2).toBeLessThan(0.1);
});

test('genetic xor network', () => {
  const network = nen.NeuralNetwork(2, 1, 1, 4);
  network.setRate(0.3);
  
  network.train(inputData, {fitness: (a, b, i) => {
      const values1 = network.forward(inputData[i], a);
      const error1 = network.error(outputData[i]);
      const values2 = network.forward(inputData[i], b);
      const error2 = network.error(outputData[i]);
      return error1 < error2;
  }, error: (i) => { 
      const values = network.forward(inputData[i]);
      return network.error(outputData[i]);
  }}, { error: 0.5, sync: true });

  const output1 = network.forward([1, 0])[0];
  const output2 = network.forward([0, 0])[0];
  expect(output1).toBeGreaterThan(0.9);
  expect(output2).toBeLessThan(0.1);
});

test('async genetic xor network', async () => {
  const network = nen.NeuralNetwork(2, 1, 1, 4);
  network.setRate(0.3);
  await network.train(inputData, {fitness: (a, b, i) => {
      const values1 = network.forward(inputData[i], a);
      const error1 = network.error(outputData[i]);
      const values2 = network.forward(inputData[i], b);
      const error2 = network.error(outputData[i]);
      return error1 < error2;
  }, error: (i) => { 
      const values = network.forward(inputData[i]);
      return network.error(outputData[i]);
  }}, { error: 0.5 });
  const output1 = network.forward([1, 0])[0];
  const output2 = network.forward([0, 0])[0];
  expect(output1).toBeGreaterThan(0.9);
  expect(output2).toBeLessThan(0.1);
});
