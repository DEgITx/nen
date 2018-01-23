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
  expect(output1).toBeGreaterThan(0.895);
  expect(output2).toBeLessThan(0.1);
});

test('genetic xor network', () => {
  const network = nen.NeuralNetwork(2, 1, 1, 4);
  network.setRate(0.3);
  
  let errors = []
  let error = 1
  network.train((a, b, iteration) => {
      const i = iteration % inputData.length;
      const error1 = network.error(outputData[i], inputData[i], a);
      const error2 = network.error(outputData[i], inputData[i], b);
      return error1 < error2;
  }, (iteration) => {
      const i = iteration % inputData.length;
      const values = network.forward(inputData[i]);
      errors[i] = network.error(outputData[i]);
      if(i == errors.length - 1)
      {
        for(const e of errors)
          error += e
        error /= errors.length
      }
      return error
  }, { error: 0.5, sync: true })

  const output1 = network.forward([1, 0])[0];
  const output2 = network.forward([0, 0])[0];
  expect(output1).toBeGreaterThan(0.9);
  expect(output2).toBeLessThan(0.1);
});

test('async genetic xor network', async () => {
  const network = nen.NeuralNetwork(2, 1, 1, 4);
  network.setRate(0.3);

  let called = 0
  let errors = []
  let error = 1
  await network.train((a, b, iteration) => {
      const i = iteration % inputData.length;
      const error1 = network.error(outputData[i], inputData[i], a);
      const error2 = network.error(outputData[i], inputData[i], b);
      return error1 < error2;
  }, (iteration) => {
      const i = iteration % inputData.length;
      const values = network.forward(inputData[i]);
      errors[i] = network.error(outputData[i]);
      if(i == errors.length - 1)
      {
        for(const e of errors)
          error += e
        error /= errors.length
      }
      return error
  }, { error: 0.5, iteration: (i, error) => {
    called++;
  } })

  const output1 = network.forward([1, 0])[0];
  const output2 = network.forward([0, 0])[0];
  expect(output1).toBeGreaterThan(0.9);
  expect(output2).toBeLessThan(0.1);
  expect(called).toBeGreaterThan(0);
});

test('iteration callback test sync', () => {
  const network = nen.NeuralNetwork(2, 1, 1, 4);
  network.setRate(0.1);
  let called = 0;
  network.train(inputData, outputData, { error: 0.5, sync: true, iteration: (i, error) => {
    called++;
  }});
  expect(called).toBe(network.iterations() / 4);
});

test('iteration callback test async', async () => {
  const network = nen.NeuralNetwork(2, 1, 1, 4);
  network.setRate(0.1);
  let called = 0;
  await network.train(inputData, outputData, { error: 0.5, sync: false, iteration: (i, error) => {
    called++;
  }});
  expect(called).toBe(network.iterations() / 4);
  expect(called).toBeGreaterThan(0);
});

test('limit iterations', () => {
  const network = nen.NeuralNetwork(2, 1, 4, 8);
  network.setRate(0.02);
  network.train(inputData, outputData, { error: 0.5, sync: true, iterations: 7 });
  expect(network.iterations()).toBe(7 * 4);
});