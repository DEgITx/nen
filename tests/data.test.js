const nen = require(`../build/Release/nen`)

const normCore = (x, min, max) => {
  return (x - min) / (max - min)
}

const norm = (data, min, max) => {
  return data.map((x) => normCore(x, min, max))
}

const deNormCore = (y, min, max) => {
  return min + y * (max - min)
}

const deNorm = (data, min, max) => {
  return data.map((y) => deNormCore(y, min, max))
}

log = Math.log;

test('mul', () => {
  const mulInput = [
    norm([log(1), log(3)], 0, 10),
    norm([log(2), log(7)], 0, 10),
    norm([log(6), log(5)], 0, 10),
    norm([log(5), log(5)], 0, 10),
    norm([log(2), log(3)], 0, 10),
    norm([log(1), log(8)], 0, 10),
    norm([log(7), log(7)], 0, 10),
    norm([log(3), log(6)], 0, 10),
    norm([log(6), log(6)], 0, 10),
    norm([log(8), log(4)], 0, 10),
    norm([log(10), log(5)], 0, 10),
    norm([log(6), log(7)], 0, 10),
    norm([log(8), log(8)], 0, 10),
    norm([log(9), log(9)], 0, 10),
    norm([log(5), log(8)], 0, 10),
    norm([log(5), log(7)], 0, 10)
  ]
  const mulOutput = [
    norm([log(3)], 0, 10),
    norm([log(14)], 0, 10),
    norm([log(30)], 0, 10),
    norm([log(25)], 0, 10),
    norm([log(6)], 0, 10),
    norm([log(8)], 0, 10),
    norm([log(49)], 0, 10),
    norm([log(18)], 0, 10),
    norm([log(36)], 0, 10),
    norm([log(32)], 0, 10),
    norm([log(50)], 0, 10),
    norm([log(42)], 0, 10),
    norm([log(64)], 0, 10),
    norm([log(81)], 0, 10),
    norm([log(40)], 0, 10),
    norm([log(35)], 0, 10)
  ]

  const network = nen.NeuralNetwork(2, 1, 1, 22);
  //network.setMultiThreads(false)
  //network.setShuffle(false)
  network.setRate(0.02);

  console.time('mul')
  const errors = network.train(mulInput, mulOutput, { error: 0.001, sync: true });
  console.timeEnd('mul')
  console.log('mul iterations', network.iterations())
  expect(errors.length).toBe(mulOutput.length);
  expect(Math.exp(deNorm(network.forward(norm([log(8), log(8)], 0, 10)), 0, 10))).toBeGreaterThan(60)
  expect(Math.exp(deNorm(network.forward(norm([log(8), log(8)], 0, 10)), 0, 10))).toBeLessThan(68)
  expect(Math.exp(deNorm(network.forward(norm([log(4), log(5)], 0, 10)), 0, 10))).toBeGreaterThan(16)
  expect(Math.exp(deNorm(network.forward(norm([log(4), log(5)], 0, 10)), 0, 10))).toBeLessThan(24)
});


test('momorize image', () => {
  const network = nen.NeuralNetwork(2, 3, 2, 31);
  network.setMultiThreads(true)
  network.setShuffle(true)
  network.setRate(0.004);
  network.loadData('tests/data/logo_memory_15600.data')
  console.time('mem image')
  const errors = network.train(null, null, { error: 5.5, sync: true });
  console.timeEnd('mem image')
  console.log('mem image iterations', network.iterations())
  expect(network.iterations()).toBeGreaterThan(120000)
});
