import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

suite.add('Native random', () => Math.random());
suite.add('Jasmal random', () => T.rand());

let batchSize = 1024;
suite.add('Native random in batch', () => {
    var result = new Float64Array(batchSize);
    for (let i = 0;i < result.length;i++) {
        result[i] = Math.random();
    }
    return result;
});
suite.add('Jasmal random in batch', () => {
    return T.rand([batchSize]);
});

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
