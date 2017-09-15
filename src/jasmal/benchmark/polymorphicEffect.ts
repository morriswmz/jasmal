import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

// Checks if the JIT will deoptimize the generated operations due to different
// typed array inputs.

let n = 512;
let input = T.randi(0, n * n, [n]);
let inputInt = input.asType(T.INT32);
let inputArr = input.toArray(true);

suite.add('Native sin()', () => {
    let result = new Array<number>(n);
    for (let i = 0;i < n;i++) {
        result[i] = Math.sin(inputArr[i]);
    }
    return result;
});

suite.add('JASMAL sin() Float64 input', () => T.sin(input));
suite.add('JASMAL sin() Int32', () => T.sin(inputInt));
suite.add('JASMAL sin() array input', () => T.sin(inputArr));
suite.add('JASMAL sin() Float64 input after changing of input type', () => T.sin(input));

let input2 = T.randi(0, n * n, [n]);
let input2Arr = input2.toArray(true);
let input2Int = input2.asType(T.INT32);

suite.add('Native add()', () => {
    let result = new Array<number>(n);
    for (let i = 0;i < n;i++) {
        result[i] = inputArr[i] + input2Arr[i];
    }
    return result;
});

suite.add('JASMAL add() Float64 + Float64', () => T.add(input, input2));
suite.add('JASMAL add() Float64 + array', () => T.add(input, input2Arr));
suite.add('JASMAL add() Float64 + Int32', () => T.add(input, input2Int));
suite.add('JASMAL add() Int32 + Int32', () => T.add(inputInt, input2Int));
suite.add('JASMAL add() Float64 + Float64 after changing of input types', () => T.add(input, input2));
suite.add('JASMAL add() Float64 + Float64 in-place', () => T.add(input, input2, true));

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
