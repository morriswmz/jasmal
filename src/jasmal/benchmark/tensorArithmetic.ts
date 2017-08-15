import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

let n = 50;

(() => {
    let x = 1;
    suite.add('Native scalar sin()', () => Math.sin(x));
    suite.add('Tensor scalar sin()', () => <number>T.sin(x));
})();

(() => {
    let arrA = new Float64Array(n * n);
    for (let i = 0;i < n * n;i++) {
        arrA[i] = Math.random() - 0.5;
    }
    let A = T.fromArray(arrA);
    suite.add('Native negation', () => {
        let arrC = new Float64Array(n * n);
        for (let i = 0;i < n * n;i++) {
            arrC[i] = -arrA[i]
        }
    });
    suite.add('Tensor negation', () => {
        T.neg(A);;
    });
})();

(() => {
    let arrA = new Float64Array(n * n);
    let arrB = new Float64Array(n * n);
    for (let i = 0;i < n * n;i++) {
        arrA[i] = Math.random() - 0.5;
        arrB[i] = Math.random() - 0.5;
    }
    let A = T.fromArray(arrA);
    let B = T.fromArray(arrB);
    suite.add('Native subtraction', () => {
        let arrC = new Float64Array(n * n);
        for (let i = 0;i < n * n;i++) {
            arrC[i] = arrA[i] - arrB[i];
        }
        return arrC;
    });
    suite.add('Tensor subtraction', () => {
        return T.sub(A, B);
    });
})();

(() => {
    let arrA = new Float64Array(n);
    let arrB = new Float64Array(n);
    for (let i = 0;i < n;i++) {
        arrA[i] = Math.random() - 0.5;
        arrB[i] = Math.random() - 0.5;
    }
    let A = T.fromArray(arrA).reshape([-1,1]);
    let B = T.fromArray(arrB);
    suite.add('Native multiplication with broadcasting', () => {
        let arrC = new Float64Array(n * n);
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                arrC[i * n + j] = arrA[i] * arrB[j];
            }
        }
        return arrC;
    });
    suite.add('Tensor multiplication with broadcasting', () => {
        return T.mul(A, B);
    });
})();

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
