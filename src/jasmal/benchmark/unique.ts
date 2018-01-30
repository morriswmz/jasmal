import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
import { Tensor } from '../lib/core/tensor';
const T = JasmalEngine.createInstance();
const suite = new BenchMark.Suite();

let max = 1000;
let x = <number[]>(<Tensor>T.div(T.randi(0, max, [100000]), max)).toArray(true);

suite.add('hash', () => {
    let u = new Array<number>();
    let h: {[key: number]: boolean} = {};
    let i: number, n = x.length;
    for (i = 0;i < n;i++) {
        if (!h.hasOwnProperty(<any>x[i])) {
            h[x[i]] = true;
            u.push(x[i]);
        }
    }
    u.sort((a, b) => {
        return a > b ? 1 : (a < b ? -1 : 0);
    });
    // verification
    if (u.length !== max + 1) {
        throw new Error('Result is incorrect.');
    }
    for (i = 0;i <= max;i++) {
        if (u[i] !== i / max) {
            throw new Error('Result is incorrect.');
        }
    }
    return u;
});

suite.add('sort', () => {
    let i: number, n = x.length;
    let y = new Array<number>(n);
    for (i = 0;i < n;i++) {
        y[i] = x[i];
    }
    y.sort((a, b) => {
        return a > b ? 1 : (a < b ? -1 : 0);
    });
    let nUnique = 1;
    let cur = y[0];
    for (i = 1;i < n;i++) {
        if (y[i] !== cur) {
            y[nUnique - 1] = cur;
            cur = y[i];
            nUnique++;
        }
    }
    y[nUnique - 1] = cur;
    let u = new Array<number>(nUnique);
    for (i = 0;i < nUnique;i++) {
        u[i] = y[i];
    }
    // verification
    if (u.length !== max + 1) {
        throw new Error('Result is incorrect.');
    }
    for (i = 0;i <= max;i++) {
        if (u[i] !== i / max) {
            throw new Error('Result is incorrect.');
        }
    }
    return u;
});

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
