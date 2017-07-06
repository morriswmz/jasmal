import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

[10, 50, 100, 500].forEach((dim) => {
    let A = T.rand([dim, dim]);
    let B = T.rand([dim, dim]);
    let vc = T.rand([dim, 1]);
    let vr = T.rand([1, dim]);
    suite.add(`Real [${dim}x1] x [1x${dim}]`, () => { T.matmul(vc, vr); });
    suite.add(`Real [1x${dim}] x [${dim}x${dim}]`, () => { T.matmul(vr, A); });    
    suite.add(`Real [${dim}x${dim}] x [${dim}x1]`, () => { T.matmul(A, vc); });
    suite.add(`Real [${dim}x${dim}] x [${dim}x${dim}]`, () => { T.matmul(A, B); });    
});

[10, 50, 100, 200].forEach((dim) => {
    let A = T.rand([dim, dim]);
    suite.add(`SVD [${dim}x${dim}]`, () => T.svd(A));
});

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
