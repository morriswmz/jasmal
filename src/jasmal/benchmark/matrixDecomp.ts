import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

[10, 50, 100, 200].forEach((dim) => {
    let A = T.rand([dim, dim]);
    suite.add(`LU Compact [${dim}x${dim}]`, () => T.lu(A, true));
    suite.add(`LU Full [${dim}x${dim}]`, () => T.lu(A, false));
});

[10, 50, 100, 200].forEach((dim) => {
    let A = T.rand([dim, dim]);
    suite.add(`QR [${dim}x${dim}]`, () => T.qr(A));
});

[10, 50, 100, 200].forEach((dim) => {
    let A = T.rand([dim, dim]);
    let B = T.matmul(A, A, T.MM_TRANSPOSED);
    suite.add(`Chol [${dim}x${dim}]`, () => T.chol(B));
});

[10, 50, 100, 200].forEach((dim) => {
    let A = T.rand([dim, dim]);
    T.add(A, T.transpose(A), true);
    suite.add(`Eigen Symmetric [${dim}x${dim}]`, () => T.eig(A));
});

[10, 50, 100, 200].forEach((dim) => {
    let A = T.rand([dim, dim]);
    suite.add(`Eigen General [${dim}x${dim}]`, () => T.eig(A));
});

[10, 50, 100, 200].forEach((dim) => {
    let A = T.rand([dim, dim]);
    suite.add(`SVD [${dim}x${dim}]`, () => T.svd(A));
});

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();