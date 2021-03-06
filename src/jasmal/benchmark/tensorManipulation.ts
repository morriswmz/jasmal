import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

let A = T.fromArray([[1, 2], [3, 4]]);

suite.add('Tile a 2x2 matrix', () => {T.tile(A, [10, 10, 10]); });

let B = T.randn([100, 1]);

suite.add('Transpose 100x1 using transpose()', () => {
    return T.transpose(B);
});
suite.add('Transpose 100x1 using permuteAxis()', () => {
    return T.permuteAxis(B, [1, 0]);
});


suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
