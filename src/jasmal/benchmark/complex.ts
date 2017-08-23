import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
import { CMath } from '../lib/math/cmath';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

let reX = [1, -1, 0, 2];
let imX = [-1, 1, 2, 0];
let z = T.fromArray(reX, imX);
suite.add('native csin', () => {
    let reY = new Float64Array(reX.length);
    let imY = new Float64Array(imX.length);
    for (let i = 0;i < reX.length;i++) {
        [reY[i], imY[i]] = CMath.csin(reX[i], imX[i]);
    }
});
suite.add('Tensor csin', () => T.sin(z));

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();