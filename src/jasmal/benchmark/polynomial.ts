import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

let nCoeff = 10;

[1, 10, 50].forEach(m => {

    let P = T.rand([nCoeff]);
    let pArr = new Float64Array(P.realData);
    let X = T.rand([m, m]);
    let xArr = new Float64Array(X.realData);

    suite.add(`Native polynomial evaluation ${m}x${m}`, () => {
        let output = new Float64Array(m * m);
        let m2 = m * m;
        for (let i = 0;i < m2;i++) {
            let result = pArr[0];
            let x = xArr[i];
            for (let k = 1;k < pArr.length;k++) {
                result = result * x + pArr[k];
            }
            output[i] = result;
        }
        return output;
    });

    suite.add(`JASMAL polynomial evaluation ${m}x${m}`, () => {
        return T.polyval(P, X);
    });

});

[2, 5, 20].forEach(m => {
    
    let P = T.rand([nCoeff]);
    let pArr = new Float64Array(P.realData);
    let X = T.rand([m, m]);
    let xArr = new Float64Array(X.realData);

    suite.add(`Native matrix polynomial evaluation ${m}x${m}`, () => {
        let acc = new Float64Array(m * m);
        let i: number, j: number, k: number, l: number;
        for (k = 0;k < m;k++) {
            acc[k * m + k] = pArr[0];
        }
        for (l = 1;l < pArr.length;l++) {
            // multiply
            let out = new Float64Array(m * m);
            for (i = 0;i < m;i++) {
                for (j = 0;j < m;j++) {
                    let s = 0;
                    for (k = 0;k < m;k++) {
                        s += acc[i * m + k] * xArr[k * m + j];
                    }
                    out[i * m + j] = s;
                }
            }
            // add
            for (i = 0;i < m;i++) {
                out[i * m + i] += pArr[l];
            }
            // update
            acc = out;
        }
        return acc;
    });

    suite.add(`JASMAL matrix polynomial evaluation ${m}x${m}`, () => {
        return T.polyvalm(P, X);
    });

});


suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
