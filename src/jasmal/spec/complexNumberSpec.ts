import { CMath } from '../lib/math/cmath';
import { ComplexNumber } from '../lib/complexNumber';
import { checkComplex } from './testHelper';

describe('complex arithmetic', () => {
    it('should perform complex division', () => {
        let x: ComplexNumber, y: ComplexNumber, z: ComplexNumber;
        for (let i = 0;i < 100;i++) {
            x = new ComplexNumber((Math.random() - 0.5) * 100, (Math.random() - 0.5) * 100);
            y = new ComplexNumber((Math.random() - 0.5) * 100, (Math.random() - 0.5) * 100);
            z = x.divc(y);
            checkComplex(z.mulc(y), x, 1e-10);
        }
    });
    it('should compute the square root correctly', () => {
        let x: ComplexNumber, y: ComplexNumber;
        for (let i = 0;i < 100;i++) {
            x = new ComplexNumber((Math.random() - 0.5) * 100, (Math.random() - 0.5) * 100);
            let [reY, imY] = CMath.csqrt(x.re, x.im);
            y = new ComplexNumber(reY, imY);
            checkComplex(y.mulc(y), x, 1e-10);
        }
    });
});
