import { JasmalEngine } from '..';
import { Tensor } from '../lib/tensor';
import { checkTensor } from './testHelper';
import { EPSILON } from '../lib/constant';
const T = JasmalEngine.createInstance();

describe('abs()', () => {
    it('should return a scalar if the input is a scalar', () => {
        expect(T.abs(-1)).toEqual(1);
        expect(T.abs(T.complexNumber(1, -2))).toEqual(Math.sqrt(5));
    });
    it('should return a tensor even if the input is a one-element array/tensor', () => {
        checkTensor(T.abs([-1]), Tensor.scalar(1));
        checkTensor(T.abs([[-1]]), T.fromArray([[1]]));
        checkTensor(T.abs(T.fromArray([1], [-2])), Tensor.scalar(Math.sqrt(5)));
    });
    it('should return the absolute values of real numbers', () => {
        let actual = T.abs([-1, 1, NaN, Infinity, -Infinity]);
        let expected = T.fromArray([1, 1, NaN, Infinity, Infinity]);
        checkTensor(actual, expected);
    });
    it('should also return the absolute values of complex numbers', () => {
        let actual = T.abs(T.fromArray([1, 1, Infinity], [-1, 0, 2]));
        let expected = T.fromArray([Math.sqrt(2), 1, Infinity]);
        checkTensor(actual, expected, EPSILON);
    });
    it('should calculate the absolute values of real numbers in place', () => {
        let x = T.fromArray([-7, 2, 0], [], T.INT32);
        T.abs(x, true);
        let expected = T.fromArray([7, 2, 0], [], T.INT32);
        checkTensor(x, expected);
    });
    it('should calculate the absolute values of complex numbers in place', () => {
        let x = T.fromArray([-1, 0, 1], [0, -4, 1]);
        T.abs(x, true);
        let expected = T.fromArray([1, 4, Math.sqrt(2)], [0, 0, 0]);
        checkTensor(x, expected);
    });
    it('should throw if in place calculation is not possible', () => {
        // input is a scalar
        let case1 = () => { T.abs(1, true); };
        // input is a array
        let case2 = () => { T.abs([-1, 1], true); };
        expect(case1).toThrow();
        expect(case2).toThrow();
    });
});

describe('conj()', () => {
    it('should return the same value for real numbers', () => {
        let actual = T.conj(T.fromArray([1, -3.14, NaN, Infinity]));
        let expected = T.fromArray([1, -3.14, NaN, Infinity]);
        checkTensor(actual, expected);
    });
    it('should return the conjugate for complex numbers', () => {
        let actual = T.conj(T.fromArray([1, -3.14, 1e9], [-1, Infinity, 1e8]));
        let expected = T.fromArray([1, -3.14, 1e9], [1, -Infinity, -1e8]);
        checkTensor(actual, expected);
    });
});

describe('sqrt()', () => {
    it('should return the square roots for nonnegative numbers', () => {
        let actual = T.sqrt(T.fromArray([0, 2, 16, NaN, Infinity]));
        let expected = T.fromArray([0, Math.sqrt(2), 4, NaN, Infinity]);
        checkTensor(actual, expected);
    });
    it('should return the square roots for negative numbers', () => {
        let actual = T.sqrt(T.fromArray([1, -1, -Infinity, -2]));
        let expected = T.fromArray([1, 0, 0, 0], [0, 1, Infinity, Math.sqrt(2)]);
        checkTensor(actual, expected);
    });
    it('should return the square roots for complex numbers', () => {
        let x = T.fromArray(
            [0, 1, -2, 0, 3, Infinity, -Infinity, Infinity, 3, -10],
            [0, 1, 0, -2, 4, -9999999, 0.0000001, Infinity, Infinity, -Infinity]
        )
        let actual = T.sqrt(x);
        let expected = T.fromArray(
            [0, 1.098684113467810, 0, 1, 2, Infinity, 0, Infinity, Infinity, Infinity],
            [0, 0.455089860562227, Math.sqrt(2), -1, 1, 0, Infinity, Infinity, Infinity, -Infinity]
        );
        checkTensor(actual, expected, 1e-14);
    });
    it('should calculate the square roots in place for real numbers', () => {
        let x = T.fromArray([-1, 2, -3]);
        T.sqrt(x, true);
        let expected = T.fromArray([0, Math.sqrt(2), 0], [1, 0, Math.sqrt(3)]);
        checkTensor(x, expected);
    });
});

describe('square()', () => {
    it('should return the squares of real numbers', () => {
        let actual = T.square([-1, 1.5, 3]);
        let expected = T.fromArray([1, 1.5*1.5, 9]);
        checkTensor(actual, expected);
    });
    it('should compute the squares of real numbers in place', () => {
        let x = T.fromArray([-2, Infinity, 0.5]);
        T.square(x, true);
        let expected = T.fromArray([4, Infinity, 0.5*0.5]);
        checkTensor(x, expected);
    });
    it('should return the squares of complex numbers', () => {
        let x = T.fromArray([-1, 5], [2.5, 3]);
        let actual = T.square(x);
        let expected = T.fromArray([-5.25, 16], [-5, 30]);
        checkTensor(actual, expected);
    });
});

describe('exp()', () => {
    it('should compute the exponentiation of real numbers', () => {
        let actual = T.exp([0, -2, 3.3, -Infinity, Infinity]);
        let expected = T.fromArray([1, Math.exp(-2), Math.exp(3.3), 0, Infinity]);
        checkTensor(actual, expected, EPSILON);
    });
    it('should compute the exponentiation of complex numbers', () => {
        let z = T.fromArray([0, 2, 4], [-1, -0.1, 2]);
        let actual = T.exp(z);
        let expected = T.fromArray(
            [ 0.54030230586813977,  7.35214159590899640, -22.720847417619233],
            [-0.84147098480789650, -0.73767471615133029,  49.645957334580565]
        );
        checkTensor(actual, expected, EPSILON * 50);
    });
});

describe('log()', () => {
    it('should compute the logarithm of positive numbers', () => {
        let actual = T.log([0.1, 1, Math.E, 1e12]);
        let expected = T.fromArray([Math.log(0.1), 0, 1, Math.log(1e12)]);
        checkTensor(actual, expected, EPSILON);
    });
    it('should compute the logarithm of real numbers', () => {
        let actual = T.log([0, -1, -Math.E, -1e12]);
        let expected = T.fromArray(
            [-Infinity, 0, 1, Math.log(1e12)],
            [0, Math.PI, Math.PI, Math.PI]
        );
        checkTensor(actual, expected, EPSILON);
    });
    it('should compute the logarithm of complex numbers', () => {
        let z = T.fromArray([0, 0.5, -3], [-1, 0.8, 9]);
        let actual = T.log(z);
        let expected = T.fromArray(
            [0, -5.8266908127975671e-2, 2.2499048351651325],
            [-1.5707963267948966, 1.0121970114513341, 1.8925468811915389]
        );
        checkTensor(actual, expected, 1e-14);
    });
});

// TODO: add math tests
describe('sin()', () => {
    it('should compute the sine of real numbers', () => {
        let actual = T.sin([0, Math.PI/2, 0.5]);
        let expected = T.fromArray([0, Math.sin(Math.PI/2), Math.sin(0.5)]);
        checkTensor(actual, expected, EPSILON);
    });
    it('should compute the sine of complex numbers', () => {
        let z = T.fromArray([0.5, 0.8], [-6, 4.6]);
        let actual = T.sin(z);
        let expected = T.fromArray(
            [96.707627492897842, 35.686445260153995],
            [-177.01994941200556, 34.652193500570938]
        );
        checkTensor(actual, expected, 1e-12);
    });
});

describe('cos()', () => {
    it('should compute the cosine of real numbers', () => {
        let actual = T.cos([0, Math.PI/2, 0.5]);
        let expected = T.fromArray([1, Math.cos(Math.PI/2), Math.cos(0.5)]);
        checkTensor(actual, expected, EPSILON);
    });
    it('should compute the cosine of complex numbers', () => {
        let z = T.fromArray([0, 0.5, 1], [-1, 0.5, 3]);
        let actual = T.cos(z);
        let expected = T.fromArray(
            [1.5430806348152439, 0.9895848833999199, 5.4395809910197643],
            [0, -0.24982639750046154, -8.429751080849945]
        );
        checkTensor(actual, expected, 1e-14);
    });
});

describe('tan()', () => {
    it('should compute the tangent of real numbers', () => {
        let actual = T.tan([0, Math.PI/4, Math.PI/2, -1]);
        let expected = T.fromArray([0, Math.tan(Math.PI/4), Math.tan(Math.PI/2), Math.tan(-1)]);
        checkTensor(actual, expected, EPSILON * 2);
    });
    it('should compute the tangent of complex numbers', () => {
        let z = T.fromArray([0, 0.3, -2], [-1, 0.5, 1.5]);
        let actual = T.tan(z);
        let expected = T.fromArray(
            [0, 0.23840508333812324, 0.080391015310168221],
            [-0.76159415595576485, 0.49619706577350758, 1.0641443991765371]
        );
        checkTensor(actual, expected, 1e-14);
    });
});

describe('cot()', () => {
    it('should compute the cotangent of real numbers', () => {
        let actual = T.cot([0, Math.PI/4, Math.PI/2, -2]);
        let expected = T.fromArray([NaN, 1.0, 6.123233995736766e-17, 0.45765755436028577]);
        checkTensor(actual, expected, EPSILON * 2);
    });
    it('should compute the cotangent of complex numbers', () => {
        let z = T.fromArray([0, 0.5, -2], [-1, 0.4, 8]);
        let actual = T.cot(z);
        let expected = T.fromArray(
            [0, 1.0556222918520826, 1.7033377703904913e-7],
            [1.3130352854993315, -1.1141257265554689, -9.9999985288419813e-1]
        );
        checkTensor(actual, expected, 1e-14);
    });
});

describe('pow2', () => {
    it('should compute the power for real numbers', () => {
        let x = T.fromArray([1, 2, 3.3, -3]);
        let y = T.fromArray([2, -1.2, 4.4, -2]);
        let actual = T.pow(x, y);
        let expected = T.fromArray([1, Math.pow(2, -1.2), Math.pow(3.3, 4.4), 1/9]);
        checkTensor(actual, expected, EPSILON);
    });
    it('should compute the power of real numbers in place', () => {
        let actual = T.pow(T.fromArray([1, 2, 3, 4]), 2, true);
        let expected = T.fromArray([1, 4, 9, 16]);
        checkTensor(actual, expected);
    });
    it('should compute the power for real numbers with possible complex output', () => {
        let x = T.fromArray([2, -2, -2]);
        let y = T.fromArray([10, -6, 3.3]);
        let actual = T.pow(x, y);
        let expected = T.fromArray(
            [1024, 1.5625e-2, -5.7891882368512873],
            [0, 0, -7.968134023406492]);
        checkTensor(actual, expected, 1e-13);
    });
    it('should compute the power for complex numbers', () => {
        let x = T.fromArray(
            [0, 0,   4,  8, 3.2],
            [3, 2, 1.1, -2,   0]
        );
        let y = T.fromArray(
            [2,   0,    0, -2, 2.4],
            [1, 3.3, -4.2,  7,  -1]
        );
        let actual = T.pow(x, y);
        let expected = T.fromArray(
            [-0.85095334231007735,
             -3.6831822518019123e-3,
             2.941872544447163,
             -7.3579410095963541e-2,
             6.4646816470884474],
            [-1.6661950031664541,
             4.2284316598590176e-3,
             0.93476190747432153,
             3.5518314725028179e-2,
             -14.970265377417148]
        );
        checkTensor(actual, expected, 1e-14);
    });
    it('should compute the power of complex numbers in place', () => {
        let x = T.fromArray([1, -1], [-2, 4]);
        let actual = T.pow(x, 2, true);
        let expected = T.fromArray([-3, -15], [-4, -8]);
        checkTensor(actual, expected, EPSILON * 10);
    });
});