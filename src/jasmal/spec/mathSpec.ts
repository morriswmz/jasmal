import { JasmalEngine } from '../index';
import { Tensor } from '../lib/tensor';
import { checkTensor, checkComplex, testUnaryOpInBatch, checkNumber } from './testHelper';
import { EPSILON } from '../lib/constant';
const T = JasmalEngine.createInstance();
const CN = T.complexNumber;

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
    it('should accept typed arrays as inputs (and convert them to FLOAT64 internally)', () => {
        let arr = new Int32Array([-3, 5]);
        let actual = T.abs(arr);
        let expected = T.fromArray([3, 5]);
        checkTensor(actual, expected);
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
    it('should calculate the absolute values of real numbers in place after reference copy', () => {
        let arr = [-1, -2, -3];
        let x = T.fromArray(arr);
        let y = x.copy();
        T.abs(y, true);
        checkTensor(y, T.fromArray([1, 2, 3]));
        checkTensor(x, T.fromArray(arr));
    });
    it('should calculate the absolute values of complex numbers in place after reference copy', () => {
        let arrRe = [-1, 2, 0];
        let arrIm = [1, 2, 3];
        let x = T.fromArray(arrRe, arrIm);
        let y = x.copy();
        T.abs(y, true);
        checkTensor(y, T.fromArray([Math.sqrt(2), Math.sqrt(8), 3], [0, 0, 0]));
        checkTensor(x, T.fromArray(arrRe, arrIm));
    });
    it('should throw if in place calculation is not possible', () => {
        // input is a scalar
        let case1 = () => { T.abs(1, true); };
        // input is an array
        let case2 = () => { T.abs([-1, 1], true); };
        // input is a typed array
        let case3 = () => { T.abs(new Float32Array([1, 2]), true); }
        expect(case1).toThrow();
        expect(case2).toThrow();
        expect(case3).toThrow();
    });
});

describe('sign()', () => {
    it('should evaluate the sign function for a real vector', () => {
        testUnaryOpInBatch(T.sign, [
            [-Infinity, -1, 0],
            [0.2, 1, 0],
            [0, 0, 0],
            [NaN, NaN, 0],
            [-2, -1, 0],
            [Infinity, 1, 0]
        ], true)
    });
    it('should evaluate the sign function for a complex vector', () => {
        testUnaryOpInBatch(T.sign, [
            [CN(1, -2), CN(4.4721359549995793e-1, -8.9442719099991586e-1), 14],
            [CN(-2, 5), CN(-3.7139067635410372e-1, 9.2847669088525941e-1), 14],
            [CN(3, 8), CN(3.5112344158839170e-1, 9.3632917756904455e-1), 14]
        ], false);
    });
});

describe('min2()', () => {
    it('should compute the minimum element-wise', () => {
        let actual = T.min2([[-1], [2]], [-3, -1]);
        let expected = T.fromArray([[-3, -1], [-3, -1]]);
        checkTensor(actual, expected);
    });
    it('should work for complex vectors with zero imaginary parts', () => {
        let x = T.fromArray([1, 2, 3], [0, 0, 0]);
        let actual = T.min2(x, [2, 1, 0]);
        let expected = T.fromArray([1, 1, 0]);
        checkTensor(actual, expected);
    });
});

describe('max2()', () => {
    it('should compute the maximum element-wise', () => {
        let actual = T.max2([[1], [2]], [3, -1]);
        let expected = T.fromArray([[3, 1], [3, 2]]);
        checkTensor(actual, expected);
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

describe('angle()', () => {
    it('should return zero for real numbers', () => {
        testUnaryOpInBatch(T.angle, [
            [0, 0, 0],
            [1, 0, 0],
            [-22, 0, 0]
        ], true);
    });
    it('should return the angle for complex numbers', () => {
        testUnaryOpInBatch(T.angle, [
            [CN(1, 2), Math.atan2(2, 1), 0],
            [CN(-3, 0.5), Math.atan2(0.5, -3), 0],
            [CN(4, -2), Math.atan2(-2, 4), 0]
        ], true);
    });
});

describe('rad2deg()', () => {
    it('should convert radians to degrees', () => {
        testUnaryOpInBatch(T.rad2deg, [
            [0, 0, 0],
            [2, 2 * 180/Math.PI, 0],
            [-1.5, -1.5 * 180/Math.PI, 0]
        ], true);
    });
});

describe('deg2rad()', () => {
    it('should convert degrees to radians', () => {
        testUnaryOpInBatch(T.deg2rad, [
            [0, 0, 0],
            [60, 60/180*Math.PI, 0],
            [-36, -36/180*Math.PI, 0]
        ], true);
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
    it('should compute the squares of real numbers in-place', () => {
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
    it('should return the squares of complex numbers in-place', () => {
        let x = T.fromArray([-0.5, 4], [6, -3]);
        let actual = T.square(x, true);
        let expected = T.fromArray([-35.75, 7], [-6, -24]);
        checkTensor(actual, expected);
    });
});

describe('pow()', () => {
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

describe('realpow()', () => {
    it('should return the same values as Math.pow()', () => {
        let x = [0, 2, 55, -2, -3];
        let y = [1, 4, 3.5, -0.4, -2];
        let expectedArr = new Array<number>(x.length);
        for (let i = 0;i < x.length;i++) {
            expectedArr[i] = Math.pow(x[i], y[i]);
        }
        let actual = T.realpow(x, y);
        let expected = T.fromArray(expectedArr);
        checkTensor(actual, expected);
    });
});

describe('floor()', () => {
    it('should evaluate the floor function for every element', () => {
        testUnaryOpInBatch(T.floor, [
            [9.3, 9, 0],
            [0, 0, 0],
            [0.6, 0, 0],
            [-0.2, -1, 0]
        ], true);
    });
});

describe('ceil()', () => {
    it('should evaluate the ceil function for every element', () => {
        testUnaryOpInBatch(T.ceil, [
            [9.3, 10, 0],
            [0, 0, 0],
            [-0.2, 0, 0]
        ], true);
    });
});

describe('round()', () => {
    it('should evaluate the round function for every element', () => {
        testUnaryOpInBatch(T.round, [
            [9.3, 9, 0],
            [0.55, 1, 0],
            [0, 0, 0],
            [-0.2, 0, 0],
            [-0.51, -1, 0]
        ], true);
    });
});

describe('fix()', () => {
    it('should evaluate the fix function for every element', () => {
        testUnaryOpInBatch(T.fix, [
            [1.55, 1, 0],
            [0, 0, 0],
            [-0.2, 0, 0],
            [-3.8, -3, 0]
        ], true);
    });
});

/**
 * Trigonometry
 */
 
describe('sin()', () => {
    it('should compute the sine of real numbers', () => {
        testUnaryOpInBatch(T.sin, [
            [0, 0, EPSILON],
            [Math.PI/2, Math.sin(Math.PI/2), EPSILON],
            [0.5, Math.sin(0.5), EPSILON]
        ], true);
    });
    it('should compute the sine of complex numbers', () => {
        testUnaryOpInBatch(T.sin, [
            [CN(0.5, -6), CN(9.6707627492897842e1, -1.7701994941200556e2), 15],
            [CN(0.8, 4.6), CN(3.5686445260153995e1, 3.4652193500570938e1), 15]
        ], false);
    });
});

describe('cos()', () => {
    it('should compute the cosine of real numbers', () => {
        testUnaryOpInBatch(T.cos, [
            [0, 1, EPSILON],
            [Math.PI/2, Math.cos(Math.PI/2), EPSILON],
            [0.5, Math.cos(0.5), EPSILON]
        ], true);
    });
    it('should compute the cosine of complex numbers', () => {
        testUnaryOpInBatch(T.cos, [
            [CN(0, -1), CN(1.5430806348152437, 0), 15],
            [CN(0.5, 0.5), CN(9.8958488339991990e-1, -2.4982639750046154e-1), 15],
            [CN(1, 3), CN(5.4395809910197643, -8.4297510808499450), 15]
        ], false);
    });
});

describe('tan()', () => {
    it('should compute the tangent of real numbers', () => {
        testUnaryOpInBatch(T.tan, [
            [0, 0, EPSILON],
            [Math.PI/4, Math.tan(Math.PI/4), EPSILON],
            [Math.PI/2, Math.tan(Math.PI/2), EPSILON],
            [-1, Math.tan(-1), EPSILON]
        ], true);
    });
    it('should compute the tangent of complex numbers', () => {
        testUnaryOpInBatch(T.tan, [
            [CN(0, -1), CN(0, -7.6159415595576485e-1), 14],
            [CN(0.3, 0.5), CN(2.3840508333812321e-1, 4.9619706577350764e-1), 14],
            [CN(-2, 1.5), CN(8.0391015310168193e-2, 1.0641443991765371), 14]
        ], false);
    });
});

describe('cot()', () => {
    it('should compute the cotangent of real numbers', () => {
        testUnaryOpInBatch(T.cot, [
            [0, NaN, EPSILON],
            [Math.PI/4, 1.0, EPSILON],
            [Math.PI/2, 6.123233995736766e-17, EPSILON],
            [-2, 0.45765755436028577, EPSILON]
        ], true);
    });
    it('should compute the cotangent of complex numbers', () => {
        testUnaryOpInBatch(T.cot, [
            [CN(0, -1), CN(0, 1.3130352854993315), 15],
            [CN(0.5, 0.4), CN(1.0556222918520826, -1.1141257265554689), 15],
            [CN(-2, 8), CN(1.7033377701610623e-7, -9.9999985288419813e-1), 14]
        ], false);
    });
});

 describe('asin()', () => {
    it('should compute the inverse sine of a real number', () => {
        expect(T.asin(0.5)).toEqual(Math.asin(0.5));
        checkComplex(T.asin(2), T.complexNumber(Math.PI / 2, -1.3169578969248166), 1e-14);
        checkComplex(T.asin(T.complexNumber(1, 2)), T.complexNumber(4.2707858639247592e-1, 1.5285709194809975), 1e-14);
    });
    it('should compute the inverse sine for a real vector', () => {
        testUnaryOpInBatch(T.asin, [
            [-1, -Math.PI/2, 15],
            [0.05, 5.0020856805770016e-2, 15],
            [14, CN(Math.PI/2, -3.3309265526412517), 15]
        ], false);
    });
    it('should compute the inverse sine for a complex vector', () => {
        testUnaryOpInBatch(T.asin, [
            [CN(8, -8), CN(7.8344506323200080e-1, -3.1191680344383275), 14],
            [CN(2.5, -16), CN(1.5470672855346543e-1, -3.4787030473473450), 14],
            [CN(-2.5, 0.01), CN(-1.5664320046169189, 1.5668096281544301), 13],
            [CN(-4e-3, -8192), CN(-4.8828124636198238e-07, -9.7040605315646431), 8]
        ], false);
    });
});

describe('acos()', () => {
    it('should compute the inverse cosine of a real number', () => {
        expect(T.acos(0.4)).toEqual(Math.acos(0.4));
        checkComplex(T.acos(-2), T.complexNumber(Math.PI, -1.3169578969248166), 1e-14);
        checkComplex(T.acos(T.complexNumber(1, 2)), T.complexNumber(1.1437177404024206, -1.5285709194809980), 1e-14);
    });
    it('should compute the inverse cosine for a real vector', () => {
        testUnaryOpInBatch(T.acos, [
            [  -1,                   Math.PI, 15],
            [0.01,        1.5607961601207294, 15],
            [ 8.2, CN(0, 2.7935424012671657), 13]
        ], false);
    });
    it('should compute the inverse cosine for a complex vector', () => {
        testUnaryOpInBatch(T.acos, [
            [CN(8, 8.2), CN(7.9964725006154480e-1, -3.1317134967322042), 11],
            [CN(2.5, -11.3), CN(1.3538479613188765, 3.1435321043680937), 13],
            [CN(-3, 0.01), CN(3.1380571371772006, -1.7627538031107808), 15],
            [CN(-8.5e-3, -1125), CN(1.5708038823474688, 7.7186856927813130), 10]
        ], false);
    });
});

describe('atan()', () => {
    it('should compute the inverse tangent for a real vector', () => {
        testUnaryOpInBatch(T.atan, [
            [0, 0, 15],
            [0.9, Math.atan(0.9), 15],
            [-248, Math.atan(-248), 15],
            [8.7e9, Math.atan(8.7e9), 15],
            [Infinity, Math.atan(Infinity), 15]
        ], false);
    });
    it('should compute the inverse tangent for a complex vector', () => {
        testUnaryOpInBatch(T.atan, [
            [CN(0.1, 8), CN(1.5692092824500021, 1.2563706123023594e-1), 14],
            [CN(24, -0.8), CN(1.5291998327194452, -1.3849491828050295e-3), 12],
            [CN(1124, -8e3), CN(1.5707791042715509, -1.2258023608365283e-4), 10]
        ], false);
    });
});

describe('acot()', () => {
    it('should compute the inverse cotangent for a real vector', () => {
        testUnaryOpInBatch(T.acot, [
            [0, Math.PI/2, 15],
            [Infinity, 0, 15],
            [0.2, 1.3734007669450159, 15],
            [-9, -1.1065722117389563e-1, 15]
        ], false);
    });
    it('should compute the inverse cotangent for a complex vector', () => {
        testUnaryOpInBatch(T.acot, [
            [CN(-0.5, 3), CN(-6.0311834290051360e-2, -3.3529348145985532e-1), 15],
            [CN(128, -39), CN(7.1487532404470042e-3, 2.1780546567428069e-3), 13],
            [CN(0.2, -0.01), CN(1.3733822741793942, 9.6156453976447061e-3), 13]
        ], false);
    });
});

describe('sinh()', () => {
    it('should compute the hyperbolic sine for a real vector', () => {
        testUnaryOpInBatch(T.sinh, [
            [-4, -2.7289917197127750e1, 15],
            [-0.25, -2.5261231680816831e-1, 15],
            [0, 0, 15],
            [25, 3.6002449668692940e10, 15],
            [Infinity, Infinity, 15]
        ], false);
    });
    it('should compute the hyperbolic sine for a complex vector', () => {
        testUnaryOpInBatch(T.sinh, [
            [CN(-2.2, 3), CN(4.4125006753895839, 6.4462326019085248e-1), 15],
            [CN(8, -0.01), CN(1.4904043024692915e3, -1.4904543200570302e1), 15],
            [CN(0, 2), CN(0,9.0929742682568171e-1), 15]
        ], false);
    });
});

describe('cosh()', () => {
    it('should compute the hyperbolic cosine for a real vector', () => {
        testUnaryOpInBatch(T.cosh, [
            [0, 1, 15],
            [-1, 1.5430806348152437, 15],
            [8, 1.4904791612521781e+3, 15],
            [170, 3.3808969052425046e+73, 15]
        ], false);
    });
    it('should compute the hyperbolic cosine for a complex vector', () => {
        testUnaryOpInBatch(T.cosh, [
            [CN(0.5, -0.5), CN(9.8958488339991990e-1, -2.4982639750046154e-1), 15],
            [CN(42, 0.01), CN(8.6959398924906022e+17, 8.6962297687487410e+15), 15],
            [CN(-0.2, 10), CN(-8.5590897239735397e-1, 1.0953103576443095e-1), 13]
        ], false);
    });
});

describe('tanh()', () => {
    it('should compute the hyperbolic tangent for a real vector', () => {
        testUnaryOpInBatch(T.tanh, [
            [0, 0, 15],
            [1, 7.6159415595576485e-1, 15],
            [-9, -9.9999996954004100e-1, 15],
            [Infinity, 1, 15]
        ], false);
    });
    it('should compute the hyperbolic tangent for a complex vector', () => {
        testUnaryOpInBatch(T.tanh, [
            [CN(1, -1), CN(1.0839233273386946, -2.7175258531951180e-1), 14],
            [CN(8, -0.05), CN(9.9999977605408963e-1, -2.2469536938381050e-8), 13],
            [CN(-1, 20), CN(-1.1717475060430786, 2.4072734799016834e-1), 15]
        ], false);
    });
});

describe('coth()', () => {
    it('should compute the hyperbolic cotangent for a real vector', () => {
        testUnaryOpInBatch(T.coth, [
            [0, Infinity, 15],
            [-0.8, -1.5059407020437063, 14],
            [1.5, 1.1047913929825119, 15],
            [-8, -1.0000002250703748, 15]
        ], false);
    });
    it('should compute the hyperbolic cotangent for a complex vector', () => {
        testUnaryOpInBatch(T.coth, [
            [CN(0.1, -0.1), CN(5.0333776929527225, 4.9667111955975427), 13],
            [CN(8.2, 0.5), CN(1.0000000815149539, -1.2695203687850526e-7), 15],
            [CN(1e-3, -8), CN(1.0216277703960035e-3, -1.4706491370377722e-1), 13]
        ], false);
    });
});

describe('asinh()', () => {
    it('should compute the inverse hyperbolic sine for a real vector', () => {
        testUnaryOpInBatch(T.asinh, [
            [0, 0, 15],
            [0.5, 4.8121182505960347e-1, 15],
            [40, 4.3821828480654981, 15],
            [-16384, -1.0397207709330502e1, 9]
        ], false);
    });
    it('should compute the inverse hyperbolic sine for a complex vector', () => {
        testUnaryOpInBatch(T.asinh, [
            [CN(0.5, -0.3), CN(4.9790294283028769e-1, -2.6955564142495020e-1), 15],
            [CN(144, -1), CN(5.6629966465739843, -6.9441653882313563e-3), 15],
            [CN(-0.1, 6666), CN(-9.4979224301619531, 1.5707813252946414), 15]
        ]);
    });
});

describe('acosh()', () => {
    it('should compute the inverse hyperbolic cosine for real scalar inputs', () => {
        checkNumber(T.acosh(1.04), 2.8190828905414689e-1, 14, false);
        checkComplex(T.acosh(0.5), CN(0, 1.0471975511965976), 15, false);
        checkComplex(T.acosh(-64), CN(4.8519692231746738, 3.1415926535897931), 15, false);
    });
    it('should compute the inverse hyperbolic cosine for a real vector', () => {
        testUnaryOpInBatch(T.acosh, [
            [82330, 1.2011638020815534e+1, 15],
            [5, 2.2924316695611777, 15],
            [1, 0, 15]
        ], false);
    });
    it('should compute the inverse hyperbolic cosine for a complex vector', () => {
        testUnaryOpInBatch(T.acosh, [
            [CN(0, 5), CN(2.3124383412727525, 1.5707963267948966), 15],
            [CN(0.5, -2), CN(1.4657153519472905, -1.3497776911720127), 14],
            [CN(-128, 10), CN(5.5482049442505463, 3.0636236643061188), 15]
        ], false);
    });
});

describe('atanh()', () => {
    it('should compute the inverse hyperbolic tangent for real scalar inputs', () => {
        checkNumber(T.atanh(0.5), 5.4930614433405478e-1, 15, false);
        checkNumber(T.atanh(-1), -Infinity, 15, false);
        checkComplex(T.atanh(5), CN(2.0273255405408219e-1, -1.5707963267948966), 15, false);
    });
    it('should compute the inverse hyperbolic tangent for a real vector', () => {
        testUnaryOpInBatch(T.atanh, [
            [-0.01, -1.0000333353334763e-2, 14],
            [0.9999, 4.9517187756430978, 15],
            [1, Infinity, 15]
        ], false);
    });
    it('should compute the inverse hyperbolic tangent for a complex vector', () => {
        testUnaryOpInBatch(T.atanh, [
            [CN(0.5, -0.5), CN(4.0235947810852513e-1, -5.5357435889704520e-1), 15],
            [CN(-1.02, -0.99), CN(-4.1028948149813871e-1, -1.0233455448596347), 14],
            [CN(42, 3), CN(2.3693027864166154e-2, 1.5691033310076210), 13]
        ], false);
    });
});

describe('acoth()', () => {
    it('should compute the inverse hyperbolic cotangent for real scalar inputs', () => {
        checkNumber(T.acoth(576), 1.7361128553745606e-3, 13, false);
        checkNumber(T.acoth(1), Infinity, 15, false);
        checkComplex(T.acoth(0.5), CN(5.4930614433405489e-1, -1.5707963267948966), 14, false);
    });
    it('should compute the inverse hyperbolic cotangent for a real vector', () => {
        testUnaryOpInBatch(T.acoth, [
            [1.0001, 4.9517687756430018, 12],
            [-32767, -3.0518509485471961e-5, 15],
            [-1, -Infinity, 15]
        ], false);
    });
    it('should compute the inverse hyperbolic cotangent for a complex vector', () => {
        testUnaryOpInBatch(T.acoth, [
            [CN(0.9, 1.01), CN(3.7573064530738270e-1, -5.9044706373941935e-1), 14],
            [CN(-1.5, 23), CN(-2.8182543637043644e-3, -4.3267442009331684e-2), 12],
            [CN(500, -0.01), CN(2.0000026658730805e-3, 4.0000159984639798e-8), 12]
        ], false);
    });
});
