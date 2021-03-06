import { JasmalEngine } from '../index';
import { checkTensor, checkComplex } from './testHelper';
const T = JasmalEngine.createInstance();
T.seed(32);

describe('polyval()', () => {
    let p1 = [3, 2, -1];
    let p2 = T.fromArray([-2, 3, 4], [0, 1, -1]);
    let A = T.fromArray([[-1, 0], [3, -5]]);
    let B = T.fromArray([[2, -2], [4, -1]], [[0, -1], [1, -2]]);
    it('should evaluate a polynomial with real coefficients for a real scalar input', () => {
        expect(T.polyval(p1, 2)).toEqual(15);    
    });
    it('should evaluate a polynomial with real coefficients for a complex scalar input', () => {
        checkComplex(T.polyval(p1, T.complexNumber(1, -1)), T.complexNumber(1, -8));    
    });
    it('should evaluate a polynomial with complex coefficients for a real scalar input', () => {
        checkComplex(T.polyval(p2, 2), T.complexNumber(2, 1));   
    });
    it('should evaluate a polynomial with complex coefficients for a complex scalar input', () => {
        checkComplex(T.polyval(p2, T.complexNumber(1, -2)), T.complexNumber(15, 2));   
    });
    it('should evaluate a polynomial with real coefficients for a real matrix input', () => {
        let actual = T.polyval(p1, A);
        let expected = T.fromArray([[0, -1], [32, 64]]);
        checkTensor(actual, expected);
    });
    it('should evaluate a polynomial with real coefficients for a complex matrix input', () => {
        let actual = T.polyval(p1, B);
        let expected = T.fromArray([[15, 4], [52, -12]], [[0, 10], [26, 8]]);
        checkTensor(actual, expected);
    });
    it('should evaluate a polynomial with complex coefficients for a real matrix input', () => {
        let actual = T.polyval(p2, A);
        let expected = T.fromArray([[-1, 4], [-5, -61]], [[-2, -1], [2, -6]]);
        checkTensor(actual, expected);
    });
    it('should evaluate a polynomial with complex coefficients for a complex matrix input', () => {
        let actual = T.polyval(p2, B);
        let expected = T.fromArray([[2, -7], [-15, 9]], [[1, -14], [-10, -16]]);
        checkTensor(actual, expected);
    });
    it('should infer the output data type correctly', () => {
        let p = T.fromArray([1, 1], [], T.INT32);
        let x = T.fromArray([0, 1], [], T.LOGIC);
        let actual = T.polyval(p, x);
        let expected = T.fromArray([1, 2], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('polyvalm()', () => {
    let p1 = [1, 2, -1, 3, 5];
    let p2 = T.fromArray([1, 1, 2, -3, 5], [-1, 0, 4, 2, 0]);
    let A1 = T.fromArray([[-1, 2, 0], [-2, 3, 1], [-3, 5, 2]]);
    let A2 = T.fromArray(
        [[-1, -3, 5], [2, -4, -3], [-1, 4, 0]],
        [[3, 3, -2], [0, -1, 0], [4, -1, 4]]);
    it('should behave like polyval() if the input x is a scalar', () => {
        expect(T.polyvalm(p1, 2)).toEqual(39);
    });
    it('should evaluate a matrix polynomial with real coefficients for a real matrix', () => {
        let actual = T.polyvalm(p1, A1);
        let expected = T.fromArray(
            [[ -62, 100,  46],
             [-169, 253, 119],
             [-311, 457, 226]]);
        checkTensor(actual, expected);
    });
    it('should evaluate a matrix polynomial with real coefficients for a complex matrix', () => {
        let actual = T.polyvalm(p1, A2);
        let expected = T.fromArray(
            [[-1051,  338, -2223],
             [  -54, -918,    17],
             [ 1351, -127,  -729]],
            [[-1399, 1182,  -683],
             [   52,  -64,   983],
             [ -565, -411, -1981]]);
        checkTensor(actual, expected);
    });
    it('should evaluate a matrix polynomial with complex coefficients for a real matrix', () => {
        let actual = T.polyvalm(p2, A1);
        let expected = T.fromArray(
            [[ -54,  84,  44],
             [-150, 224, 108],
             [-280, 408, 204]],
            [[  31, -46, -24],
             [  82,-121, -59],
             [ 153,-223,-110]]);
        checkTensor(actual, expected);
    });
    it('should evaluate a matrix polynomial with complex coefficients for a complex matrix', () => {
        let actual = T.polyvalm(p2, A2);
        let expected = T.fromArray(
            [[-2269, 1573, -3364],
             [ -310, -890,  1091],
             [ 1149, -986, -2465]],
            [[ -516,  536,  1001],
             [   59, 1321,   987],
             [-1996, -226, -1451]]);
        checkTensor(actual, expected);
    });
    it('should handle data type correctly', () => {
        let p = T.fromArray([1, 1], [], T.INT32);
        let x = T.fromArray([[1, 1], [2, 2]], [], T.INT32);
        let actual = T.polyvalm(p, x);
        let expected = T.fromArray([[2, 1], [2, 3]], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('polyfit()', () => {
    it('should fit a sine function with a cubic polynomial', () => {
        let x = T.linspace(0, 5, 15);
        let y = T.sin(x);
        let actual = T.polyfit(x, y, 3);
        let expected = T.fromArray([
            0.089382847758163117,
            -0.84554781931970624,
            1.8125575000851049,
            -0.12591214978029500
        ]);
        checkTensor(actual, expected, 12, false);
    });
});

describe('roots()', () => {
    it('should find roots for a linear equation', () => {
        checkTensor(T.roots([2, 3]), T.fromArray([-1.5]));
    });
    it('should handle trailing zeros correctly', () => {
        checkTensor(T.roots([1, 1, 0, 0, 0]), T.fromArray([0, 0, 0, -1]));
    });
    it('should return zeros for x^3 = 0', () => {
        checkTensor(T.roots([1, 0, 0, 0]), T.fromArray([0, 0, 0]));
    });
    it('should return the roots of (x-1)(x-2)...(x-10)=0', () => {
        let actual = T.sort(
            T.roots([1, -55, 1320, -18150, 157773, -902055, 3416930, -8409500, 12753576, -10628640, 3628800]),
            'asc', false);
        let expected = T.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        checkTensor(actual, expected, 1e-8);
    });
    it('should ignore the zeros at the beginning for the real case', () => {
        let p = T.fromArray([0, 0, 1, 0, -1]);
        let actual = T.roots(p);
        let expected = T.fromArray([-1, 1]);
        checkTensor(actual, expected, 1e-14);
    });
    it('should ignore the zeros at the beginning for the complex case', () => {
        let p = T.fromArray([0, 0, 3, 1], [0, 0, 4, 2]);
        let actual = T.roots(p);
        let expected = T.fromArray([-0.44], [-0.08]);
        checkTensor(actual, expected, 1e-14);
    });
    it('should return the roots for a polynomial with real coefficients', () => {
        let actual = T.roots([1, 4, 6, 5, 2]);
        let expected = T.fromArray(
            [-2, -0.5, -0.5, -1],
            [0, 0.8660254037844386, -0.8660254037844386, 0]
        );
        checkTensor(actual, expected, 1e-14);
    });
    it('should return the roots for a polynomial with complex coefficients', () => {
        let actual = T.roots(T.fromArray([1, 4, 2, 3], [1, 0, -2.4, 2]));
        let expected = T.fromArray(
            [-2.1485614876360608, 0.031587776302088186, 0.11697371133397387],
            [0.83069451997990695, 1.7799154861417656, -0.6106100061216736]
        );
        checkTensor(actual, expected, 1e-14);
    });
    it('should throw for illegal inputs', () => {
        // all zeros
        expect(() => T.roots([0, 0, 0])).toThrow();
        // not enough degree
        expect(() => T.roots([0, 0, 0, 0, 2])).toThrow();
        // not a vector
        expect(() => T.roots([[0.1, 2], [-0.3, 4]])).toThrow();
        // contains NaN or Infinity
        expect(() => T.roots([1, 1, 2, NaN])).toThrow();
        expect(() => T.roots([1, 1, Infinity, 0])).toThrow();
    });
});
