import { JasmalEngine } from '../index';
import { ComplexNumber } from '../lib/complexNumber';
import { checkTensor, checkArrayLike } from './testHelper';
const T = JasmalEngine.createInstance();

describe('reshape()', () => {
    it('should reshape a tensor without altering its data', () => {
        let M = T.fromArray(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11]
        );
        let N = T.reshape(M, [2, 3, 2]);
        expect(N.shape).toEqual([2, 3, 2]);
        expect(N.strides).toEqual([6, 2, 1]);
        // reference copy expected
        expect(N.realData).toBe(M.realData);
        expect(N.imagData).toBe(M.imagData);
    });
    it('should accept a nested array as its input', () => {
        let arr = [[0, 1, 2], [3, 4, 5]];
        let actual = T.reshape(arr, [3, 2]);
        let expected = T.fromArray([[0, 1], [2, 3], [4, 5]]);
        checkTensor(actual, expected);
        // should not change arr
        expect(arr).toEqual([[0, 1, 2], [3, 4, 5]]);
    });
    it('should automatic calculate the new shape when there is a -1 in the specified new shape', () => {
        let M = T.ones([3, 4, 6]);
        let N = T.reshape(M, [2, -1, 3]);
        expect(N.shape).toEqual([2, 12, 3]);
        expect(N.strides).toEqual([36, 3, 1]);
    });
    it('should throw if the new shape is invalid', () => {
        let M = T.ones([2, 3, 4]);
        // number of elements changed
        let case1 = () => { T.reshape(M, [2, 3, 2]); };
        // more than one -1
        let case2 = () => { T.reshape(M, [2, -1, 2, -1]); };
        // indivisible -1
        let case3 = () => { T.reshape(M, [-1, 5, 2]); };
        // non integers
        let case4 = () => { T.reshape(M, [1, 0.5, -1]); };
        expect(case1).toThrow();
        expect(case2).toThrow();
        expect(case3).toThrow();
        expect(case4).toThrow();
    });
});

describe('prependAxis()', () => {
    it('should prepend a new axis to a complex number', () => {
        let Y = T.prependAxis(T.complexNumber(-3, 27));
        expect(Y.shape).toEqual([1, 1]);
        expect(Y.strides).toEqual([1, 1]);
        expect(Y.realData[0]).toBe(-3);
        expect(Y.imagData[0]).toBe(27);
    });
    it('should prepend a new axis to an array', () => {
        let arr = [1, 2, 5.5];
        let Y = T.prependAxis(arr);
        expect(Y.shape).toEqual([1, 3]);
        expect(Y.strides).toEqual([3, 1]);
        checkArrayLike(Y.realData, arr);
    });
    it('should prepend a new axis to a tensor', () => {
        let X = T.rand([2, 3, 3]);
        let Y = T.prependAxis(X);
        expect(Y.shape).toEqual([1, 2, 3, 3]);
        expect(Y.strides).toEqual([18, 9, 3, 1]);
        // should be a reference copy
        expect(X === Y).toBe(false);
        expect(Y.realData).toBe(X.realData);
    });
});

describe('appendAxis()', () => {
    it('should append a new axis to a complex number', () => {
        let Y = T.appendAxis(T.complexNumber(1.5, -9));
        expect(Y.shape).toEqual([1, 1]);
        expect(Y.strides).toEqual([1, 1]);
        expect(Y.realData[0]).toBe(1.5);
        expect(Y.imagData[0]).toBe(-9);
    });
    it('should append a new axis to an array', () => {
        let arr = [6, -1, 3];
        let Y = T.appendAxis(arr);
        expect(Y.shape).toEqual([3, 1]);
        expect(Y.strides).toEqual([1, 1]);
        checkArrayLike(Y.realData, arr);
    });
    it('should append a new axis to a tensor', () => {
        let X = T.rand([3, 2, 4]);
        let Y = T.appendAxis(X);
        expect(Y.shape).toEqual([3, 2, 4, 1]);
        expect(Y.strides).toEqual([8, 4, 1, 1]);
        // should be a reference copy
        expect(X === Y).toBe(false);
        expect(Y.realData).toBe(X.realData);
    });
});

describe('tile()', () => {
    it('should repeat a scalar', () => {
        checkTensor(T.tile(1, [2, 3, 4, 2]), T.ones([2, 3, 4, 2]));
    });
    it('should repeat rows', () => {
        let actual = T.tile([1, 1.1, 2.2], [3, 1]);
        let expected = T.fromArray(
            [[1, 1.1, 2.2],
             [1, 1.1, 2.2],
             [1, 1.1, 2.2]]);
        checkTensor(actual, expected);
    });
    it('should repeat columns', () => {
        let actual = T.tile([[-3], [Infinity], [4.2]], [1, 4]);
        let expected = T.fromArray(
            [[-3, -3, -3, -3],
             [Infinity, Infinity, Infinity, Infinity],
             [4.2, 4.2, 4.2, 4.2]]);
        checkTensor(actual, expected);
    });
    it('should tile a matrix', () => {
        let actual = T.tile([[1, -2], [4, -8]], [2, 2]);
        let expected = T.fromArray(
            [[1, -2, 1, -2],
             [4, -8, 4, -8],
             [1, -2, 1, -2],
             [4, -8, 4, -8]]);
        checkTensor(actual, expected);
    });
    it('should tile a matrix into a 3D tensor', () => {
        let actual = T.tile([[1, 3], [4, 2]], [2, 2, 2]);
        let expected = T.fromArray(
            [[[1, 3, 1, 3],
              [4, 2, 4, 2],
              [1, 3, 1, 3],
              [4, 2, 4, 2]],
             [[1, 3, 1, 3],
              [4, 2, 4, 2],
              [1, 3, 1, 3],
              [4, 2, 4, 2]]]);
        checkTensor(actual, expected);
    });
    it('should not change the data type', () => {
        let X = T.fromArray([[1, 2]], [], T.INT32);
        let actual = T.tile(X, [3, 1]);
        let expected = T.fromArray([[1, 2], [1, 2], [1, 2]], [], T.INT32);
        checkTensor(actual, expected);
    });
    it('should tile an empty tensor', () => {
        let X = T.zeros([3, 0, 2]);
        let actual = T.tile(X, [2, 3, 4]);
        let expected = T.zeros([6, 0, 8]);
        checkTensor(actual, expected);
    });
});

describe('concat()', () => {
    it('should concat scalars (not very efficiently)', () => {
        let inputs = [1, -3, new ComplexNumber(1.5, -Math.PI), Infinity];
        let expectedRe = [1, -3, 1.5, Infinity];
        let expectedIm = [0, 0, -Math.PI, 0];
        let actual = T.concat(inputs);
        expect(actual.dtype).toBe(T.FLOAT64);
        expect(actual.hasComplexStorage()).toBeTruthy();
        expect(actual.shape).toEqual([4]);
        checkArrayLike(actual.realData, expectedRe);
        checkArrayLike(actual.imagData, expectedIm);
    });
    it('should concat arrays', () => {
        let inputs = [[1, 2], [3, 4, 5], [6]];
        let expectedRe = [1, 2, 3, 4, 5, 6];
        let actual = T.concat(inputs);
        expect(actual.hasComplexStorage()).toBeFalsy();
        checkArrayLike(actual.realData, expectedRe);
    });

    let A = T.fromArray([[1, 2], [4, 8]]);
    let B = T.fromArray([[-1, -2], [-4, -8]], [[3.2, 3.1], [-0.2, 3]]);
    it('should concat matrices horizontally', () => {
        let expected = T.fromArray(
            [[1, 2, -1, -2],
             [4, 8, -4, -8]],
            [[0, 0, 3.2, 3.1],
             [0, 0, -0.2, 3]]);
        let actual = T.concat([A, B], 1);
        checkTensor(actual, expected);
    });
    it('should concat matrices vertically', () => {
        let expected = T.fromArray(
            [[1, 2], [4, 8], [-1, -2], [-4, -8], [10, 20]],
            [[0, 0], [0, 0], [3.2, 3.1], [-0.2, 3], [0, 0]]);
        let actual = T.concat([A, B, [[10, 20]]], 0);
        checkTensor(actual, expected);
    });
    it('should work when one of the matrix is empty', () => {
        let actual = T.concat([A, T.zeros([2, 0]), [[99], [98]]], 1);
        let expected = T.fromArray(
            [[1, 2, 99],
             [4, 8, 98]]
        );
        checkTensor(actual, expected);
    });
});

describe('permuteAxis()', () => {
    it('should do nothing for 1D arrays', () => {
        let actual = T.permuteAxis([1, 2, 3], [0]);
        let expected = T.fromArray([1, 2, 3]);
        checkTensor(actual, expected);
    });
    it('should transpose a matrix', () => {
        let M = T.fromArray(
            [[1, 2, 3],
             [4, 5, 6]],
            [[-1, -2, -3],
             [-4, -5, -6]]
        );
        let MCopy = M.copy(true);
        let actual = T.permuteAxis(M, [1, 0]);
        let expected = T.fromArray(
            [[1, 4],
             [2, 5],
             [3, 6]],
            [[-1, -4],
             [-2, -5],
             [-3, -6]],
        );
        checkTensor(actual, expected);
        // should not change M
        checkTensor(M, MCopy);
    });
    it('should permute axis of a tensor', () => {
        let M = T.fromArray(
            [[[1, 2], [3, 4], [5, 6]],
             [[7, 8], [9, 10], [11, 12]]]
        );
        let actual = T.permuteAxis(M, [2, 0, 1]);
        let expected = T.fromArray(
            [[[1, 3, 5],
              [7, 9, 11]],
             [[2, 4, 6],
              [8, 10, 12]]]
        );
        checkTensor(actual, expected);
    });
    it('should permute the axes for an empty tensor', () => {
        let X = T.ones([0, 2, 4]);
        let actual = T.permuteAxis(X, [2, 1, 0]);
        let expected = T.ones([4, 2, 0]);
        checkTensor(actual, expected);
    });
});

describe('real()', () => {
    it('should return the real part of a number', () => {
        checkTensor(T.real(3.14), T.fromArray([3.14]));
    });
    it('should return the real part of a complex number', () => {
        checkTensor(T.real(T.complexNumber(2, 3)), T.fromArray([2]));
    });
    it('should return the real part of an array', () => {
        let arr = [[1, 2], [4, 8]];
        let actual = T.real(arr);
        let expected = T.fromArray(arr);
        checkTensor(actual, expected);
    });
    it('should return the real part of a complex tensor', () => {
        let A = T.fromArray([[[1, 3]]], [[[2, 4]]], T.INT32);
        let actual = T.real(A);
        let expected = T.fromArray([[[1, 3]]], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('imag()', () => {
    it('should return the imaginary part of a number', () => {
        checkTensor(T.imag(3.14), T.fromArray([0]));
    });
    it('should return the imaginary part of a complex number', () => {
        checkTensor(T.imag(T.complexNumber(2, 3)), T.fromArray([3]));
    });
    it('should return the imaginary part of an array', () => {
        let arr = [[1, 2, 3], [4, 8, 12]];
        let actual = T.imag(arr);
        let expected = T.zeros([2, 3]);
        checkTensor(actual, expected);
    });
    it('should return the imaginary part of a real tensor', () => {
        let A = T.fromArray([[1], [3], [5]]);
        let actual = T.imag(A);
        let expected = T.fromArray([[0], [0], [0]]);
        checkTensor(actual, expected);
    });
    it('should return the imaginary part of a complex tensor', () => {
        let A = T.fromArray([[[1, 3]]], [[[2, 4]]], T.INT32);
        let actual = T.imag(A);
        let expected = T.fromArray([[[2, 4]]], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('isempty()', () => {
    it('should return true for empty tensors', () => {
        expect(T.isempty(T.zeros([0]))).toBe(true);
        expect(T.isempty(T.zeros([0, 5]))).toBe(true);
        expect(T.isempty(T.zeros([3, 2, 0]))).toBe(true);
    });
    it('should return true for empty arrays', () => {
        expect(T.isempty([])).toBe(true);
        expect(T.isempty([[], [], []])).toBe(true);
        expect(T.isempty(new Int32Array(0))).toBe(true);        
    });
    it('should return false for non-empty tensors', () => {
        expect(T.isempty(T.zeros([1]))).toBe(false);
        expect(T.isempty(T.ones([2, 3]))).toBe(false);
    });
    it('should return false for non-empty arrays', () => {
        expect(T.isempty([1, 2, 3])).toBe(false);
        expect(T.isempty([[1], [2]])).toBe(false);
        expect(T.isempty(new Float64Array([1, 2]))).toBe(false);
    });
    it('should return false for scalars', () => {
        expect(T.isempty(NaN)).toBe(false);
        expect(T.isempty(T.complexNumber(2, 3))).toBe(false);
        expect(T.isempty(0.01)).toBe(false);
    });
});

describe('isreal()', () => {
    it('should return true for a number, a JavaScript array or typed array', () => {
        expect(T.isreal(0.9)).toBe(true);
        expect(T.isreal(NaN)).toBe(true);
        expect(T.isreal(Infinity)).toBe(true);
        expect(T.isreal([0, 1, 2, 3])).toBe(true);
        expect(T.isreal([[1, 2], [3, 4]])).toBe(true);
        expect(T.isreal(new Float64Array([1, 2]))).toBe(true);
        expect(T.isreal(new Uint8Array([1, 2]))).toBe(true);
        expect(T.isreal(new Int32Array([1, 2]))).toBe(true);
    });
    it('should return true if a complex number\'s imaginary part is zero, and false if not', () => {
        expect(T.isreal(T.complexNumber(1, 0))).toBe(true);
        expect(T.isreal(T.complexNumber(0, 2))).toBe(false);        
    });
    it('should return true if a tensor\'s imaginary part is zero, and false if not', () => {
        expect(T.isreal(T.ones([3, 3]).ensureComplexStorage())).toBe(true);
        expect(T.isreal(T.zeros([2, 6]))).toBe(true);
        expect(T.isreal(T.fromArray([1, 2], [0, 1]))).toBe(false);
    });
});

describe('isnan()', () => {
    it('should return a boolean for a scalar input.', () => {
        expect(T.isnan(NaN)).toBe(1);
        expect(T.isnan(0)).toBe(0);
    });
    it('should apply isNaN() to each element and return a logic tensor for a real input', () => {
        let x = T.fromArray([[1, NaN], [Infinity, 2]]);
        let actual = T.isnan(x);
        let expected = T.fromArray([[0, 1], [0, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should apply isNaN() to each element and return a logic tensor for a complex input', () => {
        let x = T.fromArray([[1, NaN], [Infinity, 2]], [[NaN, NaN], [2, NaN]]);
        let actual = T.isnan(x);
        let expected = T.fromArray([[1, 1], [0, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('isinf()', () => {
    it('should return a boolean for a scalar input.', () => {
        expect(T.isinf(Infinity)).toBe(1);
        expect(T.isinf(0)).toBe(0);
    });
    it('should apply isinf() to each element and return a logic tensor for a real input', () => {
        let x = T.fromArray([[1, NaN], [Infinity, -Infinity]]);
        let actual = T.isinf(x);
        let expected = T.fromArray([[0, 0], [1, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should apply isinf() to each element and return a logic tensor for a complex input', () => {
        let x = T.fromArray([[1, Infinity, -1], [-5, Infinity, 0]], [[NaN, 3, -9], [Infinity, Infinity, NaN]]);
        let actual = T.isinf(x);
        let expected = T.fromArray([[0, 1, 0], [1, 1, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('linspace()', () => {
    it('should return 11 evenly spaced points starting from 0 and stopping at 1', () => {
        checkTensor(T.linspace(0, 1, 11), T.fromArray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), 1e-15);
    });
    it('should return the second argument when the number of points is 1', () => {
        checkTensor(T.linspace(2, -2, 1), T.fromArray([-2]));
    });
});

describe('logspace()', () => {
    it('should return 10 logarithmically spaced points from 10^1 to 10^8', () => {
        let actual = T.logspace(1, 8, 10);
        let expectedArr = new Array<number>(10);
        for (let i = 0;i < 10;i++) {
            expectedArr[i] = Math.pow(10, 1 + (8 - 1) / 9 * i);
        }
        let expected = T.fromArray(expectedArr);
        checkTensor(actual, expected, 13, false);
    });
});

describe('find()', () => {
    it('should find non zeros elements for a real tensor', () => {
        let actual = T.find([0, 1, 0, NaN, -2, Infinity, 0, 0]);
        expect(actual).toEqual([1, 3, 4, 5]);
    });
    it('should find non zeros elements for a complex tensor', () => {
        let x = T.fromArray([[0, 0, 1], [2, 0, 0]], [[NaN, 0, 0], [Infinity, Infinity, 0]]);
        let actual = T.find(x);
        expect(actual).toEqual([0, 2, 3, 4]);
    });
});
