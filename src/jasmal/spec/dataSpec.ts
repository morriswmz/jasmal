import { JasmalEngine } from '../index';
import { Tensor } from '../lib/tensor';
import { checkTensor, checkComplex } from './testHelper';
import { ComplexNumber } from '../lib/complexNumber';
const T = JasmalEngine.createInstance();


describe('min()', () => {
    let v1 = [6, -1, 3, 0.5, -6.18];
    let v2 = [6, -Infinity, 3, NaN, -9.2];
    let v3 = T.fromArray([[3], [-1.2], [Infinity], [-4]]);
    let A = T.fromArray([[1,4,-1], [2,-3,-2]]);

    it('should return the minimum in a 1D vector', () => {
        expect(T.min(v1)).toEqual([-6.18, 4]);
        expect(T.min(v2)).toEqual([-Infinity, 1]);
    });
    it('should return the minimum in a 1D vector as a tensor when keepDims = true', () => {
        let [x1, i1] = T.min(v1, -1, true);
        checkTensor(x1, T.fromArray([-6.18]));
        checkTensor(i1, T.fromArray([4], [], T.INT32));
        let [x2, i2] = T.min(v2, -1, true);
        checkTensor(x2, T.fromArray([-Infinity]));
        checkTensor(i2, T.fromArray([1], [], T.INT32));
    });
    it('should return the minimum in a 2D column vector', () => {
        expect(T.min(v3)).toEqual([-4, 3]);
    });
    it('should return the minimum in a 2D column vector as a tensor with keepDims = true', () => {
        let [M, I] = T.min(v3, -1, true);
        checkTensor(M, T.fromArray([[-4]]));
        checkTensor(I, T.fromArray([[3]], [], T.INT32));
    });
    it('should return the minimums in a 2D column vector as a 1D vector when axis = 1', () => {
        let [M, I] = T.min(v3, 1);
        checkTensor(M, T.fromArray([3, -1.2, Infinity, -4]));
        checkTensor(I, T.fromArray([0, 0, 0, 0], [], T.INT32));
    });
    it('should return the minimum in a 2D column vector as a 1D vector when axis = 0', () => {
        let [M, I] = T.min(v3, 0);
        checkTensor(M, T.fromArray([-4]));
        checkTensor(I, T.fromArray([3], [], T.INT32));
    });
    it('should return the minimum in a matrix', () => {
        expect(T.min(A)).toEqual([-3, 4]);
    });
    it('should return the minimum in a matrix as a 1x1 tensor when keepDims = true)', () => {
        let [M, I] = T.min(A, -1, true);
        checkTensor(M, T.fromArray([[-3]]));
        checkTensor(I, T.fromArray([[4]], [], T.INT32));
    });
    it('should return the minimums along all columns', () => {
        let [M, I] = T.min(A, 0);
        checkTensor(M, T.fromArray([1, -3, -2]));
        checkTensor(I, T.fromArray([0, 1, 1], [], T.INT32));
    });
    it('should return the minimums along all columns as a 2D tensor when keepDims = true', () => {
        let [M, I] = T.min(A, 0, true);
        checkTensor(M, T.fromArray([[1, -3, -2]]));
        checkTensor(I, T.fromArray([[0, 1, 1]], [], T.INT32));
    });
    it('should return the minimums along all rows', () => {
        let [M, I] = T.min(A, 1);
        checkTensor(M, T.fromArray([-1, -3]));
        checkTensor(I, T.fromArray([2, 1], [], T.INT32));
    });
    it('should return the minimums along all rows as a 2D tensor when keepDims = true', () => {
        let [M, I] = T.min(A, 1, true);
        checkTensor(M, T.fromArray([[-1], [-3]]));
        checkTensor(I, T.fromArray([[2], [1]], [], T.INT32));
    });
    it('should keep other singleton dimensions intact', () => {
        let [M, I] = T.min([[[1, -2]]], 2);
        checkTensor(M, T.fromArray([[-2]]));
        checkTensor(I, T.fromArray([[1]], [], T.INT32));
    });
    it('should work on an empty tensor', () => {
        let [M, I] = T.min(T.zeros([3, 4, 0]), 1, true);
        checkTensor(M, T.zeros([3, 1, 0]));
        checkTensor(I, T.zeros([3, 1, 0], T.INT32));
    });
});

describe('max()', () => {
    it('should return the maximum in a vector', () => {
        expect(T.max([1, 11, -111, NaN, 6])).toEqual([11, 1]);
        expect(T.max([6, -Infinity, 3, Infinity, -9.2])).toEqual([Infinity, 3]);
    });
});

describe('sum()', () => {
    let A = T.fromArray([[1,2,3],[4,5,6]]);
    let C = T.fromArray([[1,2,3],[4,5,6]], [[-1,-2,-3],[-4,-5,-6]]);
    it('should return the sum of all elements for a real matrix', () => {
        let sum = T.sum(A);
        expect(sum).toBe(21);    
    });
    it('should sum along all the columns for a real matrix', () => {
        let sum = <Tensor>T.sum(A, 0);
        checkTensor(sum, T.fromArray([5, 7, 9]));
    });
    it('should sum along all the columns for a real matrix (with keepDims = true)', () => {
        let sum = <Tensor>T.sum(A, 0, true);
        checkTensor(sum, T.fromArray([[5, 7, 9]]));
    });
    it('should sum along all the rows for a real matrix', () => {
        let sum = <Tensor>T.sum(A, 1);
        checkTensor(sum, T.fromArray([6, 15]));
    });
    it('should sum along all the rows for a real matrix (with keepDims = true)', () => {
        let sum = <Tensor>T.sum(A, 1, true);
        checkTensor(sum, T.fromArray([[6], [15]]));
    });
    it('should return the sum of all elements for a complex matrix', () => {
        let sum = T.sum(C);
        checkComplex(sum, new ComplexNumber(21, -21));
    });
    it('should sum along all the columns for a complex matrix', () => {
        let sum = <Tensor>T.sum(C, 0);
        checkTensor(sum, T.fromArray([5, 7, 9], [-5, -7, -9]));
    });
    it('should sum along all the columns for a complex matrix (with keepDims = true)', () => {
        let sum = <Tensor>T.sum(C, 0, true);
        checkTensor(sum, T.fromArray([[5, 7, 9]], [[-5, -7, -9]]));
    });
    it('should sum along all the rows for a complex matrix', () => {
        let sum = <Tensor>T.sum(C, 1);
        checkTensor(sum, T.fromArray([6, 15], [-6, -15]));
    });
    it('should sum along all the rows for a complex matrix (with keepDims = true)', () => {
        let sum = <Tensor>T.sum(C, 1, true);
        checkTensor(sum, T.fromArray([[6], [15]], [[-6], [-15]]));
    });
});

describe('prod()', () => {
    it('should return the product of all the elements in a real vector', () => {
        expect(T.prod([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).toBe(3628800);
    });
    it('should return the product of all the elements in a complex vector', () => {
        let v = T.fromArray([1, 4, 9], [-6, -7.2, 8.3]);
        checkComplex(T.prod(v), new ComplexNumber(-93.84, -606.16), 1e-10);
    });
});

describe('cumsum()', () => {
    let A = T.fromArray([[1, 2, 3], [4, 5, 6]],
                        [[-1, 4, -9], [-2, 4, -6]]);
    it('should return the cumulative sum for a complex vector', () => {
        let expected = Tensor.fromArray(
            [1, 3, 6, 10, 15, 21],
            [-1, 3, -6, -8, -4, -10]
        );
        let actual = T.cumsum(A);
        checkTensor(actual, expected);
    });
    it('should return the cumulative sum along all the columns for a complex matrix', () => {
        let expected = Tensor.fromArray(
            [[1, 2, 3], [5, 7, 9]],
            [[-1, 4, -9], [-3, 8, -15]]
        );
        let actual = T.cumsum(A, 0);
        checkTensor(actual, expected);
    });
    it('should return the cumulative sum along all the rows for a complex matrix', () => {
        let expected = Tensor.fromArray(
            [[1, 3, 6], [4, 9, 15]],
            [[-1, 3, -6], [-2, 2, -4]]
        );
        let actual = T.cumsum(A, 1);
        checkTensor(actual, expected);
    });
    it('should return the cumulative sum along the first dimension for a real 3D array', () => {
        let B = Tensor.fromArray(
            [[[1, 3.5], [2, 4], [-7, -9]],[[-3, 4], [-1, 2], [6.2, 7]]]
        );
        let expected = Tensor.fromArray(
            [[[1, 3.5], [2, 4], [-7, -9]],[[-2, 7.5], [1, 6], [-0.8, -2]]]
        );
        let actual = T.cumsum(B, 0);
        checkTensor(actual, expected, 1e-12);
    });
});

describe('mean()', () => {
    it('should return the same scalar for a scalar input', () => {
        expect(T.mean(42)).toBe(42);
    });
    it('should return the mean of a vector', () => {
        expect(T.mean([1, 2, 3, 4])).toEqual(2.5);
    });
    it('should return the mean of each row for a real matrix', () => {
        let A = T.fromArray(
            [[1, 2, 3, 4, 5, 6],
             [1, 2, 3, 4, 5, Infinity],
             [1, 2, 3, 4, 5, NaN]]
        );
        let expected = T.fromArray([[3.5], [Infinity], [NaN]]);
        let actual = <Tensor>T.mean(A, 1, true);
        checkTensor(actual, expected);
    });
});

describe('mode()', () => {
    it('should return the mode of a real vector', () => {
        expect(T.mode([1, 1, 2, 3, 5, 7])).toBe(1);
    });
    it('should only return the smallest most occurring element', () => {
        expect(T.mode([[2, 3, 4, 2], [3, 2, 1, 3]])).toBe(2);
    });
    it('should ignore the NaNs', () => {
        expect(T.mode([NaN, Infinity, Infinity, NaN, NaN, -1])).toBe(Infinity);
    });
    it('should return NaN if all the elements are NaNs', () => {
        expect(T.mode([[NaN, NaN]])).toBeNaN();
    });
});

describe('var()', () => {
    it('should return zero for a scalar input', () => {
        expect(T.var(999)).toBe(0);
    });
    it('should return the variance of each row for a real matrix', () => {
        let A = T.fromArray(
            [[1, 1, 1, 1],
             [1, 2, 4, 8],
             [1, 3, NaN, 4],
             [Infinity, 2, 4, 5]]
        );
        let expected = T.fromArray([[0], [9.583333333333334], [NaN], [NaN]]);
        let actual = <Tensor>T.var(A, 1, true);
        checkTensor(actual, expected);
    });
});

describe('median()', () => {
    it('should return the median of each row for a real matrix', () => {
        let A = T.fromArray(
            [[3, 1, 8, 1, 5, 2],
             [5, 1, 2, 3, 1, Infinity],
             [NaN, 2, NaN, NaN, NaN, NaN]]
        );
        let expected = T.fromArray([[2.5], [2.5], [NaN]]);
        let actual = <Tensor>T.median(A, 1, true);
        checkTensor(actual, expected);
    });
});

describe('cov()', () => {
    let X = T.fromArray([[1, 2, 3, 5], [2, 5, 6, 7]]);
    let CX = T.fromArray([
        [2.9166666666666665, 3.3333333333333335],
        [3.3333333333333335, 4.6666666666666670]
    ]);
    let Y = T.fromArray([[-3, -4, -5, -6], [1, 2, 3, 6], [2, 4, 6, 8]]);
    let Z = T.fromArray(
        [[1, 2, 4, 8], [2, 4, 8, 16]],
        [[1, 0, 1, 0], [2, 0, 3, 0]]
    );
    let CZ = T.fromArray(
        [[9.9166666666666661, 20],
         [20, 40.583333333333336]],
        [[0, -8.3333333333333329e-2],
         [8.3333333333333329e-02, 0]]
    );
    it('should produce the same result as var() for a real vector', () => {
        let x = [0, 1, 2, 4, 8];
        checkTensor(T.cov(x), T.reshape(T.var(x), [1, 1]));
    });
    it('should compute the covariance matrix for a real matrix input', () => {
        checkTensor(T.cov(X), CX, 15, false);
    });
    it('should compute the covariance matrix for a real matrix input with samples stored as rows', () => {
        checkTensor(T.cov(T.transpose(X), undefined, false), CX, 15, false);
    });
    it('should compute the covariance matrix for a complex matrix input', () => {
        checkTensor(T.cov(Z), CZ, 15, false);
    });
    it('should compute the covariance matrix for a complex matrix input with samples stored as rows', () => {
        checkTensor(T.cov(T.transpose(Z), undefined, false), CZ, 15, false);
    });
    it('should compute the cross-covariance matrix for real matrix inputs', () => {
        let actual = T.cov(X, Y);
        let expected = T.fromArray(
            [[-2.1666666666666665, 3.6666666666666665, 4.3333333333333330],
             [-2.6666666666666665, 4.0000000000000000, 5.3333333333333330]]
        );
        checkTensor(actual, expected, 15, false);
    });
    it('should compute the cross-covariance matrix for complex matrix inputs', () => {
        let actual = T.cov(X, Z);
        let expected = T.fromArray(
            [[5.250, 10.5],
             [5.6666666666666670, 1.1333333333333334e1]],
            [[0.5, 9.1666666666666663e-01],
             [0.66666666666666663, 1]]
        );
        checkTensor(actual, expected, 15, false);
    });
    it('should work for a INT32 input', () => {
        let XI = X.asType(T.INT32, true);
        checkTensor(T.cov(XI), CX);
    });
    it('should throw for invalid inputs', () => {
        // not 1D or 2D inputs
        expect(() => T.cov([[[1]]])).toThrow();
        // inconsistent number of samples
        expect(() => T.cov([[1, 2, 3], [1, 2, 3]], [[1, 2], [3, 4]])).toThrow();
    });
});

describe('corrcoef()', () => {
    let X = T.fromArray([[1, 2, 3, 4], [2, 4, 5, 8]]);
    let CX = T.fromArray(
        [[1, 9.8115578103921230e-1],
         [9.811557810392123e-1, 1]]
    );
    let Y = T.fromArray([[-1, -2, -3, -4], [3, 3, 5, 5], [1, 2, 4, 8]]);
    let Z = T.fromArray(
        [[0, 2, 4, 6], [1, 2, 4, 6]],
        [[0, 1, 1, 2], [2, 3, 3, 3]]
    );
    let CZ = T.fromArray(
        [[1, 9.7475464993298833e-1],
         [9.7475464993298833e-1, 1]],
        [[0, 1.0830607221477648e-1],
         [-1.0830607221477648e-1, 0]]
    );
    it('should produce 1 for a vector input', () => {
        checkTensor(T.corrcoef([1, 2, 4, 19]), T.ones([1, 1]), 1e-15);
    });
    it('should compute the correlation coefficients for a real matrix input', () => {
        checkTensor(T.corrcoef(X), CX, 1e-15);
    });
    it('should compute the correlation coefficients for a real matrix input with samples stored as rows', () => {
        checkTensor(T.corrcoef(T.transpose(X), undefined, false), CX, 1e-15);
    });
    it('should compute the correlation coefficients for a complex matrix input', () => {
        checkTensor(T.corrcoef(Z), CZ, 1e-15);
    });
    it('should compute the correlation coefficients for a complex matrix input with samples stored as rows', () => {
        checkTensor(T.corrcoef(T.transpose(Z), undefined, false), CZ, 1e-15);
    });
    it('should compute the correlation coefficients for two real matrix inputs', () => {
        let actual = T.corrcoef(X, Y);
        let expected = T.fromArray(
            [[-1, 8.9442719099991597e-1, 9.5916630466254393e-1],
             [-9.8115578103921230e-1, 8.0829037686547622e-1, 9.7985506174586101e-1]]
        );
        checkTensor(actual, expected, 1e-15);
    });
    it('should compute the correlation coefficients for complex inputs', () => {
        let actual = T.corrcoef(X, Z);
        let expected = T.fromArray(
            [[9.5346258924559235e-1, 9.6553511822001026e-1],
             [9.3549533144292885e-1, 9.5320624763879636e-1]],
            [[-2.8603877677367773e-1, -1.7038855027411945e-1],
             [-2.9541957835039862e-1, -1.6131182652348861e-1]]
        );
        checkTensor(actual, expected, 1e-15);
    });
});

describe('sort()', () => {
    let A = T.fromArray([[1, 3, NaN, 4], [Infinity, -2, 99, 4]]);
    it('should sort in ascending order', () => {
        let [actualX, actualIndices] = T.sort(A, 'asc', true);
        let expectedX = T.fromArray([-2, 1, 3, 4, 4, 99, Infinity, NaN]);
        let expectedIndices = [5, 0, 1, 3, 7, 6, 4, 2];
        checkTensor(actualX, expectedX);
        expect(actualIndices).toEqual(expectedIndices);
    });
    it('should sort in descending order', () => {
        let [actualX, actualIndices] = T.sort(A, 'desc', true);
        let expectedX = T.fromArray([NaN, Infinity, 99, 4, 4, 3, 1, -2]);
        let expectedIndices = [2, 4, 6, 3, 7, 1, 0, 5];
        checkTensor(actualX, expectedX);
        expect(actualIndices).toEqual(expectedIndices);
    });
});

describe('sortRows()', () => {
    let A = T.fromArray(
        [[     NaN,  -6,   5,  -6],
         [       7, NaN,  -1,  -2],
         [       5,   8,   3,  -6],
         [Infinity, NaN, NaN,  -5],
         [       3,   5,   4, NaN],
         [     NaN,  -6,   5,  -6]]
    );
    let ACopy = A.copy(true);
    it('should sort the rows in ascending order', () => {
        let actual = T.sortRows(A, 'asc', false);
        let expected = T.fromArray(
            [[       3,   5,   4, NaN],
             [       5,   8,   3,  -6],
             [       7, NaN,  -1,  -2],
             [Infinity, NaN, NaN,  -5],
             [     NaN,  -6,   5,  -6],
             [     NaN,  -6,   5,  -6]]
        );
        checkTensor(actual, expected);
        // should not change A
        checkTensor(A, ACopy);
    });
    it('should sort the rows in descending order and return the indices', () => {
        let [actualY, actualI] = T.sortRows(A, 'desc', true);
        let expectedY = T.fromArray(
            [[     NaN,  -6,   5,  -6],
             [     NaN,  -6,   5,  -6],
             [Infinity, NaN, NaN,  -5],
             [       7, NaN,  -1,  -2],
             [       5,   8,   3,  -6],
             [       3,   5,   4, NaN]]
        );
        let expectedI = [0, 5, 3, 1, 2, 4];
        checkTensor(actualY, expectedY);
        expect(actualI).toEqual(expectedI);
        // should not change A
        checkTensor(A, ACopy);
    });
});

describe('hist()', () => {
    it('should produce the histogram', () => {
        let x = T.fromArray([1, 1, 2, 3, 2, 4, 6, 7, 5, 2, 3, 9, 7, 6]);
        let expectedH = T.fromArray([5, 3, 1, 4, 1]);
        let expectedE = T.linspace(1, 9, 6);
        let [actualH, actualE] = T.hist(x, 5);
        checkTensor(actualH, expectedH);
        checkTensor(actualE, expectedE);
    });
    it('should produce the histogram with infinities and NaNs', () => {
        let x = T.fromArray([1, 1, 2, 3, NaN, 2, 4, 6, 7, 5, 2, 3, 9, 7, 6, Infinity, Infinity, -Infinity]);
        let expectedH = T.fromArray([6, 3, 1, 4, 3]);
        let expectedE = T.linspace(1, 9, 6);
        let [actualH, actualE] = T.hist(x, 5);
        checkTensor(actualH, expectedH);
        checkTensor(actualE, expectedE);
    });
    it('should produce the histogram when there is only one finite value', () => {
        let x = T.fromArray([0.8, Infinity, 0.8, NaN, -Infinity, -Infinity, 0.8, NaN]);
        let expectedH = T.fromArray([2, 0, 3, 0, 1]);
        let expectedE = T.fromArray([-2, -1, 0, 1, 2, 3]);
        let [actualH, actualE] = T.hist(x, 5);
        checkTensor(actualH, expectedH);
        checkTensor(actualE, expectedE);
    });
});

describe('fft()/ifft()', () => {
    let v = [1, 0, 2, 0, 3, 0, 2, 0];
    let A = T.fromArray(
        [[1, 2],
         [2, 2],
         [1, 2],
         [3, 2]],
        [[0, 1],
         [2, 2],
         [0, 2],
         [2, 1]],
    );
    let ACopy = A.copy(true);
    it('should compute the FFT of a real vector', () => {
        let actual = T.fft(v);
        let expected = T.fromArray([8, -2, 0, -2, 8, -2, 0, -2]).ensureComplexStorage();
        checkTensor(actual, expected, 1e-15);
    });
    it('should compute the FFT along the columns of a complex matrix', () => {
        let actual = T.fft(A, 0);
        let expected = T.fromArray(
            [[7, 8],
             [0, 1],
             [-3, 0],
             [0, -1]],
            [[4, 6],
             [1, -1],
             [-4, 0],
             [-1, -1]]
        );
        checkTensor(actual, expected, 1e-15);
        // should not change A
        checkTensor(A, ACopy);
    });
    it('ifft(fft(A, 0), 0) should give back A', () => {
        let actual = T.ifft(T.fft(A, 0), 0);
        checkTensor(actual, A, 1e-13);
        // should not change A
        checkTensor(A, ACopy);
    });
    it('ifft(fft(X)) should give back X for a random tensor X', () => {
        let shape = [3, 16, 7];
        let X = T.complex(T.rand(shape), T.rand(shape));
        let XCopy = X.copy(true);
        let actual = T.ifft(T.fft(X, 1), 1);
        checkTensor(actual, X, 1e-13);
        // should not change X
        checkTensor(X, XCopy);
    });
});
