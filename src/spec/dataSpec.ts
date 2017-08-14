import { JasmalEngine } from '..';
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

describe('unique()', () => {
    it('should find unique real numbers', () => {
        let x = T.fromArray([3, 4, 3, 2, 3, -1, NaN, -1, Infinity, NaN]);
        let expectedY = T.fromArray([-1, 2, 3, 4, Infinity, NaN, NaN]);
        let expectedIy = [5, 3, 0, 1, 8, 6, 9];
        let expectedIx = [[5, 7], [3], [0, 2, 4], [1], [8], [6], [9]];
        let [actualY, actualIy, actualIx] = T.unique(x, true);
        checkTensor(actualY, expectedY);
        expect(actualIy).toEqual(expectedIy);
        expect(actualIx).toEqual(expectedIx);
    });
    it('should find unique complex numbers', () => {
        let x = T.fromArray(
            [3, 3, 3, 3, -1, -1, Infinity, NaN,   0],
            [2, 2, 5, 5, -2, -2,        0,   2, NaN]);
        let expectedY = T.fromArray(
            [-1,   0, 3, 3, Infinity, NaN],
            [-2, NaN, 2, 5,        0,   2]);
        let expectedIy = [4, 8, 0, 2, 6, 7];
        let expectedIx = [[4, 5], [8], [0, 1], [2, 3], [6], [7]];
        let [actualY, actualIy, actualIx] = T.unique(x, true);
        checkTensor(actualY, expectedY);
        expect(actualIy).toEqual(expectedIy);
        expect(actualIx).toEqual(expectedIx);
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