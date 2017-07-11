import { JasmalEngine } from '..';
import { Tensor } from '../lib/tensor';
import { checkTensor, checkComplex } from './testHelper';
import { ComplexNumber } from '../lib/complexNumber';
const T = JasmalEngine.createInstance();

describe('min()', () => {
    it('should return the minimum in a vector', () => {
        expect(T.min([6, -1, 3, 0.5, -6.18])).toEqual([-6.18, 4]);
        expect(T.min([6, -Infinity, 3, NaN, -9.2])).toEqual([-Infinity, 1]);
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

describe('var()', () => {
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