import { JasmalEngine } from '../';
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
