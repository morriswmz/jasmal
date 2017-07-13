import { JasmalEngine } from '../';
import { ComplexNumber } from '../lib/complexNumber';
import { Tensor } from '../lib/tensor';
import { checkTensor, checkArrayLike } from './testHelper';
const T = JasmalEngine.createInstance();

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