import { JasmalEngine } from '../index';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

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

describe('ismember()', () => {
    it('should test membership for two real scalars', () => {
        expect(T.ismember(1, 2)).toBe(false);
        expect(T.ismember(3, 3, true)).toEqual([true, 0]);
        expect(T.ismember(NaN, NaN)).toBe(false);
    });
    it('should check if a scalar exists in a vector', () => {
        expect(T.ismember(1, [2, 0, 1])).toBe(true);
        expect(T.ismember(NaN, [NaN, 2, 5], true)).toEqual([false, -1]);
        expect(T.ismember(T.complexNumber(2, 5), T.fromArray([3, 2, 1], [5, 5, 4]), true)).toEqual([true, 1]);
    });
    let A = T.fromArray([[7, 3, 3], [1, 5, 5]]);
    let B = T.fromArray([[2, 7, 8, 5], [7, 5, 1, 4]]);
    let C = T.fromArray(
        [[-2, 8.5, -2, NaN], [3, 1, 0.2, NaN]],
        [[ 5,  -4,  5,   0], [0, 2,   3, NaN]]);
    let D = T.fromArray(
        [[8, 7, -2, 3, NaN], [3, 0.2, 1, 2, NaN]],
        [[1, 0,  5, 0,   0], [1,   3, 1, 2, NaN]]
    );
    it('should check if each element in a real matrix is in another real matrix', () => {
        let actual = T.ismember(A, B);
        let expected = T.fromArray([[1, 0, 0], [1, 1, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should check if each element in a real matrix is in another real matrix and return indices', () => {
        let [M, I] = T.ismember(A, B, true);
        checkTensor(M, T.fromArray([[1, 0, 0], [1, 1, 1]], [], T.LOGIC));
        checkTensor(I, T.fromArray([[1, -1, -1], [6, 3, 3]], [], T.INT32));
    });
    it('should check if each element in a real matrix is in another complex matrix', () => {
        let actual = T.ismember(A, C);
        let expected = T.fromArray([[0, 1, 1], [0, 0, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should check if each element in a real matrix is in another complex matrix and return indices', () => {
        let [M, I] = T.ismember(A, D, true);
        checkTensor(M, T.fromArray([[1, 1, 1], [0, 0, 0]], [], T.LOGIC));
        checkTensor(I, T.fromArray([[1, 3, 3], [-1, -1, -1]], [], T.INT32));
    });
    it('should check if each element in a complex matrix is in another real matrix', () => {
        let actual = T.ismember(C, A);
        let expected = T.fromArray([[0, 0, 0, 0], [1, 0, 0, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should check if each element in a complex matrix is in another real matrix and return indices', () => {
        let [M, I] = T.ismember(D, A, true);
        checkTensor(M, T.fromArray([[0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], [], T.LOGIC));
        checkTensor(I, T.fromArray([[-1, 0, -1, 1, -1], [-1, -1, -1, -1, -1]], [], T.INT32));
    });
    it('should check if each element in a complex matrix is in another complex matrix', () => {
        let actual = T.ismember(C, D);
        let expected = T.fromArray([[1, 0, 1, 0], [1, 0, 1, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should check if each element in a complex matrix is in another complex matrix and return indices', () => {
        let [M, I] = T.ismember(C, D, true);
        checkTensor(M, T.fromArray([[1, 0, 1, 0], [1, 0, 1, 0]], [], T.LOGIC));
        checkTensor(I, T.fromArray([[2, -1, 2, -1], [3, -1, 6, -1]], [], T.INT32));
    });
    it('should ignore the data type and check by values', () => {
        let x = T.fromArray([1, 0, 1], [], T.LOGIC);
        let y = T.fromArray([2, 3, 0, 1]);
        checkTensor(T.ismember(x, y), T.fromArray([1, 1, 1], [], T.LOGIC));
    });
});

describe('union()', () => {
    let x = [5, 3, 2, 1, 0, 1];
    let y = [[4, 0, 9], [7, 8, 7]];
    let xUy = T.fromArray([0, 1, 2, 3, 4, 5, 7, 8, 9]);
    let z = T.fromArray([3, 2, 1, 1, 1], [0, 4, 8, 7, 8]);
    let w = T.fromArray([3, 3, 1], [0, 0, 7]);
    let zUw = T.fromArray([1, 1, 2, 3], [7, 8, 4, 0]);
    it('should return the union of two real inputs', () => {
        checkTensor(T.union(x, y), xUy);
    });
    it('should return the union of two real inputs and output the indices', () => {
        let [z, ix, iy] = T.union(x, y, true);
        checkTensor(z, xUy);
        expect(ix).toEqual([4, 3, 2, 1, 0]);
        expect(iy).toEqual([0, 3, 4, 2]);
    });
    it('should return the union of two complex inputs', () => {
        checkTensor(T.union(z, w), zUw);
    });
    it('should return the union of two complex inputs and output the indices', () => {
        let [u, iz, iw] = T.union(z, w, true);
        checkTensor(u, zUw);
        expect(iz).toEqual([3, 2, 1, 0]);
        expect(iw).toEqual([]);
    });
    it('should keep all NaNs', () => {
        let a = T.fromArray([NaN, 1, 1], [0, NaN, NaN]);
        let b = T.fromArray([NaN, 1], [0, NaN]);
        let actual = T.union(a, b);
        let expected = T.fromArray(
            [1, 1, 1, NaN, NaN],
            [NaN, NaN, NaN, 0, 0]
        );
        checkTensor(actual, expected);
    });
    it('should coerce the data type correctly', () => {
        let a = T.fromArray([1, 1, 2], [], T.INT32);
        let b = T.fromArray([2, 2, 1]);
        let actual = T.union(a, b);
        let expected = T.fromArray([1, 2]);
        checkTensor(actual, expected);
    });
});

describe('interset()', () => {
    let x = T.fromArray([-2, 5, 5, 0, 3, 3, 3, 3]);
    let y = T.fromArray([[8, 4, 3], [3, 0, 2]]);
    let v = T.fromArray([5, 5, 3]);
    it('should return the intersection of two real inputs', () => {
        let actual = T.intersect(x, y);
        let expected = T.fromArray([0, 3]);
        checkTensor(actual, expected);
    });
    it('should return the intersection of two real inputs and output indices', () => {
        let [z, ix, iy] = T.intersect(x, y, true);
        checkTensor(z, T.fromArray([0, 3]));
        checkTensor(x.get(ix), z);
        checkTensor(y.get(iy), z);
    });
    it('should return the intersection between a short vector and a long vector and output indices', () => {
        let [z, iv, iy] = T.intersect(v.toArray(true), x, true);
        checkTensor(z, T.fromArray([3, 5]));
        checkTensor(v.get(iv), z);
        checkTensor(x.get(iy), z);
    });
    it('should return the intersection between a long vector and a short vector and output indices', () => {
        let [z, ix, iv] = T.intersect(x, v.toArray(true), true);
        checkTensor(z, T.fromArray([3, 5]));
        checkTensor(x.get(ix), z);
        checkTensor(v.get(iv), z);
    });
    it('should return an empty tensor if there is no common elements', () => {
        let [z, ix, iy] = T.intersect([1, 1, NaN], [2, 2, NaN], true);
        checkTensor(z, T.zeros([0]));
        expect(ix).toEqual([]);
        expect(iy).toEqual([]);
    });
    it('should coerce to the correct data type', () => {
        let a = T.fromArray([1, 2, 4, 8], [], T.INT32);
        let b = [8, 8, 8, 1];
        let aIb = T.fromArray([1, 8]);
        checkTensor(T.intersect(a, b), aIb);
        checkTensor(T.intersect(b, a), aIb);
    });
});

describe('setdiff()', () => {
    let x = T.fromArray([9, 9, 9, 8, 8, 3, 3, 2, 2, 2, 2, 1]);
    let y = T.fromArray([1, 1, 1, 1, 1, 1, 3, NaN, Infinity, Infinity]);
    let xDy = T.fromArray([2, 8, 9]);
    let z = T.fromArray([-1, -2, Infinity, 3, 5, 0], [8, 3, 0, 2, 2, 8]);
    let w = T.fromArray([[Infinity, 0], [0, 0]], [[0, 8], [8, 2]]);
    let zDw = T.fromArray([-2, -1, 3, 5], [3, 8, 2, 2]);
    it('should return the set difference for two real inputs', () => {
        checkTensor(T.setdiff(x, y), xDy);
    });
    it('should return the set difference for two real inputs and indices', () => {
        let [d, id] = T.setdiff(x, y, true); 
        checkTensor(d, xDy);
        checkTensor(x.get(id), d);
    });
    it('should return the set difference for two complex inputs', () => {
        checkTensor(T.setdiff(z, w), zDw);
    });
    it('should return the set difference for two complex inputs and indices', () => {
        let [d, id] = T.setdiff(z, w, true); 
        checkTensor(d, zDw);
        checkTensor(z.get(id), d);
    });
    it('should not remove NaNs', () => {
        let actual = T.setdiff([1, 2, NaN, NaN], [NaN, NaN, 1]);
        let expected = T.fromArray([2, NaN, NaN]);
        checkTensor(actual, expected);
    });
    it('should not change the data type', () => {
        let a = T.fromArray([1, 2, 3, 4], [], T.INT32);
        let b = T.fromArray([4, 1], [], T.FLOAT64);
        let actual = T.setdiff(a, b);
        let expected = T.fromArray([2, 3], [], T.INT32);
        checkTensor(actual, expected);
    });
});
