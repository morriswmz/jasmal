import { JasmalEngine } from '..';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

let A = T.fromArray([[1.1, 2, 3], [-2, 5, 7.2]]);
let B = T.fromArray([[1.1, -1, 5], [0, 5, 8.2]]);
let C = T.fromArray([[2, 2, 3], [-5, 6, 8]], [[-1, 0, -1], [-2, 7, 8]]);
let D = T.fromArray([[1, 2, 4], [-5, 4, 8]], [[-1, 2, -1], [-2, 3, 8]]);

describe('eq()', () => {
    it('should test equality between a two real matrices', () => {
        let actual = T.eq(A, B);
        let expected = T.fromArray([[1, 0, 0], [0, 1, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should test equality between a real matrix and a complex matrix', () => {
        let actual = T.eq(A, C);
        let expected = T.fromArray([[0, 1, 0], [0, 0, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should test equality between two complex matrices', () => {
        let actual = T.eq(C, D);
        let expected = T.fromArray([[0, 0, 0], [1, 0, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('neq()', () => {
    it('should test non-equality between a two real matrices', () => {
        let actual = T.neq(A, B);
        let expected = T.fromArray([[0, 1, 1], [1, 0, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should test non-equality between a real matrix and a complex matrix', () => {
        let actual = T.neq(A, C);
        let expected = T.fromArray([[1, 0, 1], [1, 1, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
    it('should test non-equality between two complex matrices', () => {
        let actual = T.neq(C, D);
        let expected = T.fromArray([[1, 1, 1], [0, 1, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('lt()', () => {
    it('should test for "less than" between two real matrices', () => {
        let actual = T.lt(A, B);
        let expected = T.fromArray([[0, 0, 1], [1, 0, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('le()', () => {
    it('should test "less than or equal" between two real matrices', () => {
        let actual = T.le(A, B);
        let expected = T.fromArray([[1, 0, 1], [1, 1, 1]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('gt()', () => {
    it('should test for "greater than" between two real matrices', () => {
        let actual = T.gt(A, B);
        let expected = T.fromArray([[0, 1, 0], [0, 0, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('ge()', () => {
    it('should test "greater than or equal" between two real matrices', () => {
        let actual = T.ge(A, B);
        let expected = T.fromArray([[1, 1, 0], [0, 1, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

let v1 = T.fromArray([0, 0, 1, 1], [], T.LOGIC);
let v2 = T.fromArray([0, 1, 0, 1], [], T.LOGIC);

describe('and()', () => {
    it('should evaluate the logic and between two logic vectors', () => {
        let actual = T.and(v1, v2);
        let expected = T.fromArray([0, 0, 0, 1], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('or()', () => {
    it('should evaluate the logic or between two logic vectors', () => {
        let actual = T.or(v1, v2);
        let expected = T.fromArray([0, 1, 1, 1], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('xor()', () => {
    it('should evaluate the logic xor between two logic vectors', () => {
        let actual = T.xor(v1, v2);
        let expected = T.fromArray([0, 1, 1, 0], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('not()', () => {
    it('should evaluate the logic not for a logic vector', () => {
        let actual = T.not(v1);
        let expected = T.fromArray([1, 1, 0, 0], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('all()', () => {
    it('should return true for a logic vector with no elements being false', () => {
        let l = T.fromArray([1, 1, 1, 1, 1], [], T.LOGIC);
        expect(T.all(l)).toBe(true);
    });
    it('should return true for a real vector with no non-zero elements', () => {
        expect(T.all([-2.2, NaN, Infinity, 3.4, 999])).toBe(true);
    });
    it('should return false for a real vector with one zero element', () => {
        expect(T.all([-2.2, NaN, Infinity, 0, 999])).toBe(false);
    });
    it('should return true for a complex matrix with no non-zero elements', () => {
        let C = T.fromArray([[0.5, -3], [4, 2]], [[6, 8], [-2.2, 7]]);
        expect(T.all(C)).toBe(true);
    });
    it('should return false for a complex matrix with one zero element', () => {
        let C = T.fromArray([[0, -3], [4, 0]], [[6, 0], [-2, 0]], T.INT32);
        expect(T.all(C)).toBe(false);
    });
});

describe('any()', () => {
    it('should return true for a logic vector with at least one element being true', () => {
        let l = T.fromArray([0, 0, 0, 1, 1], [], T.LOGIC);
        expect(T.any(l)).toBe(true);
    });
    it('should return false for a logic vector with all the elements being false', () => {
        let l = T.fromArray([0, 0, 0, 0, 0], [], T.LOGIC);
        expect(T.any(l)).toBe(false);
    });
    it('should return true for a real vector with at least one non-zero element', () => {
        expect(T.any([0, NaN, 2, 0])).toBe(true);
    });
    it('should return true for a complex matrix with at least one non-zero element', () => {
        let C = T.fromArray([[0, 0], [0, 0]], [[0, 0], [-0.1, 0]]);
        expect(T.any(C)).toBe(true);
    });
    it('should return false for a zero complex matrix', () => {
        let C = T.zeros([3, 4], T.INT32).ensureComplexStorage();
        expect(T.any(C)).toBe(false);
    });
});
