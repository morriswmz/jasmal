import { JasmalEngine } from '../';
import { checkTensor } from './testHelper';
import { Tensor } from '../lib/tensor';
import { ComplexNumber } from '../lib/complexNumber';
const T = JasmalEngine.createInstance();

describe('tril()/triu()', () => {
    let C = T.fromArray(
        [[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12],
         [13, 14, 15, 16],
         [17, 18, 19, 20]],
        [[ -1,  -2,  -3,  -4],
         [ -5,  -6,  -7,  -8],
         [ -9, -10, -11, -12],
         [-13, -14, -15, -16],
         [-17, -18, -19, -20]],
        T.INT32);
    it('should return the lower triangular of a complex matrix with k = 1', () => {
        let actual = T.tril(C, 1);
        let expected = T.fromArray(
            [[ 1,  2,  0,  0],
             [ 5,  6,  7,  0],
             [ 9, 10, 11, 12],
             [13, 14, 15, 16],
             [17, 18, 19, 20]],
            [[ -1,  -2,   0,   0],
             [ -5,  -6,  -7,   0],
             [ -9, -10, -11, -12],
             [-13, -14, -15, -16],
             [-17, -18, -19, -20]],
            T.INT32);
        checkTensor(actual, expected);
    });
    it('should return the lower triangular of a complex matrix with k = -1', () => {
        let actual = T.tril(C, -1);
        let expected = T.fromArray(
            [[ 0,  0,  0,  0],
             [ 5,  0,  0,  0],
             [ 9, 10,  0,  0],
             [13, 14, 15,  0],
             [17, 18, 19, 20]],
            [[  0,   0,   0,   0],
             [ -5,   0,   0,   0],
             [ -9, -10,   0,   0],
             [-13, -14, -15,   0],
             [-17, -18, -19, -20]],
            T.INT32);
        checkTensor(actual, expected);
    });
    it('should return the upper triangular of a complex matrix with k = 1', () => {
        let actual = T.triu(C, 1);
        let expected = T.fromArray(
            [[0, 2, 3,  4],
             [0, 0, 7,  8],
             [0, 0, 0, 12],
             [0, 0, 0,  0],
             [0, 0, 0,  0]],
            [[0, -2, -3,  -4],
             [0,  0, -7,  -8],
             [0,  0,  0, -12],
             [0,  0,  0,   0],
             [0,  0,  0,   0]],
            T.INT32);
        checkTensor(actual, expected);
    });
    it('should return the upper triangular of a complex matrix with k = -1', () => {
        let actual = T.triu(C, -1);
        let expected = T.fromArray(
            [[1,  2,  3,  4],
             [5,  6,  7,  8],
             [0, 10, 11, 12],
             [0,  0, 15, 16],
             [0,  0,  0, 20]],
            [[-1,  -2,  -3,  -4],
             [-5,  -6,  -7,  -8],
             [ 0, -10, -11, -12],
             [ 0,   0, -15, -16],
             [ 0,   0,   0, -20]],
            T.INT32);
        checkTensor(actual, expected);
    });
});

describe('isSymmetric()', () => {
    it('should return true for a real symmetric matrix', () => {
        expect(T.isSymmetric([[1, 9], [9, 2]])).toBe(true);
    });
    it('should return true for a real skew symmetric matrix when skew = true', () => {
        expect(T.isSymmetric([[0, 9], [-9, 0]], true)).toBe(true);
    });
    it('should return false for a real non-symmetric matrix', () => {
        expect(T.isSymmetric([[1, 9], [8, 1]])).toBe(false);
    });
    it('should return false for a real matrix that is not skew symmetric when skew = true', () => {
        expect(T.isSymmetric([[1, 9], [9, 2]], true)).toBe(false);
    });
    it('should return true for a complex symmetric matrix', () => {
        let A = T.fromArray([[2, 3], [3, 2]], [[4, -1], [-1, 0]]);
        expect(T.isSymmetric(A)).toBe(true);
    });
    it('should return true for a complex skew symmetric matrix when skew = true', () => {
        let A = T.fromArray([[0, 3], [-3, 0]], [[0, -1], [1, 0]]);
        expect(T.isSymmetric(A, true)).toBe(true);
    });
    it('should return false for a complex matrix that is not symmetric', () => {
        let A = T.fromArray([[2, 3], [3, 2]], [[4, -1], [1, 0]]);
        expect(T.isSymmetric(A)).toBe(false);
    });
    it('should return false for a complex matrix that is not skew symmetric when skew = true', () => {
        let A = T.fromArray([[0, 3], [-3, 0]], [[0, 1], [1, 0]]);
        expect(T.isSymmetric(A, true)).toBe(false);
    });
});

describe('isHermitian()', () => {
    it('should return true for a real symmetric matrix', () => {
        expect(T.isHermitian([[1, 9], [9, 2]])).toBe(true);
    });
    it('should return true for a real skew symmetric matrix when skew = true', () => {
        expect(T.isHermitian([[0, 9], [-9, 0]], true)).toBe(true);
    });
    it('should return false for a real non-symmetric matrix', () => {
        expect(T.isHermitian([[1, 9], [8, 1]])).toBe(false);
    });
    it('should return false for a real matrix that is not skew symmetric when skew = true', () => {
        expect(T.isHermitian([[1, 9], [9, 2]], true)).toBe(false);
    });
    it('should return true for a complex Hermitian matrix', () => {
        let A = T.fromArray([[2, 3], [3, 2]], [[0, -1], [1, 0]]);
        expect(T.isHermitian(A)).toBe(true);
    });
    it('should return true for a complex skew Hermitian matrix when skew = true', () => {
        let A = T.fromArray([[0, 3], [-3, 0]], [[0, -1], [-1, 0]]);
        expect(T.isHermitian(A, true)).toBe(true);
    });
    it('should return false for a complex matrix that is not Hermitian', () => {
        let A = T.fromArray([[2, 3], [3, 2]], [[0, -1], [-1, 0]]);
        expect(T.isHermitian(A)).toBe(false);
    });
    it('should return false for a complex matrix that is not skew Hermitian when skew = true', () => {
        let A = T.fromArray([[0, 3], [-3, 0]], [[0, -1], [1, 0]]);
        expect(T.isHermitian(A, true)).toBe(false);
    });
});

describe('matmul()', () => {

    let M = T.fromArray([[8, 1, 6], [3, 5, 7], [4, 9, 2]]);
    let N = T.fromArray([[8, 1, 6], [3, 5, 7], [4, 9, 2]],
                        [[1, 2, 3], [1, 4, 9], [1, 8, 27]])

    it('should handle real x real', () => {
        let actual = <Tensor>T.matmul(M, M);
        let expected = T.fromArray([[91, 67, 67], [67, 91, 67], [67, 67, 91]]);
        checkTensor(actual, expected);
    });

    it('should handle real x real with the "transposed" modifier', () => {
        let actual = <Tensor>T.matmul(M, M, T.MM_TRANSPOSED);
        let expected = T.fromArray([[101, 71, 53], [71, 83, 71], [53, 71, 101]]);
        checkTensor(actual, expected);
    });

    it('should handle real x complex', () => {
        let actual = <Tensor>T.matmul(M, N);
        let expected = T.fromArray([[91, 67, 67], [67, 91, 67], [67, 67, 91]],
                                   [[15, 68, 195], [15, 82, 243], [15, 60, 147]]);
        checkTensor(actual, expected);
    });

    it('should handle real x complex with the "transposed" modifier', () => {
        let actual = <Tensor>T.matmul(M, N, T.MM_TRANSPOSED);
        let expected = T.fromArray([[101, 71, 53], [71, 83, 71], [53, 71, 101]],
                                   [[28, 66, 178], [34, 86, 232], [28, 58, 130]]);
        checkTensor(actual, expected);
    });

    it('should handle real x complex with the "hermitian" modifier', () => {
        let actual = <Tensor>T.matmul(M, N, T.MM_HERMITIAN);
        let expected = T.fromArray([[101, 71, 53], [71, 83, 71], [53, 71, 101]],
                                   [[-28, -66, -178], [-34, -86, -232], [-28, -58, -130]]);
        checkTensor(actual, expected);
    });

    let A = T.fromArray([[1, 4, 9], [2, -13, 4], [4, 8, 3]],
                        [[8, 11, -17], [-2, -9, 8], [-7, 1, 6]]);
    let B = T.fromArray([[23, 3], [-3, 7], [2, -1]],
                        [[2, -2], [5, 10], [0, 1]]);

    it('should handle complex x complex multiplications', () => {
        let actual = <Tensor>T.matmul(A, B);
        let expected = T.fromArray([[-42, -55], [142, -11], [83, 35]],
                                   [[139, 165], [-64, -207], [-104, 55]]);
        checkTensor(actual, expected);
    });

});

describe('kron()', () => {
    it('should compute the Kronecker product between two real vectors', () => {
        let actual = T.kron([1, 2], [2, 7]);
        let expected = T.fromArray([2, 7, 4, 14]);
        checkTensor(actual, expected);
    });
    it('should compute the Kronecker product between two real matrices', () => {
        let actual = T.kron([[1, 2], [3, 4]], [[2, 1], [4, 9], [-1, 1]]);
        let expected = T.fromArray(
            [[2, 1, 4, 2],
             [4, 9, 8, 18],
             [-1, 1, -2, 2],
             [6, 3, 8, 4],
             [12, 27, 16, 36],
             [-3, 3, -4, 4]]
        );
        checkTensor(actual, expected);
    });
});

describe('inv()', () => {
    it('should computes the inverse for real matrices', () => {
        let A = T.fromArray([[1,2], [3,4]]);
        let ACopy = A.copy(true);
        let actual = <Tensor>T.matmul(A, T.inv(A));
        let expected = T.eye(2);
        checkTensor(actual, expected, 1e-12);
        checkTensor(A, ACopy); // should not change anything
    });
    it('should computes the inverse for complex matrices', () => {
        let A = T.fromArray([[1,2], [3,4]], [[-1, 0], [0, -1]]);
        let ACopy = A.copy(true);        
        let actual = <Tensor>T.matmul(A, T.inv(A));
        let expected = T.eye(2).ensureComplexStorage();
        checkTensor(actual, expected, 1e-12);
        checkTensor(A, ACopy); // should not change anything
    });
});

describe('det()', () => {
    it('should return zero for a zero matrix', () => {
        expect(T.det(T.zeros([3,3]))).toBe(0);
    });
    it('should return one for an identity matrix', () => {
        expect(T.det(T.eye(3))).toBe(1);
    });
    it('should return the determinant for a real matrix', () => {
        let A = T.fromArray(
            [[0.5529, 0.2107, 0.1157],
             [0.2050, 0.3213, 0.1193],
             [0.0322, 0.1498, 0.0621]]);
        expect(Math.abs(<number>T.det(A) - 1.6340056809999999e-3) < 1e-10).toBeTruthy();
    });
    it('should return the determinant for a complex matrix', () => {
        let A = T.fromArray(
            [[1, 2, 1],
             [-2, 9.5, 2],
             [3, 0, 3]],
            [[-3, 1, -2],
             [-5, 4.5, 2],
             [9, 7, -13]]);
        let actual = <ComplexNumber>T.det(A);
        expect(actual.re).toBeCloseTo(-345, 1e-12);
        expect(actual.im).toBeCloseTo(-549, 1e-12);
    });
});

describe('rank()', () => {
    it('should return 0 for a zero matrix', () => {
        expect(T.rank(T.zeros([5, 8]))).toBe(0);
    });
    it('should return full rank for an identity matrix', () => {
        expect(T.rank(T.eye(10))).toBe(10);
    });
    it('should return the rank for a real matrix', () => {
        expect(T.rank(T.fromArray([[1, 1, 0],[2, 2, 0],[3, 3, 3]]))).toBe(2);
    });
    it('should return full rank for a full rank complex matrix', () => {
        let C = T.fromArray([[1, 4, 2], [3, 5, 8], [9, 11, 17]],
                            [[-1, 2, 5], [-9, 2, 7], [2, 2, 3]]);
        expect(T.rank(C)).toBe(3);
    });
});
