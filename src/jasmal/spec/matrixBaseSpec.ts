import { JasmalEngine } from '../';
import { checkTensor, checkNumber, checkComplex } from './testHelper';
import { Tensor } from '../lib/tensor';
import { ComplexNumber } from '../lib/complexNumber';
import { EPSILON } from '../lib/constant';
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

describe('diag()', () => {
    it('should throw when input is not a matrix/vector', () => {
        let case1 = () => T.diag([[[1]]]);
        expect(case1).toThrow();
    });
    let A1 = T.fromArray(
        [[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14]]
    );
    let A2 = T.fromArray(
        [[ 0, -1,  -2],
         [-3, -4,  -5],
         [-5, -6,  -7],
         [-8, -9, -10]],
        [[0, 1,  2],
         [3, 4,  5],
         [5, 6,  7],
         [8, 9, 10]]
    );
    it('should extract the main diagonal of a real matrix', () => {
        let actual = T.diag(A1);
        let expected = T.fromArray([0, 6, 12]);
        checkTensor(actual, expected);
    });
    it('should extract the first upper diagonal of a real matrix', () => {
        let actual = T.diag(A1, 1);
        let expected = T.fromArray([1, 7, 13]);
        checkTensor(actual, expected);
    });
    it('should extract the last upper diagonal of a real matrix', () => {
        let actual = T.diag(A1, A1.shape[1] - 1);
        let expected = T.fromArray([4]);
        checkTensor(actual, expected);
    });
    it('should extract the first lower diagonal of a real matrix', () => {
        let actual = T.diag(A1, -1);
        let expected = T.fromArray([5, 11]);
        checkTensor(actual, expected);
    });
    it('should extract the main diagonal of a complex matrix', () => {
        let actual = T.diag(A2);
        let expected = T.fromArray([0, -4, -7], [0, 4, 7]);
        checkTensor(actual, expected);
    });
    it('should extract the first upper diagonal of a complex matrix', () => {
        let actual = T.diag(A2, 1);
        let expected = T.fromArray([-1, -5], [1, 5]);
        checkTensor(actual, expected);
    });
    it('should extract the first lower diagonal of a complex matrix', () => {
        let actual = T.diag(A2, -1);
        let expected = T.fromArray([-3, -6, -10], [3, 6, 10]);
        checkTensor(actual, expected);
    });
    it('should extract the last lower diagonal of a complex matrix', () => {
        let actual = T.diag(A2, -3);
        let expected = T.fromArray([-8], [8]);
        checkTensor(actual, expected);
    });
    it('should not change the data type when extracting the diagonal elements', () => {
        let X = T.ones([3, 3], T.INT32);
        let x = T.diag(X);
        expect(x.dtype).toEqual(X.dtype);
    });
    it('should throw if k is out of bounds', () => {
        let case1 = () => T.diag(A1, 5);
        let case2 = () => T.diag(A1, -6);
        expect(case1).toThrow();
        expect(case2).toThrow();
    });

    it('should create a real matrix with the specified vector as the main diagonal', () => {
        let actual = T.diag([1, 2, 3]);
        let expected = T.fromArray([[1, 0, 0], [0, 2, 0], [0, 0, 3]]);
        checkTensor(actual, expected);
    });
    it('should create a real matrix with the specified vector as the 2nd upper diagonal', () => {
        let actual = T.diag([1, 2, 3], 2);
        let expected = T.fromArray(
            [[0, 0, 1, 0, 0],
             [0, 0, 0, 2, 0],
             [0, 0, 0, 0, 3],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        );
        checkTensor(actual, expected);
    });
    it('should create a complex matrix with the specified vector as the 2nd lower diagonal', () => {
        let z = T.fromArray([1, 2, 3], [-1, -2, -3]);
        let actual = T.diag(z, -2);
        let expected = T.fromArray(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 2, 0, 0, 0],
             [0, 0, 3, 0, 0]],
            [[ 0,  0,  0, 0, 0],
             [ 0,  0,  0, 0, 0],
             [-1,  0,  0, 0, 0],
             [ 0, -2,  0, 0, 0],
             [ 0,  0, -3, 0, 0]]
        );
        checkTensor(actual, expected);
    });
    it('should not change the data type when creating a diagonal matrix', () => {
        let x = T.fromArray([1, 2, 3], [], T.INT32);
        let X = T.diag(x);
        expect(X.dtype).toEqual(x.dtype);
    });
});

describe('vander()', () => {
    it('should create a real square Vandermonde matrix with default options', () => {
        let actual = T.vander([1, 2, 3, 4]);
        let expected = T.fromArray(
            [[1, 1, 1, 1],
             [8, 4, 2, 1],
             [27, 9, 3, 1],
             [64, 16, 4, 1]]
        );
        checkTensor(actual, expected);
    });
    it('should create a real square Vandermonde matrix with increasing = true', () => {
        let actual = T.vander([1, -2, 5], 4, true);
        let expected = T.fromArray(
            [[1, 1, 1, 1],
             [1, -2, 4, -8],
             [1, 5, 25, 125]]
        );
        checkTensor(actual, expected);
    });
    it('should create a complex square Vandermonde matrix with increasing = false', () => {
        let actual = T.vander(T.fromArray([1, 2, 0.5], [-2, 4, 7]));
        let expected = T.fromArray(
            [[-3, 1, 1],
             [-12, 2, 1],
             [-48.75, 0.5,1]],
            [[-4, -2, 0],
             [16, 4, 0],
             [7, 7, 0]]
        );
        checkTensor(actual, expected);
    });
    it('should create a complex square Vandermonde matrix with increasing = true', () => {
        let actual = T.vander(T.fromArray([0.2, 1.5], [0.8, -2]), 2, true);
        let expected = T.fromArray(
            [[1, 0.2],
             [1, 1.5]],
            [[0, 0.8],
             [0, -2]]
        );
        checkTensor(actual, expected);
    });
    it('should keep the data type', () => {
        let actual = T.vander(T.fromArray([2, 3], [], T.INT32));
        let expected = T.fromArray([[2, 1], [3, 1]], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('matmul()', () => {

    let M = T.fromArray([[8, 1, 6], [3, 5, 7], [4, 9, 2]]);
    let N = T.fromArray([[8, 1, 6], [3, 5, 7], [4, 9, 2]],
                        [[1, 2, 3], [1, 4, 9], [1, 8, 27]]);

    it('should handle real scalar x real scalar', () => {
        checkTensor(T.matmul(3, 5), T.fromArray([[15]]));
    });

    it('should handle real scalar x real vector', () => {
        checkTensor(T.matmul(3, [1, 2, 3]), T.fromArray([[3, 6, 9]]));
    });

    it('should throw for real scalar x matrix larger than 1x1', () => {
        checkTensor(T.matmul(2, T.fromArray([[2]])), T.fromArray([[4]]));
        expect(() => T.matmul(2, [[1, 2], [3, 4]])).toThrow();
    });

    it('should handle real scalar x complex scalar', () => {
        checkTensor(T.matmul(3, T.complexNumber(-1, 2)), T.fromArray([[-3]], [[6]]));
    });

    it('should handle real scalar x complex scalar with the "Hermitian" modifier', () => {
        checkTensor(T.matmul(3, T.complexNumber(-1, 2), T.MM_HERMITIAN), T.fromArray([[-3]], [[-6]]));
    });

    it('should handle complex scalar x complex scalar', () => {
        checkTensor(T.matmul(T.complexNumber(2, 5), T.complexNumber(-1, 2)), T.fromArray([[-12]], [[-1]]));
    });

    it('should handle real x real', () => {
        let actual = T.matmul(M, M);
        let expected = T.fromArray([[91, 67, 67], [67, 91, 67], [67, 67, 91]]);
        checkTensor(actual, expected);
    });

    it('should handle real x real with the "transposed" modifier', () => {
        let actual = T.matmul(M, M, T.MM_TRANSPOSED);
        let expected = T.fromArray([[101, 71, 53], [71, 83, 71], [53, 71, 101]]);
        checkTensor(actual, expected);
    });

    it('should handle real x complex', () => {
        let actual = T.matmul(M, N);
        let expected = T.fromArray([[91, 67, 67], [67, 91, 67], [67, 67, 91]],
                                   [[15, 68, 195], [15, 82, 243], [15, 60, 147]]);
        checkTensor(actual, expected);
    });

    it('should handle real x complex with the "transposed" modifier', () => {
        let actual = T.matmul(M, N, T.MM_TRANSPOSED);
        let expected = T.fromArray([[101, 71, 53], [71, 83, 71], [53, 71, 101]],
                                   [[28, 66, 178], [34, 86, 232], [28, 58, 130]]);
        checkTensor(actual, expected);
    });

    it('should handle real x complex with the "Hermitian" modifier', () => {
        let actual = T.matmul(M, N, T.MM_HERMITIAN);
        let expected = T.fromArray([[101, 71, 53], [71, 83, 71], [53, 71, 101]],
                                   [[-28, -66, -178], [-34, -86, -232], [-28, -58, -130]]);
        checkTensor(actual, expected);
    });

    let A = T.fromArray([[1, 4, 9], [2, -13, 4], [4, 8, 3]],
                        [[8, 11, -17], [-2, -9, 8], [-7, 1, 6]]);
    let B = T.fromArray([[23, 3], [-3, 7], [2, -1]],
                        [[2, -2], [5, 10], [0, 1]]);

    it('should handle complex x complex multiplications', () => {
        let actual = T.matmul(A, B);
        let expected = T.fromArray([[-42, -55], [142, -11], [83, 35]],
                                   [[139, 165], [-64, -207], [-104, 55]]);
        checkTensor(actual, expected);
    });

    let A10 = T.fromArray(
        [[10, 10,  1,  1, 4,  7,  3,  3,  9,  1],
         [ 2, 10,  9,  3, 8,  7,  6,  9,  6,  6],
         [10,  5, 10,  1, 8,  2,  7,  3,  6,  8],
         [ 7,  9,  7,  1, 2,  2,  9, 10, 10, 10],
         [ 1,  2,  8,  9, 5,  5, 10,  4,  3,  2],
         [ 3,  5,  8,  7, 5, 10,  6,  2,  8,  6],
         [ 6, 10,  4,  4, 7,  4,  2,  3,  8,  5],
         [10,  8,  7, 10, 8,  6,  2,  7,  4,  1],
         [10, 10,  2,  1, 8,  3,  3,  5,  6,  4],
         [ 2,  7,  8,  5, 3,  8,  9,  4,  1,  2]]
    );
    let B10 = T.fromArray(
        [[ 3, -4,  5,  5,  2, -4, -1, -1, -2,  3],
         [-1, -2, -4, -3, -1,  5, -2,  4,  3,  0],
         [ 1,  5,  3, -2,  1,  5,  0, -4, -3,  0],
         [-3, -3,  4, -3,  0,  0, -4, -4,  2,  0],
         [ 2,  4,  4, -3, -4,  0, -3, -3, -3, -1],
         [-2,  1, -4,  4, -2, -1,  5,  2, -1,  1],
         [ 2,  5, -1,  1, -3,  5,  5,  3,  2,  1],
         [ 2, -4, -2,  1, -3, -1,  1,  2,  3,  4],
         [ 3,  0,  4, -3, -2, -3, -4,  0, -4,  3],
         [ 0, -3,  0,  4,  0,  3, -2,  1,  5,  2]]
    );
    let C10 = T.fromArray(
        [[51, -35, 32,  14,  -55,  -4, -31,  40, -26,  77],
         [46,  23, 13, -22, -100, 101, -20,  22,  19,  77],
         [82,  30, 99,  20,  -53,  71, -40, -19, -20,  77],
         [84, -29, 35,  22,  -77,  85, -30,  52,  44, 120],
         [19,  58, 51, -26,  -70,  84,   7, -26,   8,  42],
         [21,  31, 49,  -4,  -71,  65, -18, -10, -10,  64],
         [40, -21, 54, -24,  -61,  40, -72,   6,  -5,  63],
         [33, -34, 87, -17,  -60,  23, -61, -37, -13,  72],
         [63, -35, 51,   7,  -62,  21, -55,  23,  -4,  74],
         [11,  46, -7,  -1,  -64, 103,  36,  18,  22,  43]]
    );
    it('should perform matrix multiplication between two 10x10 real matrices', () => {
        let actual = T.matmul(A10, B10);
        checkTensor(actual, C10);
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

describe('norm()', () => {
    // vector norms
    it('should calculate the 0-norm for a real vector', () => {
        expect(T.norm([0, -1, 2, 0, Infinity], 0)).toEqual(3);
    });
    it('should calculate the 0-norm for a complex vector', () => {
        let x = T.fromArray([3.3, 0, 0, 0, 0], [0, 0, -1, 2.2, 0]);
        expect(T.norm(x, 0)).toEqual(3);
    });
    let v1 = T.fromArray([0, 1, 2, -2.2, 7]);
    let v2 = T.fromArray([3, 4, -2, 0.5], [5, 7, -2.2, 4.2]);
    it('should calculate the 1-norm for a real vector', () => {
        checkNumber(T.norm(v1, 1), 12.2, EPSILON);
    });
    it('should calculate the 1-norm for a complex vector', () => {
        checkNumber(T.norm(v2, 1), 21.096080589118870, EPSILON);
    });
    it('should calculate the 2-norm for a real vector', () => {
        checkNumber(T.norm(v1, 2), 7.6707235643060425, EPSILON);
    });
    it('should calculate the 2-norm for a complex vector', () => {
        checkNumber(T.norm(v2, 2), 11.212938954618455, EPSILON);
    });
    it('should calculate the inf-norm for a real vector', () => {
        checkNumber(T.norm(v1, Infinity), 7, EPSILON);
    });
    it('should calculate the inf-norm for a complex vector', () => {
        checkNumber(T.norm(v2, Infinity), 8.0622577482985491, EPSILON);
    });
    it('should calculate the p-norm for a real vector when p = 0.5', () => {
        checkNumber(T.norm(v1, 0.5), 42.813526056081564, 100 * EPSILON);
    });
    it('should calculate the p-norm for a complex vector when p = 0.5', () => {
        checkNumber(T.norm(v2, 0.5), 81.632343580332119, 100 * EPSILON);
    });
    it('should calculate the p-norm for a real vector when p = 4', () => {
        checkNumber(T.norm(v1, 4), 7.0292804929079091, 10 * EPSILON);
    });
    it('should calculate the p-norm for a complex vector when p = 4', () => {
        checkNumber(T.norm(v2, 4), 8.7190042200424944, 10 * EPSILON);
    });
    // matrix norms
    let A = T.fromArray(
        [[ 8, 5,  3, -5, 10],
         [-3, 1,  8,  8,  5],
         [ 4, 0,  7, -9,  1],
         [-6, 9,  2,  0,  0],
         [-9, 3, -6, -6, -8]]
    );
    let ACopy = A.copy(true);
    let B = T.fromArray(
        [[ 1, -1, -2,  9,  1],
         [-1, -9, -6, -8,  9],
         [ 4, 10,  0,  5, -1],
         [ 3, -6, -3, -4, 10],
         [-4, -7, 10, -1, -3]],
        [[5, -6,  2,  0, -2],
         [4, -7,  8, 10, -6],
         [1, 10,  4, -6, -1],
         [4, -6, -6,  8,  0],
         [4, -9, -2,  3, -7]]
    );
    let BCopy = B.copy(true);
    it('should calculate the 1-norm for a real matrix', () => {
        checkNumber(T.norm(A, 1), 30, EPSILON);
    });
    it('should calculate the 1-norm for a complex matrix', () => {
        checkNumber(T.norm(B, 1), 51.513688030250499, 1e-14);
    });
    it('should calculate the 2-norm for a real matrix', () => {
        checkNumber(T.norm(A, 2), 20.367050601556127, 1e-13);
        // should not change A
        checkTensor(A, ACopy);
    });
    it('should calculate the 2-norm for a complex matrix', () => {
        checkNumber(T.norm(B, 2), 31.465936386976061, 1e-13);
        // should not change B
        checkTensor(B, BCopy);
    });
    it('should calculate the inf-norm for a real matrix', () => {
        checkNumber(T.norm(A, Infinity), 32, EPSILON);
    });
    it('should calculate the inf-norm for a complex matrix', () => {
        checkNumber(T.norm(B, Infinity), 49.14776217786671, 10 * EPSILON);
    });
    it('should calculate the Frobenius norm for a real matrix', () => {
        checkNumber(T.norm(A, 'fro'), 29.664793948382652, 10 * EPSILON);
    });
    it('should calculate the Frobenius norm for a complex matrix', () => {
        checkNumber(T.norm(B, 'fro'), 40.459856648287818, 10 * EPSILON);
    });
    // error handling
    it('should throw errors when input is invalid', () => {
        // ndim > 2
        let case1 = () => { T.norm([[[1]]], 2); };
        let case2 = () => { T.norm(T.ones([3, 2, 3]), 1); };
        // invalid p
        let case3 = () => { T.norm(v1, -2.5); };
        let case4 = () => { T.norm(A, 1.5); };
        expect(case1).toThrow();
        expect(case2).toThrow();
        expect(case3).toThrow();
        expect(case4).toThrow();
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
        checkNumber(T.det(A), 1.6340056809999999e-3, 1e-14);
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
        checkComplex(actual, new ComplexNumber(-345, -549), 1e-12);
    });
    it('should return the determinant for a diagonal complex matrix', () => {
        let A = T.fromArray(
            [[2, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, -1]]
        );
        checkNumber(T.det(A), 4, EPSILON);
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
