import { JasmalEngine } from '../';
import { checkTensor, maxAbs } from './testHelper';
import { Tensor } from '../lib/tensor';
import { ComplexNumber } from '../lib/complexNumber';
const T = JasmalEngine.createInstance();
T.seed(42);

function validateSVD(A: Tensor, U: Tensor, S: Tensor, V: Tensor, eps: number = 1e-12): void {
    let [m, n] = A.shape;
    let ns = Math.min(m, n);
    // U^T U = I
    let Ut = T.transpose(U);
    if (Ut.hasComplexStorage()) {
        checkTensor(T.matmul(Ut, Ut, T.MM_HERMITIAN), T.eye(ns).ensureComplexStorage(), eps);
    } else {
        checkTensor(T.matmul(Ut, Ut, T.MM_TRANSPOSED), T.eye(ns), eps);
    }
    // V^T V = I
    let Vt = T.transpose(V);
    if (Vt.hasComplexStorage()) {
        checkTensor(T.matmul(Vt, Vt, T.MM_HERMITIAN), T.eye(n).ensureComplexStorage(), eps);
    } else {
        checkTensor(T.matmul(Vt, Vt, T.MM_TRANSPOSED), T.eye(n), eps);
    }
    // USV^T = A
    // Since A may contain large elements, we scale the tolerance factor
    // according to A.
    let tolA = eps * Math.max(maxAbs(A.realData), A.hasComplexStorage() ? maxAbs(A.imagData) : 0);
    if (A.hasComplexStorage()) {
        checkTensor(T.matmul(U, T.matmul(S, V, T.MM_HERMITIAN)), A, tolA);
    } else {
        checkTensor(T.matmul(U, T.matmul(S, V, T.MM_TRANSPOSED)), A, tolA);
    }
}

function validateEVD(A: Tensor, E: Tensor, V: Tensor, hermitian: boolean, eps: number = 1e-12): void {
    let n = A.shape[0];
    let Q: Tensor;
    // Since A may contain large elements, we scale the tolerance factor
    // according to A.
    let tolA = eps * Math.max(maxAbs(A.realData), A.hasComplexStorage() ? maxAbs(A.imagData) : 0);
    if (hermitian) {
        // E should be unitary
        if (E.hasComplexStorage()) {
            Q = <Tensor>T.matmul(E, E, T.MM_HERMITIAN);
            checkTensor(Q, T.eye(n).ensureComplexStorage(), eps);
        } else {
            Q = <Tensor>T.matmul(E, E, T.MM_TRANSPOSED);
            checkTensor(Q, T.eye(n), eps);
        }
        // E V E' = A
        if (E.hasComplexStorage()) {
            Q = <Tensor>T.matmul(T.matmul(E, V), E, T.MM_HERMITIAN);
            checkTensor(Q, A, tolA);
        } else {
            Q = <Tensor>T.matmul(T.matmul(E, V), E, T.MM_TRANSPOSED);
            checkTensor(Q, A, tolA);
        }
    } else {
        // A * E = E * V
        Q = <Tensor>T.sub(T.matmul(E, V), T.matmul(A, E));
        let Z = T.zeros(Q.shape);
        if (Q.hasComplexStorage()) {
            Z.ensureComplexStorage();
        }
        checkTensor(Q, Z, tolA);
    }
}

function validateQR(A: Tensor, Q: Tensor, R: Tensor, P: Tensor, eps: number = 1e-12): void {
    let [m, n] = A.shape;
    // Since A may contain large elements, we scale the tolerance factor
    // according to A.
    let tolA = eps * Math.max(maxAbs(A.realData), A.hasComplexStorage() ? maxAbs(A.imagData) : 0);
    // Q^H Q = I
    let I = <Tensor>T.matmul(Q, Q, T.MM_HERMITIAN);
    if (I.hasComplexStorage()) {
        checkTensor(I, T.eye(m).ensureComplexStorage(), eps);
    } else {
        checkTensor(I, T.eye(m), eps);
    }
    // A P = Q R
    let Z = T.zeros(A.shape);
    if (A.hasComplexStorage()) {
        Z.ensureComplexStorage();
    }
    checkTensor(T.sub(T.matmul(A, P), T.matmul(Q, R)), Z, tolA);
}

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

describe('svd()', () => {
    let A: Tensor, U: Tensor, S: Tensor, V: Tensor;
    let shapes = [[2, 4], [10, 15], [20, 30], [40, 30], [50, 50]];
    it('should perform SVD for a zero matrix', () => {
        [U, S, V] = T.svd(T.zeros([4, 3]));
        checkTensor(U, T.eye(4, 3));
        checkTensor(S, T.zeros([3, 3]));
        checkTensor(V, T.eye(3))
    });
    it('should perform SVD for a simple real matrix', () => {
        A = T.fromArray([[1,2,3],[4,5,6]]);
        let ACopy = A.copy(true);
        [U, S, V] = T.svd(A);
        validateSVD(A, U, S, V);
        checkTensor(A, ACopy); // should not change anything
    });
    for (let i = 5;i < 20;i += 5) {
        it(`should perform SVD for a ${i}x${i} Hilbert matrix`, () => {
            A = T.hilb(i);
            [U, S, V] = T.svd(A);
            validateSVD(A, U, S, V);
        });
    }
    for (let i = 0;i < shapes.length;i++) {
        it(`should perform SVD for a ${shapes[i][0]}x${shapes[i][1]} random matrix`, () => {
            A = T.randn(shapes[i]);
            [U, S, V] = T.svd(A);
            validateSVD(A, U, S, V);    
        });
    }
    it('should perform SVD for a simple complex matrix', () => {
        A = T.fromArray([[2, 1, 1], [1, 1, 0], [1, 0, 1]],
                        [[1, 1, 0], [1, 1, 0], [1, 0, 0]]);
        let ACopy = A.copy(true);        
        [U, S, V] = T.svd(A);
        validateSVD(A, U, S, V);
        checkTensor(A, ACopy); // should not change anything
    });
    for (let i = 0;i < shapes.length;i++) {
        it(`should perform SVD for a ${shapes[i][0]}x${shapes[i][1]} random complex matrix`, () => {
            A = T.complex(T.randn(shapes[i]), T.randn(shapes[i]));
            [U, S, V] = T.svd(A);
            validateSVD(A, U, S, V);
        });
    }
    
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

describe('cond()', () => {
    it('should return one for an identity matrix', () => {
        expect(T.cond(T.eye(4))).toBe(1);
    });
    it('should return a huge number for a singular matrix', () => {
        expect(T.cond(T.ones([5, 8]))).toBeGreaterThan(1e60);
    });
    it('should return the condition number for a Hilbert matrix', () => {
        expect(Math.abs(T.cond(T.hilb(10)) - 1.602502816811318e13) < 1e9).toBeTruthy();
    });
});

describe('pinv()', () => {
    it('should return a zero matrix for a zero matrix', () => {
        checkTensor(T.pinv(T.zeros([3, 5])), T.zeros([5, 3]));
    });
    it('should return an identity matrix for an identity matrix', () => {
        checkTensor(T.pinv(T.eye(3, 4)), T.eye(4, 3));
    });
    it('should return the pseudo inverse of a simple real matrix', () => {
        let A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
        let actual = T.pinv(A);
        let expected = T.fromArray(
            [[-0.94444444444444464,  0.44444444444444442],
             [-0.11111111111111084,  0.11111111111111099],
             [ 0.72222222222222199, -0.22222222222222204]]);
        checkTensor(actual, expected, 1e-14);
    });
    let shapes = [[8, 6], [12, 20], [16, 12], [12, 16], [20, 8], [24, 30], [40, 30]];
    for (let i = 0;i < shapes.length;i++) {
        it(`should compute the pseudo inverse for complex matrix of ${shapes[i][0]} x ${shapes[i][1]}`, () => {
            let A = T.complex(T.randn(shapes[i]), T.randn(shapes[i]));
            let Ainv = T.pinv(A);
            let tol = 1e-15 * Math.max(shapes[i][0], shapes[i][1]);
            // A * Ainv * A = A, Ainv * A * Ainv = Ainv
            checkTensor(T.matmul(T.matmul(A, Ainv), A), A, tol);
            checkTensor(T.matmul(T.matmul(Ainv, A), Ainv), Ainv, tol);
        });
    }
});

describe('eig()', () => {
    var shapes = [[5, 5], [7, 7], [10, 10], [15, 15], [20, 20], [30, 30], [50, 50]];
    // real symmetrical
    it('should perform eigendecomposition for a zero matrix', () => {
        let A = T.zeros([10, 10]);
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, true);
    })
    it('should perform eigendecomposition for a real diagonal matrix', () => {
        let A = T.diag([100, 1, 50, -2, -10000]);
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, true);
    });
    it('should perform eigendecomposition for a real symmetrical matrix', () => {
        let A = T.fromArray(
            [[2, 1, 0],
             [1, 2, 1],
             [0, 1, 2]]
        );
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, true);
    });
    for (let i = 0;i < shapes.length;i++) { 
        it(`should perform eigendecomposition for a ${shapes[i][0]} x ${shapes[i][0]} real symmetrical matrix`, () => {
            let A = T.rand(shapes[i]);
            T.add(A, T.transpose(A), true);
            let [E, V] = T.eig(A);
            validateEVD(A, E, V, true);
        });
    }
    // complex Hermitian
    it('should perform eigendecomposition for a Hermitian matrix', () => {
        let A = T.fromArray(
            [[2, 1, 0],
             [1, 2, 1],
             [0, 1, 2]],
            [[0, -1, 0],
             [1, 0, -1],
             [0, 1, 0]]
        );
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, true);
    });
    for (let i = 0;i < shapes.length;i++) { 
        it(`should perform eigendecomposition for a ${shapes[i][0]} x ${shapes[i][0]} complex Hermitian matrix`, () => {
            let A = T.complex(T.rand(shapes[i]), T.rand(shapes[i]));
            T.add(A, T.hermitian(A), true);
            let [E, V] = T.eig(A);
            validateEVD(A, E, V, true);
        });
    }
    // real general
    it('should perform eigendecomposition for a general real matrix', () => {
        let A = T.fromArray(
            [[ 4,  1, 1, 0],
             [-1,  1, 1, 0],
             [-1, -1, 2, 0],
             [ 0,  0, 0, 3]]);
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, false);
    });
    it('should perform eigendecomposition for a unbalanced real matrix', () => {
        let A = T.fromArray(
            [[    4,    10, 50, 0],
             [ -0.1,     1, 10, 0],
             [-0.01, -0.01,  2, 0],
             [    0,     0,  0, 3]]);
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, false);
    });
    it('should perform eigendecomposition for a Hilbert matrix', () => {
        let A = T.hilb(8);
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, false);
    });
    for (let i = 0;i < shapes.length;i++) { 
        it(`should perform eigendecomposition for a ${shapes[i][0]} x ${shapes[i][0]} general real matrix`, () => {
            let A = T.rand(shapes[i]);
            let [E, V] = T.eig(A);
            validateEVD(A, E, V, false);
        });
    }
    // complex general
    it('should perform eigendecomposition for a general complex matrix', () => {
        let A = T.fromArray(
            [[ 4,  1, 1, 0],
             [-1,  1, 1, 0],
             [-1, -1, 2, 0],
             [ 0,  0, 0, 3]],
            [[-4, -1, -1,  0],
             [ 1, -1, -1,  0],
             [ 1,  1, -2,  0],
             [ 0,  0,  0, -3]]);
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, false);
    });
    it('should perform eigendecomposition for a complex upper Hessenberg matrix', () => {
        let A = T.fromArray(
            [[  1,  10, 100, 1000],
             [0.1,   1,  10,  100],
             [  0, 0.1,   1,   10],
             [  0,   0, 0.1,    1]],
            [[  5,  25, 125,  625],
             [0.2,   5,  25,  125],
             [  0, 0.2,   5,   25],
             [  0,   0, 0.2,    5]]
        );
        let [E, V] = T.eig(A);
        validateEVD(A, E, V, false);
    });
    for (let i = 0;i < shapes.length;i++) { 
        it(`should perform eigendecomposition for a ${shapes[i][0]} x ${shapes[i][0]} general complex matrix`, () => {
            let A = T.complex(T.rand(shapes[i]), T.rand(shapes[i]));
            let [E, V] = T.eig(A);
            validateEVD(A, E, V, false);
        });
    }
});

describe('chol()', () => {
    it('should perform Cholesky decomposition for a simple real symmetric matrix', () => {
        let A = T.fromArray([[3, 1, 1], [1, 2, 1], [1, 1, 3]]);
        let L = T.chol(A);
        checkTensor(T.matmul(L, L, T.MM_TRANSPOSED), A, 1e-15);
    });
    it('should perform Cholesky decomposition for a simple Hermitian matrix', () => {
        let A = T.fromArray(
            [[3, 1, 1],
             [1, 2, 1],
             [1, 1, 3]],
            [[ 0,  1, 0],
             [-1,  0, 1],
             [ 0, -1, 0]]
        );
        let L = T.chol(A);
        checkTensor(T.matmul(L, L, T.MM_HERMITIAN), A, 1e-15);
    });
});

describe('qr()', () => {
    let shapes = [[5, 6], [8, 4], [12, 10], [20, 24], [30, 25]];
    it('should perform QR decomposition for a 10x10 Hilbert matrix', () => {
        let A = T.hilb(10);
        let [Q, R, P] = T.qr(A);
        validateQR(A, Q, R, P);
    });
    it('should perform QR decomposition for a simple real matrix.', () => {
        let A = T.fromArray([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]);
        let [Q, R, P] = T.qr(A);
        validateQR(A, Q, R, P);
    });
    for (let i = 0;i < shapes.length;i++) {
        it(`should perform QR decomposition for a random ${shapes[i][0]} x ${shapes[i][1]} real matrix`, () => {
            let A = T.randn(shapes[i]);
            let [Q, R, P] = T.qr(A);
            validateQR(A, Q, R, P);
        });
    }
    it('should perform QR decomposition for a simple complex matrix.', () => {
        let A = T.fromArray(
            [[1, 2], [1, 2], [1, 2]],
            [[1, 0], [1, 0], [1, 0]]
        );
        let [Q, R, P] = T.qr(A);
        validateQR(A, Q, R, P);
    });
    for (let i = 0;i < shapes.length;i++) {
        it(`should perform QR decomposition for a random ${shapes[i][0]} x ${shapes[i][1]} complex matrix`, () => {
            let A = T.complex(T.randn(shapes[i]), T.randn(shapes[i]));
            let [Q, R, P] = T.qr(A);
            validateQR(A, Q, R, P);
        });
    }
});

describe('linsolve()', () => {
    let shapes = [[3, 3], [6, 4], [16, 16], [20, 12], [30, 30], [50, 40], [80, 30]];

    it('should solve a real linear system with a rank deficient tall A.', () => {
        let A = T.fromArray([[1, 2, 3], [2, 4, 6], [4, 6, 8], [1, 1, 1]]);
        let B = T.ones([4, 2]);
        let actual = T.linsolve(A, B);
        let expected = T.fromArray(
            [[0.13793103448275812,0.13793103448275812 ],
             [0, 0],
             [0.10344827586206913, 0.10344827586206913]]);
        checkTensor(actual, expected, 1e-14);
    });
    for (let i = 0;i < shapes.length;i++) {
        it(`should solve a real linear system with a random ${shapes[i][0]} x ${shapes[i][1]} A`, () => {
            let A = T.randn(shapes[i]);
            let B = T.randn([shapes[i][0], 4]);
            let X = T.linsolve(A, B);
            let lhs = T.matmul(T.matmul(T.transpose(A), A), X);
            let rhs = T.matmul(T.transpose(A), B);
            checkTensor(lhs, <Tensor>rhs, 1e-13 * Math.max(shapes[i][0], shapes[i][1]));
        });
    }
    for (let i = 0;i < shapes.length;i++) {
        it(`should solve a complex linear system with a random ${shapes[i][0]} x ${shapes[i][1]} A`, () => {
            let A = T.complex(T.randn(shapes[i]), T.randn(shapes[i]));
            let B = T.complex(T.randn([shapes[i][0], 4]), T.randn([shapes[i][0], 4]));
            let X = T.linsolve(A, B);
            let lhs = T.matmul(T.matmul(T.hermitian(A), A), X);
            let rhs = T.matmul(T.hermitian(A), B);
            checkTensor(lhs, <Tensor>rhs, 1e-13 * Math.max(shapes[i][0], shapes[i][1]));
        });
    }
});