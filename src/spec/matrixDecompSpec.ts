import { JasmalEngine } from '../';
import { checkTensor, maxAbs } from './testHelper';
import { Tensor } from '../lib/tensor';
import { ComplexNumber } from '../lib/complexNumber';
const T = JasmalEngine.createInstance();

function validateSVD(A: Tensor, U: Tensor, S: Tensor, V: Tensor, eps: number = 1e-10): void {
    let [m, n] = A.shape;
    let ns = Math.min(m, n);
    // U^T U = I
    let Ut = T.transpose(U);
    if (Ut.hasComplexStorage()) {
        checkTensor(<Tensor>T.matmul(Ut, Ut, T.MM_HERMITIAN), T.eye(ns).ensureComplexStorage(), eps);
    } else {
        checkTensor(<Tensor>T.matmul(Ut, Ut, T.MM_TRANSPOSED), T.eye(ns), eps);
    }
    // V^T V = I
    let Vt = T.transpose(V);
    if (Vt.hasComplexStorage()) {
        checkTensor(<Tensor>T.matmul(Vt, Vt, T.MM_HERMITIAN), T.eye(n).ensureComplexStorage(), eps);
    } else {
        checkTensor(<Tensor>T.matmul(Vt, Vt, T.MM_TRANSPOSED), T.eye(n), eps);
    }
    // USV^T = A
    // Since A may contain large elements, we scale the tolerance factor
    // according to A.
    let tolA = eps * Math.max(maxAbs(A.realData), A.hasComplexStorage() ? maxAbs(A.imagData) : 0);
    if (A.hasComplexStorage()) {
        checkTensor(<Tensor>T.matmul(U, T.matmul(S, V, T.MM_HERMITIAN)), A, tolA);
    } else {
        checkTensor(<Tensor>T.matmul(U, T.matmul(S, V, T.MM_TRANSPOSED)), A, tolA);
    }
}

function validateEVD(A: Tensor, E: Tensor, V: Tensor, eps: number = 1e-10): void {
    let n = A.shape[0];
    let Q: Tensor;
    // E should be unitary
    if (E.hasComplexStorage()) {
        Q = <Tensor>T.matmul(E, E, T.MM_HERMITIAN);
        checkTensor(Q, T.eye(n).ensureComplexStorage(), eps);
    } else {
        Q = <Tensor>T.matmul(E, E, T.MM_TRANSPOSED);
        checkTensor(Q, T.eye(n), eps);
    }
    // E V E' = A
    // Since A may contain large elements, we scale the tolerance factor
    // according to A.
    let tolA = eps * Math.max(maxAbs(A.realData), A.hasComplexStorage() ? maxAbs(A.imagData) : 0);
    if (E.hasComplexStorage()) {
        Q = <Tensor>T.matmul(T.matmul(E, V), E, T.MM_HERMITIAN);
        checkTensor(Q, A, tolA);
    } else {
        Q = <Tensor>T.matmul(T.matmul(E, V), E, T.MM_TRANSPOSED);
        checkTensor(Q, A, tolA);
    }
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
    T.seed(192);
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

describe('eig()', () => {
    var shapes = [[5, 5], [10, 10], [15, 15], [20, 20]];
    T.seed(201);
    it('should perform eigendecomposition for a real symmetrical matrix', () => {
        let A = T.fromArray(
            [[2, 1, 0],
             [1, 2, 1],
             [0, 1, 2]]
        );
        let [E, V] = T.eig(A);
        validateEVD(A, E, V);
    });
    for (let i = 0;i < shapes.length;i++) { 
        it(`should perform eigendecomposition for a ${shapes[i][0]} x ${shapes[i][0]} real symmetrical matrix`, () => {
            let A = T.rand(shapes[i]);
            T.add(A, T.transpose(A), true);
            let [E, V] = T.eig(A);
            validateEVD(A, E, V);
        });
    }
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
        validateEVD(A, E, V);
    });
    for (let i = 0;i < shapes.length;i++) { 
        it(`should perform eigendecomposition for a ${shapes[i][0]} x ${shapes[i][0]} complex Hermitian matrix`, () => {
            let A = T.complex(T.rand(shapes[i]), T.rand(shapes[i]));
            T.add(A, T.hermitian(A), true);
            let [E, V] = T.eig(A);
            validateEVD(A, E, V);
        });
    }
});