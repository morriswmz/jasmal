import { JasmalEngine } from '..';
import { Tensor } from '../lib/tensor';
import { checkTensor, checkComplex } from './testHelper';
import { ComplexNumber } from '../lib/complexNumber';
const T = JasmalEngine.createInstance();

describe('sub()', () => {

    it('should be able to handle real scalar - real scalar', () => {
        expect(T.sub(1, 9)).toEqual(-8);
    });
    it('should be able to handle real scalar - complex scalar, or complex scalar - real scalar', () => {
        checkComplex(T.sub(1, T.complexNumber(2, 3)), T.complexNumber(-1, -3));
        checkComplex(T.sub(T.complexNumber(2, 3), 1), T.complexNumber(1, 3));
    });
    it('should be able to handle complex scalar - complex scalar', () => {
        checkComplex(T.sub(T.complexNumber(2, 3), T.complexNumber(4, -9)), T.complexNumber(-2, 12));
    });
    it('should return a tensor if any of the input is an array/tensor', () => {
        checkTensor(T.sub(1, [9]), T.fromArray([-8]));
    });

    let A = T.fromArray([[1, 3], [4, 2]]);
    let B = T.fromArray([[1, 2], [3, 4]], [[-4, -3], [-2, -1]]);
    let C = T.fromArray([[2, 6], [7, 1]]);
    let D = T.fromArray([[0, 2], [4, 6]], [[-9, -7], [-3, 1]]);

    it('should be able to handle real scalar - real matrix', () => {
        let actual = T.sub(10, A);
        let expected = T.fromArray([[9, 7], [6, 8]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle complex scalar - real matrix', () => {
        let actual = T.sub(new ComplexNumber(10, -10), A);
        let expected = T.fromArray([[9, 7], [6, 8]], [[-10, -10], [-10, -10]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle real scalar - complex matrix', () => {
        let actual = T.sub(10, B);
        let expected = T.fromArray([[9, 8], [7, 6]], [[4, 3], [2, 1]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle complex scalar - complex matrix', () => {
        let actual = T.sub(new ComplexNumber(10, -10), B);
        let expected = T.fromArray([[9, 8], [7, 6]], [[-6, -7], [-8, -9]]);
        checkTensor(actual, expected);
    });

    it('should be able to handle real matrix - real scalar', () => {
        let actual = T.sub(A, 10);
        let expected = T.fromArray([[-9, -7], [-6, -8]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - real scalar', () => {
        let actual = T.sub(A, new ComplexNumber(10, -10));
        let expected = T.fromArray([[-9, -7], [-6, -8]], [[10, 10], [10, 10]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle real matrix - complex scalar', () => {
        let actual = T.sub(B, 10);
        let expected = T.fromArray([[-9, -8], [-7, -6]], [[-4, -3], [-2, -1]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - complex scalar', () => {
        let actual = T.sub(B, new ComplexNumber(10, -10));
        let expected = T.fromArray([[-9, -8], [-7, -6]], [[6, 7], [8, 9]]);
        checkTensor(actual, expected);
    });

    it('should be able to handle real matrix - real scalar with inPlace = true', () => {
        let A1 = A.copy(true);
        let actual = T.sub(A1, 10, true);
        let expected = T.fromArray([[-9, -7], [-6, -8]]);
        expect(actual === A1).toBeTruthy();
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - real scalar with inPlace = true', () => {
        let A1 = A.copy(true);
        let actual = T.sub(A1, new ComplexNumber(10, -10), true);
        let expected = T.fromArray([[-9, -7], [-6, -8]], [[10, 10], [10, 10]]);
        expect(actual === A1).toBeTruthy();
        checkTensor(actual, expected);
    });
    it('should be able to handle real matrix - complex scalar with inPlace = true', () => {
        let B1 = B.copy(true);
        let actual = T.sub(B1, 10, true);
        let expected = T.fromArray([[-9, -8], [-7, -6]], [[-4, -3], [-2, -1]]);
        expect(actual === B1).toBeTruthy();
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - complex scalar with inPlace = true', () => {
        let B1 = B.copy(true);        
        let actual = T.sub(B1, new ComplexNumber(10, -10), true);
        let expected = T.fromArray([[-9, -8], [-7, -6]], [[6, 7], [8, 9]]);
        expect(actual === B1).toBeTruthy();        
        checkTensor(actual, expected);
    });

    it('should be able to handle real matrix - real matrix', () => {
        let actual = T.sub(A, C);
        let expected = T.fromArray([[-1, -3], [-3, 1]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - real matrix', () => {
        let actual = T.sub(B, A);
        let expected = T.fromArray([[0, -1], [-1, 2]], [[-4, -3], [-2, -1]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle real matrix - complex matrix', () => {
        let actual = T.sub(A, B);
        let expected = T.fromArray([[0, 1], [1, -2]], [[4, 3], [2, 1]]);
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - complex matrix', () => {
        let actual = T.sub(B, D);
        let expected = T.fromArray([[1, 0], [-1, -2]], [[5, 4], [1, -2]]);
        checkTensor(actual, expected);
    });

    it('should be able to handle real matrix - real matrix with inPlace = true', () => {
        let A1 = A.copy(true);
        let actual = T.sub(A1, C, true);
        let expected = T.fromArray([[-1, -3], [-3, 1]]);
        expect(actual === A1).toBeTruthy()
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - real matrix with inPlace = true', () => {
        let B1 = B.copy(true);
        let actual = T.sub(B1, A, true);
        let expected = T.fromArray([[0, -1], [-1, 2]], [[-4, -3], [-2, -1]]);
        expect(actual === B1).toBeTruthy();
        checkTensor(actual, expected);
    });
    it('should be able to handle real matrix - complex matrix with inPlace = true', () => {
        let A1 = A.copy(true);        
        let actual = T.sub(A1, B, true);
        let expected = T.fromArray([[0, 1], [1, -2]], [[4, 3], [2, 1]]);
        expect(actual === A1).toBeTruthy()        
        checkTensor(actual, expected);
    });
    it('should be able to handle complex matrix - complex matrix with inPlace = true', () => {
        let B1 = B.copy(true);        
        let actual = T.sub(B1, D, true);
        let expected = T.fromArray([[1, 0], [-1, -2]], [[5, 4], [1, -2]]);
        expect(actual === B1).toBeTruthy();        
        checkTensor(actual, expected);
    });

    it('should handle real column vector - real row vector via broadcasting', () => {
        let actual = T.sub(T.fromArray([[1], [2]]), T.fromArray([3, 4]));
        let expected = T.fromArray([[-2, -3], [-1, -2]]);
        checkTensor(actual, expected);
    });

    it('should throw when in-place operation is not possible', () => {
        // first operand is not a tensor
        let case1 = () => { T.sub(1, 2, true); };
        let case2 = () => { T.sub([1, 2], 3, true); };
        // the output shape is incompatible with the original tensor
        let case3 = () => { T.sub(T.zeros([3, 1]), T.zeros([1, 3]), true); };
        // output type is incompatible with the original tensor
        let case4 = () => { T.sub(T.zeros([2, 2], T.INT32), 1.2, true); };
        expect(case1).toThrow();
        expect(case2).toThrow();
        expect(case3).toThrow();
        expect(case4).toThrow();
    });
});
