import { JasmalEngine } from '..';
import { Tensor } from '../lib/tensor';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

describe('abs()', () => {
    it('should return a scalar if the input is a scalar', () => {
        expect(T.abs(-1)).toEqual(1);
        expect(T.abs(T.complexNumber(1, -2))).toEqual(Math.sqrt(5));
    });
    it('should return a tensor even if the input is a one-element array/tensor', () => {
        checkTensor(T.abs([-1]), Tensor.scalar(1));
        checkTensor(T.abs([[-1]]), T.fromArray([[1]]));
        checkTensor(T.abs(T.fromArray([1], [-2])), Tensor.scalar(Math.sqrt(5)));
    });
    it('should return the absolute values of real numbers', () => {
        let actual = T.abs([-1, 1, NaN, Infinity, -Infinity]);
        let expected = T.fromArray([1, 1, NaN, Infinity, Infinity]);
        checkTensor(actual, expected);
    });
    it('should also return the absolute values of complex numbers', () => {
        let actual = T.abs(T.fromArray([1, 1, Infinity], [-1, 0, 2]));
        let expected = T.fromArray([Math.sqrt(2), 1, Infinity]);
        checkTensor(actual, expected, 1e-16);
    });
});

describe('conj()', () => {
    it('should return the same value for real numbers', () => {
        let actual = T.conj(T.fromArray([1, -3.14, NaN, Infinity]));
        let expected = T.fromArray([1, -3.14, NaN, Infinity]);
        checkTensor(actual, expected);
    });
    it('should return the conjugate for complex numbers', () => {
        let actual = T.conj(T.fromArray([1, -3.14, 1e9], [-1, Infinity, 1e8]));
        let expected = T.fromArray([1, -3.14, 1e9], [1, -Infinity, -1e8]);
        checkTensor(actual, expected);
    });
});

describe('sqrt()', () => {
    it('should return the square roots for nonnegative numbers', () => {
        let actual = T.sqrt(T.fromArray([0, 2, 16, NaN, Infinity]));
        let expected = T.fromArray([0, Math.sqrt(2), 4, NaN, Infinity]);
        checkTensor(actual, expected);
    });
    it('should return the square roots for negative numbers', () => {
        let actual = T.sqrt(T.fromArray([1, -1, -Infinity, -2]));
        let expected = T.fromArray([1, 0, 0, 0], [0, 1, Infinity, Math.sqrt(2)]);
    });
    it('should return the square roots for complex numbers', () => {
        let x = T.fromArray(
            [0, 1, -2, 0, 3, Infinity, -Infinity, Infinity, 3, -10],
            [0, 1, 0, -2, 4, -9999999, 0.0000001, Infinity, Infinity, -Infinity]
        )
        let actual = T.sqrt(x);
        let expected = T.fromArray(
            [0, 1.098684113467810, 0, 1, 2, Infinity, 0, Infinity, Infinity, Infinity],
            [0, 0.455089860562227, Math.sqrt(2), -1, 1, 0, Infinity, Infinity, Infinity, -Infinity]
        );
        checkTensor(actual, expected, 1e-14);
    });
});
