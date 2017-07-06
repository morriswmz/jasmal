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
        let actual = <Tensor>T.abs([-1, 1, NaN, Infinity, -Infinity]);
        let expected = T.fromArray([1, 1, NaN, Infinity, Infinity]);
        checkTensor(actual, expected);
    });
    it('should also return the absolute values of complex numbers', () => {
        let actual = <Tensor>T.abs(T.fromArray([1, 1, Infinity], [-1, 0, 2]));
        let expected = T.fromArray([Math.sqrt(2), 1, Infinity]);
        checkTensor(actual, expected, 1e-16);
    });
});

describe('conj()', () => {
    it('should return the same value for real numbers', () => {
        let actual = <Tensor>T.conj(T.fromArray([1, -3.14, NaN, Infinity]));
        let expected = T.fromArray([1, -3.14, NaN, Infinity]);
        checkTensor(actual, expected);
    });
    it('should return the conjugate for complex numbers', () => {
        let actual = <Tensor>T.conj(T.fromArray([1, -3.14, 1e9], [-1, Infinity, 1e8]));
        let expected = T.fromArray([1, -3.14, 1e9], [1, -Infinity, -1e8]);
        checkTensor(actual, expected);
    });
});