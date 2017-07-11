import { JasmalEngine } from '../';
import { ComplexNumber } from '../lib/complexNumber';
import { Tensor } from '../lib/tensor';
import { checkArrayLike } from './testHelper';
import { DType } from '../lib/dtype';
const T = JasmalEngine.createInstance();

describe('DType constants', () => {
    it('T.* should be the same as DType.*', () => {
        expect(T.LOGIC).toEqual(DType.LOGIC);
        expect(T.INT32).toEqual(DType.INT32);
        expect(T.FLOAT64).toEqual(DType.FLOAT64);
    });
});

describe('Tensor creation', () => {
    describe('fromArray()', () => {
        it('should create a real vector', () => {
            let expected = [-1, 1, 3];
            let A = T.fromArray(expected);
            expect(A.dtype).toBe(DType.FLOAT64);
            expect(A.shape).toEqual([3]);
            expect(A.strides).toEqual([1]);
            expect(A.ndim).toBe(1);
            expect(A.size).toBe(expected.length);
            expect(A.hasComplexStorage()).toBeFalsy();
            expect(checkArrayLike(A.realData, expected));
            expect(() => A.imagData).toThrow();
        });
        it('should create a complex matrix', () => {
            let A = T.fromArray([[1, 2], [-3.5, 5]], [[-3, 9], [4, Infinity]]);
            expect(A.dtype).toBe(DType.FLOAT64);
            expect(A.shape).toEqual([2, 2]);
            expect(A.strides).toEqual([2, 1]);
            expect(A.ndim).toBe(2);
            expect(A.size).toBe(4);
            expect(A.hasComplexStorage()).toBeTruthy();
            expect(checkArrayLike(A.realData, [1, 2, -3.5, 5]));
            expect(checkArrayLike(A.imagData, [-3, 9, 4, Infinity]));
        });
        it('should create a real 4D tensor with data type INT32', () => {
            let A = T.fromArray(
                [[[[1], [2]], [[3], [4]]],
                 [[[5], [6]], [[7], [8]]]], [], DType.INT32);
            expect(A.dtype).toBe(DType.INT32);
            expect(A.shape).toEqual([2, 2, 2, 1]);
            expect(A.strides).toEqual([4, 2, 1, 1]);
            expect(A.ndim).toBe(4);
            expect(A.size).toBe(8);
            expect(A.hasComplexStorage()).toBeFalsy();
            expect(checkArrayLike(A.realData, [1, 2, 3, 4, 5, 6, 7, 8]));
            expect(() => A.imagData).toThrow();
        });
        it('should create a logic vector', () => {
            let A = T.fromArray([-1, 0.3, 0], [], DType.LOGIC);
            expect(A.dtype).toBe(DType.LOGIC);
            expect(checkArrayLike(A.realData, [1, 1, 0]));
        });
        it('should throw in invalid cases', () => {
            // empty array
            let case1 = () => { T.fromArray([]); };
            // inconsistent real and imaginary parts
            let case2 = () => { T.fromArray([1], [2,3]); };
            // inconsistent nested array
            let case3 = () => { T.fromArray([[1, 2], [3]]); };
            // dtype conversion not possible
            let case4 = () => { T.fromArray([1, NaN], [], DType.LOGIC); };
            let case5 = () => { T.fromArray([1], [2], DType.LOGIC); };
            expect(case1).toThrow();
            expect(case2).toThrow();
            expect(case3).toThrow();
            expect(case4).toThrow();
            expect(case5).toThrow();
        });
    });
    describe('toArray()', () => {
        it('should return the same array for the real part (with realOnly = true)', () => {
            let arr = [[1, 2], [3, 4]];
            expect(T.fromArray(arr).toArray(true)).toEqual(arr);
        });
        it('should return the same array for the real part, and an empty array for the imaginary part (with realOnly = false)', () => {
            let arr = [[1, 2], [3, 4]];
            let converted = T.fromArray(arr).toArray(false);
            expect(converted[0]).toEqual(arr);
            expect(converted[1]).toEqual([]);
        });
        it('should return the same array for both the real part and the imaginary part', () => {
            let arrRe = [[1, 2], [3, 4]];
            let arrIm = [[Infinity, 3.14], [-42, 7]];
            let converted = T.fromArray(arrRe, arrIm).toArray(false);
            expect(converted[0]).toEqual(arrRe);
            expect(converted[1]).toEqual(arrIm);
        });
    });
    describe('zeros()', () => {
        it('[2x2 FLOAT64]', () => {
            let x = T.zeros([2, 2], T.FLOAT64);
            expect(x.shape).toEqual([2, 2]);
            expect(x.dtype).toBe(T.FLOAT64);
            expect(x.hasComplexStorage()).toBe(false);
            let re = x.realData;
            for (let i = 0;i < re.length;i++) {
                expect(re[i]).toBe(0);
            }
        });
    });
    describe('ones()', () => {
        it('[2x3x4 INT32]', () => {
            let x = T.ones([2, 3, 4], T.INT32);
            expect(x.shape).toEqual([2, 3, 4]);
            expect(x.dtype).toBe(T.INT32);
            expect(x.hasComplexStorage()).toBe(false);
            let re = x.realData;
            for (let i = 0;i < re.length;i++) {
                expect(re[i]).toBe(1);
            }
        });
    });
    describe('setEl()/getEl() > ', () => {
        it('[2x3x2 FLOAT64]', () => {
            let x = T.zeros([2, 3, 2]);
            for (let i = 0;i < 2;i++) {
                for (let j = 0;j < 3;j++) {
                    for (let k = 0;k < 2;k++) {
                        let v = Math.random();
                        x.setEl(i, j, k, v);
                        expect(x.getEl(i, j, k)).toBe(v);
                    }
                }
            }
        });
        it('[2x3x2 FLOAT64 Complex]', () => {
            let x = T.zeros([2, 3, 2]);
            for (let i = 0;i < 2;i++) {
                for (let j = 0;j < 3;j++) {
                    for (let k = 0;k < 2;k++) {
                        let v = new ComplexNumber(Math.random(), Math.random());
                        x.setEl(i, j, k, v);
                        let vActual = <ComplexNumber>x.getEl(i, j, k);
                        expect(vActual.re).toBe(v.re);
                        expect(vActual.im).toBe(v.im);
                    }
                }
            }
        });
    });
    describe('equality tests', () => {
        let A1 = T.fromArray([[1, 2], [3, 4]], [], T.FLOAT64);
        let A2 = T.fromArray([[1, 2], [3, 4]], [], T.FLOAT64);
        let A3 = T.fromArray([[1, 2], [3, 4]], [], T.INT32);
        let B = T.fromArray([[1, 2], [3, 4]], [[0, 0], [0, 0]]);
        let C1 = T.fromArray([[1.001, 2.001], [2.999, 4.001]], [[-0.001, 0], [0.001, 0.001]]);
        let C2 = T.fromArray([[1.001, 2.008], [2.999, 4.001]], [[-0.001, 0], [0.001, 0.001]]);
        it('tests strict equality', () => {
            expect(Tensor.isEqual(A1, A2)).toBeTruthy();
            expect(Tensor.isEqual(A1, A3)).toBeFalsy();
            expect(Tensor.isEqual(A1, B)).toBeFalsy();
        });
        it('tests numerical equality', () => {
            expect(Tensor.isNumericallyEqual(A1, A2)).toBeTruthy();            
            expect(Tensor.isNumericallyEqual(A1, A3)).toBeTruthy();
            expect(Tensor.isNumericallyEqual(A1, B)).toBeTruthy();            
        });
        it('tests approximate equality', () => {
            expect(Tensor.isApproximatelyEqual(A1, A2, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, A3, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, B, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, C1, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, C2, 0.005)).toBeFalsy();
        });
    });
    describe('copy()', () => {
        let x = T.fromArray([[1, 2, 3]], [[-1, -2, -3]]);
        it('reference copy', () => {
            let y = x.copy();
            expect(y.shape).toEqual(x.shape);
            expect(y.dtype).toEqual(x.dtype);
            expect(y.realData).toBe(x.realData);
            expect(y.imagData).toBe(x.imagData);
        });
        it('copy immediately', () => {
            let y = x.copy(true);
            expect(y.shape).toEqual(x.shape);
            expect(y.dtype).toEqual(x.dtype);
            expect(y.realData === x.realData).toBeFalsy();
            expect(y.imagData === x.imagData).toBeFalsy();
            expect(y.realData).toEqual(x.realData);
            expect(y.imagData).toEqual(x.imagData);
        });
        it('setting an element after reference copy', () => {
            let y = x.copy();
            y.setEl(0, new ComplexNumber(9, -9));
            expect(y.realData[0]).toBe(9);
            expect(y.imagData[0]).toBe(-9);
            expect(x.realData[0]).toBe(1);
            expect(x.imagData[0]).toBe(-1);
        });
    });
    

});