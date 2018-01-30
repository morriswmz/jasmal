import { JasmalEngine } from '../index';
import { ComplexNumber } from '../lib/core/complexNumber';
import { Tensor } from '../lib/core/tensor';
import { checkArrayLike, checkTensor } from './testHelper';
import { DType } from '../lib/core/dtype';
import { TensorStorage } from '../lib/core/storage';
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
            checkArrayLike(A.realData, [1, 1, 0]);
        });
        it('should accept typed arrays', () => {
            let arr = new Float64Array([1, 2, 3]);
            let A = T.fromArray(arr);
            expect(A.dtype).toBe(DType.FLOAT64);
            expect(A.shape).toEqual([arr.length]);
            checkArrayLike(A.realData, arr);
            expect(A.hasComplexStorage()).toBeFalsy();
            // should do copy
            A.set(0, 100);
            expect(arr[0]).toBe(1); 
        });
        it('should accept typed arrays and create a complex tensor', () => {
            let arrRe = new Float64Array([1, 2, 3]);
            let arrIm = new Float32Array([-1, -2, -3]);
            let A = T.fromArray(arrRe, arrIm);
            expect(A.dtype).toBe(DType.FLOAT64);
            expect(A.shape).toEqual([arrRe.length]);
            checkArrayLike(A.realData, arrRe);
            checkArrayLike(A.imagData, arrIm);
            // should do copy
            A.set(0, 100);
            expect(arrRe[0]).toBe(1); 
            expect(arrIm[0]).toBe(-1); 
        });
        it('should throw in invalid cases', () => {
            // inconsistent real and imaginary parts
            let case1 = () => { T.fromArray([1], [2,3]); };
            // inconsistent nested array
            let case2 = () => { T.fromArray([[1, 2], [3]]); };
            // dtype conversion not possible
            let case3 = () => { T.fromArray([1, NaN], [], DType.LOGIC); };
            let case4 = () => { T.fromArray([1], [2], DType.LOGIC); };
            expect(case1).toThrow();
            expect(case2).toThrow();
            expect(case3).toThrow();
            expect(case4).toThrow();
        });
    });

    describe('complex()', () => {
        it('should combine two real tensors of the same shape to a complex tensor', () => {
            let re = T.fromArray([[1, 5]]);
            let im = T.fromArray([[3, 4]]);
            let z = T.complex(re, im);
            checkTensor(z, T.fromArray([[1, 5]], [[3, 4]]));
        });
        it('should copy the shape array', () => {
            let re = T.fromArray([2, 3]);
            let im = T.fromArray([6, 9]);
            let z = T.complex(re, im);
            z.prependAxis();
            expect(re.shape).toEqual([2]);
            expect(im.shape).toEqual([2]);
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
        it('should copy the shape parameter', () => {
            let shape = [3, 4];
            let x = T.zeros(shape);
            shape[1] = 10;
            expect(x.shape).toEqual([3, 4]);
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
        it('should copy the shape parameter', () => {
            let shape = [2, 3, 5];
            let x = T.ones(shape);
            shape[1] = 10;
            expect(x.shape).toEqual([2, 3, 5]);
        });
    });
});

describe('Tensor basic methods', () => {

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

    describe('ensureComplexStorage()', () => {
        it('should create the complex storage with the correct DType', () => {
            let A = T.ones([3, 4], T.INT32);
            A.ensureComplexStorage();
            let imStore = <TensorStorage>(<any>A)._im;
            expect(imStore.dtype).toBe(A.dtype);
            expect(imStore.refCount).toBe(1);
            let expectedArr = new Array(A.size);
            for (let i = 0;i < A.size;i++) {
                expectedArr[i] = 0;
            }
            checkArrayLike(imStore.data, expectedArr);
        });
        it('should do nothing when the tensor is already a complex one', () => {
            let expectedArr = [4, 5, 6];
            let A = T.fromArray([1, 2, 3], expectedArr);
            let imStore = <TensorStorage>(<any>A)._im;
            A.ensureComplexStorage();
            expect((<any>A)._im).toBe(imStore);
            expect(imStore.refCount).toBe(1);
            checkArrayLike(imStore.data, expectedArr);
        });
        it('should throw for logic tensors', () => {
            let L = T.zeros([5, 2], T.LOGIC);
            expect(() => L.ensureComplexStorage()).toThrow();
        });
    });

    describe('setEl()/getEl()', () => {
        it('getEl() should return the value set by setEl() for a 2x3x2 FLOAT64 tensor', () => {
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
        it('getEl() should return the value set by setEl() for a complex 2x3x2 FLOAT64 tensor', () => {
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
    describe('isEqual()/isNumericallyEqual()/isApproximatelyEqual()', () => {
        let A1 = T.fromArray([[1, 2], [3, 4]], [], T.FLOAT64);
        let A2 = T.fromArray([[1, 2], [3, 4]], [], T.FLOAT64);
        let A3 = T.fromArray([[1, 2], [3, 4]], [], T.INT32);
        let B = T.fromArray([[1, 2], [3, 4]], [[0, 0], [0, 0]]);
        let C1 = T.fromArray([[1.001, 2.001], [2.999, 4.001]], [[-0.001, 0], [0.001, 0.001]]);
        let C2 = T.fromArray([[1.001, 2.008], [2.999, 4.001]], [[-0.001, 0], [0.001, 0.001]]);
        it('should test strict equality', () => {
            expect(Tensor.isEqual(A1, A2)).toBeTruthy();
            expect(Tensor.isEqual(A1, A3)).toBeFalsy();
            expect(Tensor.isEqual(A1, B)).toBeFalsy();
        });
        it('should test numerical equality', () => {
            expect(Tensor.isNumericallyEqual(A1, A2)).toBeTruthy();            
            expect(Tensor.isNumericallyEqual(A1, A3)).toBeTruthy();
            expect(Tensor.isNumericallyEqual(A1, B)).toBeTruthy();            
        });
        it('should test approximate equality', () => {
            expect(Tensor.isApproximatelyEqual(A1, A2, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, A3, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, B, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, C1, 0.005)).toBeTruthy();
            expect(Tensor.isApproximatelyEqual(A1, C2, 0.005)).toBeFalsy();
        });
    });

    describe('real()', () => {
        it('should copy the shape array', () => {
            let x = T.fromArray([1, 2, 3, 4]);
            let reX = x.real();
            reX.appendAxis();
            expect(x.shape).toEqual([4]);
        });
    });

    describe('imag()', () => {
        it('should copy the shape array', () => {
            let x = T.fromArray([1, 2], [3, 4]);
            let imX = x.imag();
            imX.appendAxis();
            expect(x.shape).toEqual([2]);
        });
    });

    describe('copy()', () => {
        let x = T.fromArray([[1, 2, 3]], [[-1, -2, -3]]);
        it('it should return a "reference" copy by default', () => {
            let y = x.copy();
            expect(y.shape).toEqual(x.shape);
            expect(y.dtype).toEqual(x.dtype);
            expect(y.realData).toBe(x.realData);
            expect(y.imagData).toBe(x.imagData);
        });
        it('it should copy immediately when required', () => {
            let y = x.copy(true);
            expect(y.shape).toEqual(x.shape);
            expect(y.dtype).toEqual(x.dtype);
            expect(y.realData === x.realData).toBeFalsy();
            expect(y.imagData === x.imagData).toBeFalsy();
            expect(y.realData).toEqual(x.realData);
            expect(y.imagData).toEqual(x.imagData);
        });
        it('setting an element after reference copy should not change the original tensor', () => {
            let y = x.copy();
            y.setEl(0, new ComplexNumber(9, -9));
            expect(y.realData[0]).toBe(9);
            expect(y.imagData[0]).toBe(-9);
            expect(x.realData[0]).toBe(1);
            expect(x.imagData[0]).toBe(-1);
            expect(x.realData[1]).toBe(2);
            expect(x.imagData[1]).toBe(-2);
        });
        it('should copy the shape array', () => {
            let a = T.zeros([2, 3]);
            let b = a.copy();
            a.prependAxis(); // a: [1, 2, 3]
            // should not affect b
            expect(b.shape).toEqual([2, 3]);
            b.appendAxis(); // b: [2, 3, 1]
            // should not affect a
            expect(a.shape).toEqual([1, 2, 3]);
            let c = b.copy(true);
            b.prependAxis(); // b: [1, 2, 3, 1]
            // should not affect c
            expect(c.shape).toEqual([2, 3, 1]);
            c.appendAxis(); // c: [2, 3, 1, 1]
            // should not affect b
            expect(b.shape).toEqual([1, 2, 3, 1]);
        });
    });

    describe('asType()', () => {
        it('should make a copy in a different data type', () => {
            let x = T.fromArray([1, 2, 3], [], T.INT32);
            let y = x.asType(T.FLOAT64);
            expect(y.shape).toEqual(x.shape);
            expect(y.dtype).toBe(T.FLOAT64);
            checkArrayLike(y.realData, x.realData);
        });
        it('should cast non logic data to logic data', () => {
            let x = T.fromArray([0, 1.2, -3, 0, 0]);
            let y = x.asType(T.LOGIC);
            expect(y.shape).toEqual(x.shape);
            expect(y.dtype).toBe(T.LOGIC);
            checkArrayLike(y.realData, [0, 1, 1, 0, 0]);
        });
        it('should throw when converting NaNs to logical values', () => {
            let x = T.fromArray([1, 0, NaN]);
            expect(() => x.asType(T.LOGIC)).toThrow();
        });
        it('should copy the shape array', () => {
            let x = T.ones([3, 3]);
            let y = x.asType(T.FLOAT64);
            let z = x.asType(T.INT32);
            x.prependAxis();
            expect(y.shape).toEqual([3, 3]);
            expect(z.shape).toEqual([3, 3]);
        });
    });

    describe('getReshapedCopy()', () => {
        it('should copy the new shape array', () => {
            let x = T.ones([3, 4]);
            let newShape = [2, 2, 3];
            let y = x.getReshapedCopy(newShape);
            newShape[0] = 5;
            expect(y.shape).toEqual([2, 2, 3]);
        });
    });

});
