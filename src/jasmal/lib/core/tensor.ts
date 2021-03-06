import { TensorStorage } from './storage';
import { DType, DTypeHelper } from './dtype';
import { ComplexNumber } from './complexNumber';
import { Scalar, OpInputInfo, OpInputType, OpInput, TypedArray, DataBlock } from '../commonTypes';
import { ShapeHelper } from '../helper/shapeHelper';
import { DataHelper } from '../helper/dataHelper';
import { ObjectHelper } from '../helper/objHelper';
import { IIndexIterator, ArrayBasedIndexIterator, ConstantIndexIterator, RangedIndexIterator,
         ReversedRangedIndexIterator } from './iterator';

type OffsetCalculator = (indices: ArrayLike<number>, strides: number[]) => number;

class OffsetCalculatorFactory {

    private static _cached: Array<OffsetCalculator> = [];

    public static create(dim: number): OffsetCalculator {
        // To comply with plain JavaScript arrays, we use row-major ordering.
        if (!OffsetCalculatorFactory._cached[dim]) {
            let funcBody = '\'use strict\'; return ';
            for (let i = 0; i < dim - 1;i++) {
                funcBody += `indices[${i}] * strides[${i}] + `
            }
            funcBody += `indices[${dim - 1}];`;
            OffsetCalculatorFactory._cached[dim] = <OffsetCalculator>(new Function('indices', 'strides', funcBody));
        }
        return OffsetCalculatorFactory._cached[dim];
    }

}

export class Tensor {

    private _re: TensorStorage;
    private _im: TensorStorage;
    private _shape: number[];
    private _strides: number[];
    private _offsetCalculator: OffsetCalculator;

    /**
     * Internal constructor for Tensor objects.
     * Note: 1. The refCount of the input parameters will be automatically
     *       increased.
     *       2. The shape array should NOT come from any external source. The
     *       constructor does not make a copy of the shape array input. If it
     *       can be modified elsewhere, MAKE A COPY before passing it in.
     * @param re 
     * @param im 
     * @param shape 
     */
    protected constructor(re: TensorStorage, im: TensorStorage, shape: number[]) {
        this._re = re;
        this._re.refCount++;
        this._im = im;
        if (im !== TensorStorage.Empty) {
            if (im.dtype !== re.dtype) {
                throw new Error('Inconsistent dtype.');
            }
            this._im.refCount++;
        }
        this._shape = shape;
        this._updateStridesAndCalculator();
    }

    public static ZERO: Tensor = Tensor.zeros([1]);

    /**
     * Combines two real tensors into a complex tensor.
     * The two tensors must share the same shape and data type.
     * @param re Tensor corresponding to the real part.
     * @param im Tensor corresponding to the imaginary part.
     */
    public static complex(re: Tensor, im: Tensor): Tensor {
        if (re.hasComplexStorage() || im.hasComplexStorage()) {
            throw new Error('Real tensor(s) expected.');
        }
        if (re.dtype !== im.dtype) {
            throw new Error('Real part and imaginary part must share the same data type.');
        }
        if (!ShapeHelper.compareShape(re._shape, im._shape)) {
            throw new Error('Real part and imaginary part must share the same shape.');
        }
        // Note: the shape array needs to be copied here.
        return new Tensor(re._re, im._re, re.shape);
    }

    /**
     * Creates a tensor from JavaScript arrays.
     * @param re Real part.
     * @param im (Optional) Imaginary part.
     * @param dtype (Optional) Data type. Default value is DType.FLOAT64.
     */
    public static fromArray(re: any[] | TypedArray, im?: any[] | TypedArray, dtype: DType = DType.FLOAT64): Tensor {
        TensorStorage.ValidateDTypeSupport(dtype);
        if (re == undefined) throw new Error('Real part must be specified.');
        let isReTypedArray = ObjectHelper.isTypedArray(re);
        if (!Array.isArray(re) && !isReTypedArray) {
            throw new Error('Array expected.');
        }
        let isImTypedArray = false;
        if (im != undefined && !Array.isArray(im) && !(isImTypedArray = ObjectHelper.isTypedArray(im))) {
            throw new Error('Array expected.');
        }
        let isComplex = im && im.length > 0;
        if (isComplex && dtype === DType.LOGIC) {
            throw new Error('Cannot convert a complex array to a logic tensor.');
        }
        // determines the shape
        let shape = ShapeHelper.inferShapeFromArray(re);
        ShapeHelper.validateArrayShape(re, shape);
        if (isComplex) ShapeHelper.validateArrayShape(<any[] | TypedArray>im, shape);
        // copies the data
        let reStorage = isReTypedArray
            ? TensorStorage.fromTypedArray(<TypedArray>re, dtype)
            : TensorStorage.fromArray(<any[]>re, shape, dtype);
        let imStorage: TensorStorage;
        if (isComplex) {
            imStorage = isImTypedArray
                ? TensorStorage.fromTypedArray(<TypedArray>im, dtype)
                : TensorStorage.fromArray(<any[]>im, shape, dtype);
        } else {
            imStorage = TensorStorage.Empty;
        }
        return new Tensor(reStorage, imStorage, shape);
    }

    /**
     * Creates a new tensor filled with zeros.
     * @param shape Shape of the tensor.
     * @param dtype Data type.
     */
    public static zeros(shape: ArrayLike<number>, dtype: DType = DType.FLOAT64): Tensor {
        ShapeHelper.validateShape(shape);
        let re = TensorStorage.create(ShapeHelper.getSizeFromShape(shape), dtype);
        // Always copy the shape since it may be modified elsewhere.
        return new Tensor(re, TensorStorage.Empty, Array.isArray(shape) ? shape.slice() : Array.prototype.slice.call(shape));
    }

    /**
     * Creates a new tensor filled with ones.
     * @param shape Shape of the tensor.
     * @param dtype Data type.
     */
    public static ones(shape: ArrayLike<number>, dtype: DType = DType.FLOAT64): Tensor {
        ShapeHelper.validateShape(shape);
        let re = TensorStorage.create(ShapeHelper.getSizeFromShape(shape), dtype);
        for (let i = 0;i < re.data.length;i++) {
            re.data[i] = 1;
        }
        // Always copy the shape array since it can be modified elsewhere.
        return new Tensor(re, TensorStorage.Empty, Array.isArray(shape) ? shape.slice() : Array.prototype.slice.call(shape));
    }

    /**
     * Creates a scalar tensor.
     * @param re Real part.
     * @param im Imaginary part.
     */
    public static scalar(re: number, im?: number, dtype?: DType, ndim?: number): Tensor;
    /**
     * Create a scalar tensor from a complex number.
     * @param z A complex number.
     */
    public static scalar(z: ComplexNumber): Tensor;
    public static scalar(x: number | ComplexNumber, y?: number, dtype: DType = DType.FLOAT64, ndim: number = 1): Tensor {
        let re: number, im: number;
        if (x instanceof ComplexNumber) {
            re = x.re;
            im = x.im;
        } else {
            re = x;
            im = y || 0;
        }
        let reStorage = TensorStorage.create(1, dtype);
        reStorage.data[0] = re;
        let imStorage;
        if (im !== 0) {
            imStorage = TensorStorage.create(1, dtype);
            imStorage.data[0] = im;
        } else {
            imStorage = TensorStorage.Empty;
        }
        let T = new Tensor(reStorage, imStorage, [1]);
        if (ndim > 1) {
            let shape = new Array(ndim);
            for (let i = 0;i < ndim;i++) {
                shape[i] = 1;
            }
            T._shape = shape;
            T._updateStridesAndCalculator();
        }
        return T;
    }

    /**
     * Converts compatible data to a tensor.
     * @param x A number, complex number, numerical array, or a tensor.
     */
    public static toTensor(x: number | ComplexNumber | any[] | TypedArray): Tensor {
        if (Array.isArray(x)) {
            return Tensor.fromArray(x);
        } else if (ObjectHelper.isTypedArray(x)) {
            return Tensor.fromArray(x);
        } else if (x instanceof ComplexNumber) {
            return Tensor.scalar(x);
        } else if (Object.prototype.toString.call(x) === '[object Number]') {
            return Tensor.scalar(x);
        } else {
            throw new Error(`Cannot convert ${Object.prototype.toString.call(x)} to a tensor.`);
        }
    }

    /**
     * Checks if tensor x and tensor y are exactly equal (===). Two tensors x, y
     * are exactly equal if the following conditions are met:
     * 1. x and y have the same data type.
     * 2. x and y have the same shape.
     * 3. Either both x and y have complex storages, or neither x or y has
     *    complex storages (which implies a+0j != a).
     * 4. x and y have the same elements.
     * Note: in most of cases you want to use isNumericallyEqual() or
     * isApproximatelyEqual().
     * @param x Tensor x.
     * @param y Tensor y.
     */
    public static isEqual(x: Tensor, y: Tensor): boolean {
        if (x === y) return true;
        if (x == undefined || y == undefined) return false;
        if (!ShapeHelper.compareShape(x._shape, y._shape)) return false;
        if (x.dtype !== y.dtype) return false;
        if (x.hasComplexStorage() !== y.hasComplexStorage()) return false;
        if (!DataHelper.areArraysEqual(x.realData, y.realData)) return false;
        if (x.hasComplexStorage() && !DataHelper.areArraysEqual(x.imagData, y.imagData)) return false;
        return true;
    }

    /**
     * Checks if tensor x and tensor y are numerically equal. Two tensors x, y
     * are numerically equal if the following conditions are met:
     * 1. x and y have the same shape.
     * 2. x and y have the same elements numerically (here a+0j == a).
     * @param x Tensor x.
     * @param y Tensor y.
     */
    public static isNumericallyEqual(x: Tensor, y: Tensor): boolean {
        if (x === y) return true;
        if (x == undefined || y == undefined) return false;
        if (!ShapeHelper.compareShape(x._shape, y._shape)) return false;
        if (!DataHelper.areArraysEqual(x.realData, y.realData)) return false;
        if (x.hasComplexStorage()) {
            if (y.hasComplexStorage()) {
                return DataHelper.areArraysEqual(x.imagData, y.imagData);
            } else {
                return DataHelper.isArrayAllZeros(x.imagData);
            }
        } else {
            if (y.hasComplexStorage()) {
                return DataHelper.isArrayAllZeros(y.imagData);
            } else {
                return true;
            }
        }
    }

    /**
     * Checks if tensor x and tensor y are approximately equal. Two tensors x, y
     * are approximately equal if the following conditions are met:
     * 1. x and y have the same shape.
     * 2. |reX[i] - reY[i]| <= tolerance && |imX[i] - imY[i]| <= tolerance
     * @param x Tensor x.
     * @param y Tensor y.
     * @param tolerance Tolerance.
     */
    public static isApproximatelyEqual(x: Tensor, y: Tensor, tolerance: number): boolean {
        if (tolerance < 0) {
            throw new Error('Tolerance must be nonnegative.');
        }
        if (x === y) return true;
        if (x == undefined || y == undefined) return false;
        if (!ShapeHelper.compareShape(x._shape, y._shape)) return false;
        if (!DataHelper.areArraysApproximatelyEqual(x.realData, y.realData, tolerance)) return false;
        if (x.hasComplexStorage()) {
            if (y.hasComplexStorage()) {
                return DataHelper.areArraysApproximatelyEqual(x.imagData, y.imagData, tolerance);
            } else {
                return DataHelper.isArrayApproximatelyAllZeros(x.imagData, tolerance);
            }
        } else {
            if (y.hasComplexStorage()) {
                return DataHelper.isArrayApproximatelyAllZeros(y.imagData, tolerance);
            } else {
                return true;
            }
        }
    }

    /**
     * Analyzes the input.
     * @param value The input value.
     */
    public static analyzeOpInput(value: OpInput): OpInputInfo {
        let isOriginalTypeScalar = false;
        let hasOnlyOneElement = false, isComplex = false;
        let re = 0, im = 0;
        let reArr: ArrayLike<number> = [], imArr: ArrayLike<number> = [];
        let originalShape = [1];
        let originalType = OpInputType.Unknown;
        let originalDType = DType.FLOAT64;
        if (value instanceof Tensor) {
            reArr = value._re.data;
            hasOnlyOneElement = value.size === 1;
            if (hasOnlyOneElement) {
                re = value._re.data[0];
                if (value.hasComplexStorage()) {
                    im = value._im.data[0];
                }
                isComplex = im !== 0;
                if (isComplex) {
                    imArr = value._im.data;
                }
            } else {
                if (value.hasNonZeroComplexStorage()) {
                    imArr = value._im.data;
                    isComplex = true;
                }
            }
            // No need to make unnecessary copies here as originalShape is
            // readonly.
            originalShape = value._shape;
            originalType = OpInputType.Tensor;
            originalDType = value.dtype;
        } else if (Array.isArray(value) || ObjectHelper.isTypedArray(value)) {
            if (Array.isArray(value[0])) {
                // Detected a nested array, convert to Tensor.
                // This is not very efficient because we need to copy every
                // elements, but we have to do this because flat indexing is
                // used internally.
                let tmp = Tensor.analyzeOpInput(Tensor.fromArray(value));
                tmp.originalType = OpInputType.Array;
                return tmp;
            } else {
                // plain array or typed array
                reArr = value;
                hasOnlyOneElement = value.length === 1;
                if (hasOnlyOneElement) {
                    re = value[0];                  
                } else {
                    originalShape = [value.length];
                }
                originalType = OpInputType.Array;
            }
        } else if (value instanceof ComplexNumber) {
            re = value.re;
            im = value.im;
            isComplex = im !== 0;
            hasOnlyOneElement = true;
            originalType = OpInputType.ComplexNumber;
            isOriginalTypeScalar = true;
        } else if (typeof value === 'number') {
            re = value;
            hasOnlyOneElement = true;
            originalType = OpInputType.Number;
            isOriginalTypeScalar = true;
        } else {
            throw new Error(`Unsupported value ${value}.`);
        }
        return {
            isInputScalar: isOriginalTypeScalar,
            hasOnlyOneElement: hasOnlyOneElement,
            isComplex: isComplex,
            re: re,
            im: im,
            reArr: reArr,
            imArr: imArr,
            originalShape: originalShape,
            originalType: originalType,
            originalDType: originalDType,
            originalInput: value
        };
    }

    /**
     * Gets the data type.
     */
    public get dtype(): DType {
        return this._re.dtype;
    }

    /**
     * Returns the shape of this tensor.
     * Note: this always returns a copy of the actual shape array.
     */
    public get shape(): number[] {
        return this._shape.slice();
    }

    /**
     * Returns the strides of this tensor.
     * Note: this always returns a copy of the actual strides array.
     */
    public get strides(): number[] {
        return this._strides.slice();
    }

    private _updateStridesAndCalculator(): void {
        this._strides = ShapeHelper.computeStrides(this._shape);
        this._offsetCalculator = OffsetCalculatorFactory.create(this._shape.length);
    }

    /**
     * Returns the number of dimensions of this tensor.
     */
    public get ndim(): number {
        return this._shape.length;
    }

    /**
     * Returns the number of elements in this tensor.
     */
    public get size(): number {
        return this._re.data.length;
    }

    /**
     * Retrieves the underlying data of the real part.
     * WARNING: before writing anything directly into the underlying storage,
     *          remember to call ensureUnsharedLocalStorage().
     */
    public get realData(): DataBlock {
        return this._re.data;
    }

    /**
     * Retrieves the underlying data of the imaginary part.
     * If this tensor has no imaginary part, an error will be thrown.
     * You should always use hasComplexStorage() to check if the imaginary part
     * is available before accessing it.
     * WARNING: before writing anything directly into the underlying storage,
     *          remember to call ensureUnsharedLocalStorage().
     */
    public get imagData(): DataBlock {
        if (!this.hasComplexStorage()) {
            throw new Error('Attempting to access the imaginary part for a real matrix.')
        }
        return this._im.data;
    }

    /**
     * Returns whether this tensor is a scalar.
     */
    public isScalar(): boolean {
        return this._re.data.length === 1;
    }

    /**
     * Returns whether this tensor is empty.
     */
    public isEmpty(): boolean {
        return this._re.data.length === 0;
    }

    /**
     * Returns whether this tensor has complex storage.
     * Note: DO NOT use this method to test if a tensor is numerically complex.
     *       Use hasNonZeroComplexStorage() instead.
     */
    public hasComplexStorage(): boolean {
        return this._im !== TensorStorage.Empty;
    }

    /**
     * Returns where this tensor has non-zero complex storage.
     */
    public hasNonZeroComplexStorage(): boolean {
        return this._im !== TensorStorage.Empty && !DataHelper.isArrayAllZeros(this._im.data);
    }

    /**
     * Sets multiple elements at once. This function supports the following call
     * signatures:
     * 
     * 1. set(indices, value)
     *      Here the type of `indices` can be one of the following: number,
     *      number[], string, Tensor. The type of `value` can be one of the
     *      following: number, ComplexNumber, nested array, Tensor. If `indices`
     *      is a nested array or a tensor with more than one dimensions, its
     *      flattened version will be used. The same applies for `value`.
     *      If `value` is a scalar, all elements specified by `indices` will
     *      be set to `value`.
     *      If `value` is not a scalar, it must have the same number of elements
     *      as `indices`, and the element at `indices[i]` will be set to
     *      `value[i]`.
     * 
     * 2. set(mask, value)
     *      Here `mask` must be a logic tensor with the same shape of this
     *      tensor. The type of `value` can be one of the following: number,
     *      ComplexNumber, nested array, Tensor. `value` can be either a scalar,
     *      or an array/Tensor whose size is equal to the number of true values
     *      in `mask`.
     * 
     * 3. set(condition, value)
     *      Here `condition` is a function with the following signature:
     *      (re: number, im?: number) => boolean, and the type of `value` can be 
     *      one of the following - number, ComplexNumber.
     *      Here `condition` will be tested against every element in this
     *      tensor. If `condition` returns true for the current element, the
     *      current element will be set to `value`.
     * 
     * 4. set(i1, i2, ..., iD, value)
     *      Here D denote the number of dimensions of this tensor. The type of
     *      `i1`, ..., `iD` can be one of the following: number, number[],
     *      string Tensor. The type of `value` can be one of the following:
     *      number, ComplexNumber, number[], Tensor.
     *      Here `i1`, ..., `iD` selects a sub-tensor of shape
     *      S = [length(i1), ..., length(iD)], where length(ik) is defined as
     *      follows:
     *          1) If ik is a number, length(ik) = 1.
     *          2) If ik is a logic tensor, length(ik) is the length of
     *             find(ik).
     *          3) If ik is a string describing a slicing operation (which
     *             follows the Python syntax), length(ik) is determined by the
     *             slicing operation. For instance, if ik is '0:2', then
     *             length(ik) is 2.
     *          4) Otherwise, length(ik) is the number of elements in ik.
     *     `value` must be either a scalar or a tensor of shape S.
     * 
     * @param args Arguments.
     * @example
     *  x.set(1, 0) // set the 2nd element to 0
     *  x.set(0, 1, 2, 3) // set the (0,1,2)-th element to 3
     *  x.set([0, 1], ':', 0) // set the first two rows to 0
     *  x.set(x => x < 0, 0) // set all the negative elements to 0
     *  x.set(':', 0) // set all the elements to 0
     *  x.set('::2', ':', 0) // set all odd rows (1st, 3rd, etc.) to 0
     */
    public set(...args: any[]): void;
    public set(): void {
        if (arguments.length < 2) {
            throw new Error('Too few arguments.');
        }
        if (arguments.length !== 2 && arguments.length !== this.ndim + 1) {
            throw new Error(`Expecting ${this.ndim + 1} arguments.`);
        }
        this._handleSetGet(arguments, true);
    }

    private _setBatch(indices: ArrayLike<number>, value: number | ComplexNumber | number[] | Tensor): void {
        // _setBatch can be implemented by first flatten this tensor, and then
        // call _setSubTensor
        this._setSubTensor([new ArrayBasedIndexIterator(indices)], value);
    }
    
    private _setSubTensor(iters: IIndexIterator[], value: number | ComplexNumber | any[] | Tensor): void {
        let strides: number[];
        let trailingOffset: number = 0;
        let finalStride: number = 1;
        // determine the shape of the sub tensor
        let [shapeSub, sizeSub] = Tensor._inferShapeFromIters(iters);
        if (sizeSub === 0) {
            // no element to update
            return;
        }
        let maxLevel = iters.length - 1;
        if (iters.length === 1) {
            strides = [this.size];
        } else {
            strides = this._strides;
            // Determine trailingOffset
            // If the sub tensor has many trailing singleton dimensions, we can
            // just combine them by computing the proper trailing offset.
            // e.g. T[i, j, 0, 1, 3] = T[i, j, 0*d[2] + 1*d[1] + 3*d[0]]
            // The trailing offset if given by 0*d[2] + 1*d[1] + 3*d[0].
            let i = shapeSub.length - 1;
            while (shapeSub[i] === 1 && i > 0) {
                i--;
            }
            maxLevel = i;
            trailingOffset = 0;
            for (let k = i + 1;k < shapeSub.length;k++) {
                trailingOffset += strides[k] *= iters[k].peekNext();
                finalStride *= strides[k - 1];
            }
        }
        // process values
        let v = Tensor.analyzeOpInput(value);
        // update the sub tensor: four cases
        // It is kind of messy down there.
        this.ensureUnsharedLocalStorage();
        if (v.hasOnlyOneElement) {
            let newRe = v.re, newIm = v.im;
            if (v.isComplex) {
                // one complex value
                this.ensureComplexStorage();
                this._setSubTensorC1(iters, newRe, newIm, maxLevel, strides,
                    finalStride, trailingOffset, 0, 0);
            } else {
                this._setSubTensorR1(iters, newRe, maxLevel, strides,
                    finalStride, trailingOffset, 0, 0);
            }
        } else {
            if (v.originalShape.length === 1 && v.reArr.length !== sizeSub) {
                throw new Error('The size of the tensor/array does not match the number of indices.');
            }
            if (v.originalShape.length > 1 && !ShapeHelper.compareSqueezedShape(v.originalShape, shapeSub)) {
                throw new Error(`Attempting to update the ${ShapeHelper.shapeToString(shapeSub)} sub tensor`
                    + `with a tensor of incompatible shape ${ShapeHelper.shapeToString(v.originalShape)}.`);
            }
            // Note: we will always be reading the elements in v.re/v.im
            // consecutively. Since the number of dimensions is variable,
            // recursion is used.
            let newRe = v.reArr, newIm = v.imArr;
            let stridesSub = ShapeHelper.computeStrides(shapeSub);
            if (v.isComplex) {
                this.ensureComplexStorage();
                this._setSubTensorCN(iters, newRe, newIm, maxLevel, strides,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            } else {
                this._setSubTensorRN(iters, newRe, maxLevel, strides,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            }
        }
    }

    private _setSubTensorR1(iters: IIndexIterator[], newRe: number,
                            maxLevel: number, strides: number[], finalStride:number,
                            trailingOffset: number, level: number, offsetX: number): void {
        let iter = iters[level];
        iter.reset();        
        if (level === maxLevel) {
            // last level
            while (iter.hasNext()) {
                this._re.data[offsetX + iter.next() * finalStride + trailingOffset] = newRe;
            }
        } else {
            while (iter.hasNext()) {
                this._setSubTensorR1(iters, newRe, maxLevel, strides,
                    finalStride, trailingOffset, level + 1,
                    offsetX + strides[level] * iter.next());
            }
        }
    }

    private _setSubTensorC1(iters: IIndexIterator[],
                            newRe: number, newIm: number, maxLevel: number,      
                            strides: number[], finalStride: number,
                            trailingOffset: number, level: number, offsetX: number): void {
        let iter = iters[level];
        iter.reset();        
        if (level === maxLevel) {
            // last level
            while (iter.hasNext()) {
                let idx = iter.next();
                this._re.data[offsetX + idx * finalStride + trailingOffset] = newRe;
                this._im.data[offsetX + idx * finalStride + trailingOffset] = newIm;
            }
        } else {
            while (iter.hasNext()) {
                this._setSubTensorC1(iters, newRe, newIm, maxLevel,
                    strides, finalStride, trailingOffset, level + 1,
                    offsetX + strides[level] * iter.next());
            }
        }
    }

    private _setSubTensorRN(iters: IIndexIterator[],
                            newRe: ArrayLike<number>, maxLevel: number,
                            strides: number[], stridesSub: number[],
                            finalStride: number, trailingOffset: number, level: number,
                            offsetX: number, offsetY: number): void {
        let iter = iters[level];
        iter.reset();
        if (level === maxLevel) {
            // last level
            let j = 0;
            while (iter.hasNext()) {
                this._re.data[offsetX + iter.next() * finalStride + trailingOffset] = newRe[offsetY + j];
                j++;
            }
        } else {
            while (iter.hasNext()) {
                this._setSubTensorRN(iters, newRe, maxLevel, strides,
                    stridesSub, finalStride, trailingOffset, level + 1,
                    offsetX + strides[level] * iter.next(), offsetY);
                offsetY += stridesSub[level];
            }
        }
    }

    private _setSubTensorCN(iters: IIndexIterator[],
                            newRe: ArrayLike<number>, newIm: ArrayLike<number>,
                            maxLevel: number, strides: number[], stridesSub: number[],
                            finalStride: number, trailingOffset: number, level: number,
                            offsetX: number, offsetY: number): void {
        let iter = iters[level];
        iter.reset();
        if (level === maxLevel) {
            // last level
            let j: number = 0;
            while (iter.hasNext()) {
                let i = iter.next();
                this._re.data[offsetX + i * finalStride + trailingOffset] = newRe[offsetY + j];
                this._im.data[offsetX + i * finalStride + trailingOffset] = newIm[offsetY + j];
                j++;
            }
        } else {
            while (iter.hasNext()) {
                this._setSubTensorCN(iters, newRe, newIm, maxLevel,
                    strides, stridesSub, finalStride, trailingOffset, level + 1,
                    offsetX + strides[level] * iter.next(), offsetY);
                offsetY += stridesSub[level];
            }
        }
    }

    /**
     * Gets multiple elements at once. This function supports the following call
     * signatures:
     * 
     * 1. gets(indices, keepDims = false)
     *      Here the type of `indices` can be one of the following: number,
     *      nested array, Tensor. If `indices` is a scalar, this function is
     *      equivalent to {@link Tensor#getEl}. Otherwise, this function will
     *      attempt to construct a Tensor of the same shape according with the
     *      (i,j,...)-th element specified by `indices[i,j,...]`.
     * 
     * 2. get(mask, keepDims = false)
     *      Here `mask` must be a logic tensor with the same shape of this
     *      tensor. Elements corresponding to the `true` values in `mask` will
     *      be returned.
     * 
     * 3. get(condition, keepDims = false)
     *      Here `condition` is a function with the following signature:
     *      (re: number, im?: number) => boolean.
     *      Here `condition` will be tested against every element in this
     *      tensor. If `condition` returns true for the current element, the
     *      current element will be added to the list of returned elements.
     * 
     * 4. get(i1, i2, ..., iD, keepDims = false)
     *      Here D denote the number of dimensions of this tensor. The type of
     *      `i1`, ..., `iD` can be one of the following: number, number[],
     *      string, Tensor.
     *      Here `i1`, ..., `iD` selects a sub-tensor of shape
     *      S = [length(i1), ..., length(iD)], where length(ik) is defined as
     *      follows:
     *          1) If ik is a number, length(ik) = 1.
     *          2) If ik is a logic tensor, length(ik) is the length of
     *             find(ik).
     *          3) If ik is a string describing a slicing operation (which
     *             follows the Python syntax), length(ik) is determined by the
     *             slicing operation. For instance, if ik is '0:2', then
     *             length(ik) is 2.
     *          4) Otherwise, length(ik) is the number of elements in ik.
     *     `value` must be either a scalar or a tensor of shape S.
     * 
     * Singleton dimension removal rule: If ik is a number or a string
     * representing a single number, the k-th dimension of the resulting tensor
     * will be a singleton dimension, and will be removed by default. For other
     * cases, the k-th dimension will always be kept even if it is a singleton
     * dimension.
     * 
     * Example:
     * 
     * // A is [[1, 2], [3, 4]]
     * A.get(0, 0) // 1
     * A.get('0', '0') // 1
     * A.get(0, '0:1') // [1]
     * A.get('0:1', '0:1') // [[1]]
     * 
     * If you wish to prevent this behavior, set `keepDims` to true.
     * 
     * @param args Arguments.
     */
    public get(...args: any[]): Tensor | Scalar;
    public get(): Tensor | Scalar {
        if (arguments.length < 1) {
            throw new Error('Too few arguments.');
        }
        // check if keep dims is present
        let keepDims = false;
        let nIdx = arguments.length;
        if (arguments.length > 1) {
            if (Object.prototype.toString.call(arguments[arguments.length - 1]) === '[object Boolean]') {
                keepDims = arguments[arguments.length - 1];
                nIdx--;
            }
        }
        if (nIdx !== 1 && nIdx !== this.ndim) {
            throw new Error(`Expecting ${nIdx} index arguments.`);
        }
        return this._handleSetGet(
            nIdx === arguments.length ? arguments : Array.prototype.slice.call(arguments, 0, nIdx),
            false, keepDims);
    }

    private _getBatch(indices: ArrayLike<number>, keepDims: boolean): Tensor | Scalar {
        return this._getSubTensor([new ArrayBasedIndexIterator(indices)], keepDims);
    }

    private _getSubTensor(iters: IIndexIterator[], keepDims: boolean): Tensor | Scalar {
        let stridesX: number[];
        // determine the shape of the sub tensor
        let [shapeSub, sizeSub] = Tensor._inferShapeFromIters(iters);
        let result: Tensor;
        if (sizeSub === 0) {
            // special treatment for empty output
            result = Tensor.zeros(shapeSub, this.dtype);
        } else {
            let maxLevel = iters.length - 1;
            if (iters.length === 1) {
                stridesX = [this.size];
            } else {
                stridesX = this._strides;
            }
            result = Tensor.zeros(shapeSub, this.dtype);
            let newRe = result._re.data;
            // determine trailingOffset
            let i = shapeSub.length - 1;
            let trailingOffset = 0;
            let finalStride = 1;
            let stridesSub = ShapeHelper.computeStrides(shapeSub);
            while (shapeSub[i] === 1 && i > 0) {
                i--;
            }
            maxLevel = i;
            trailingOffset = 0;
            for (let k = i + 1;k < shapeSub.length;k++) {
                // Since the size of the sub tensor is not zero, peekNext()
                // will never throw here.
                trailingOffset += iters[k].peekNext();
                finalStride *= stridesX[k - 1];
            }
            // retrieve the sub tensor
            if (this.hasComplexStorage()) {
                result.ensureComplexStorage();
                let newIm = result._im.data;
                this._getSubTensorC(iters, newRe, newIm, maxLevel, stridesX,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            } else {
                this._getSubTensorR(iters, newRe, maxLevel, stridesX,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            }
        }
        if (keepDims) {
            return result;
        } else {
            // remove singleton dimensions along the axis where the index
            // iterator is a ConstantIndexIterator.
            let newSubShape: number[] = [];
            for (let i = 0;i < shapeSub.length;i++) {
                if (iters[i] instanceof ConstantIndexIterator) {
                    continue;
                }
                newSubShape.push(shapeSub[i]);
            }
            if (newSubShape.length === 0) {
                // returns a scalar
                return result.hasComplexStorage()
                    ? (result._im.data[0] === 0 ? result._re.data[0] : new ComplexNumber(result._re.data[0], result._im.data[0]))
                    : result._re.data[0];
            } else {
                result._shape = newSubShape;
                result._updateStridesAndCalculator();
                return result;
            }
        }
    }

    private _getSubTensorR(iters: IIndexIterator[],
                           newRe: DataBlock, maxLevel: number,
                           stridesX: number[], stridesSub: number[],
                           finalStride: number, trailingOffset: number, level: number,
                           offsetX: number, offsetSub: number): void {
        let iter = iters[level];
        iter.reset();
        if (level === maxLevel) {
            // last level
            let j = 0;
            while (iter.hasNext()) {
                newRe[offsetSub + j] = this._re.data[offsetX + iter.next() * finalStride + trailingOffset];
                j++;
            }
        } else {
            while (iter.hasNext()) {
                this._getSubTensorR(iters, newRe, maxLevel, stridesX,
                    stridesSub, finalStride, trailingOffset, level + 1,
                    offsetX + stridesX[level] * iter.next(), offsetSub);
                offsetSub += stridesSub[level];
            }
        }
    }

    private _getSubTensorC(iters: IIndexIterator[],
                           newRe: DataBlock, newIm: DataBlock, maxLevel: number,
                           stridesX: number[], stridesSub: number[],
                           finalStride: number, trailingOffset: number, level: number,
                           offsetX: number, offsetSub: number): void {
        let iter = iters[level];
        iter.reset();
        if (level === maxLevel) {
            // last level
            let j = 0;
            while (iter.hasNext()) {
                let i = iter.next();
                newRe[offsetSub + j] = this._re.data[offsetX + i * finalStride + trailingOffset];
                newIm[offsetSub + j] = this._im.data[offsetX + i * finalStride + trailingOffset];
                j++;
            }
        } else {
            while (iter.hasNext()) {
                this._getSubTensorC(iters, newRe, newIm, maxLevel,
                    stridesX, stridesSub, finalStride, trailingOffset, level + 1,
                    offsetX + stridesX[level] * iter.next(), offsetSub);
                offsetSub += stridesSub[level];
            }
        }
    }

    
    private _parseIndexIterDefs(args: ArrayLike<any>): IIndexIterator[] {
        let iters: IIndexIterator[] = [];
        for (let i = 0;i < args.length;i++) {
            // ik can be a integer, string, array/tensor of indices, logic
            // tensor as a mask. We unify them into list of index iterators.
            let ind = args[i];
            if (ind instanceof Tensor) {
                if (ind.hasNonZeroComplexStorage()) {
                    throw new Error('Indices cannot be complex');
                }
                if (ind.dtype === DType.LOGIC) {
                    // Logic mask vector.
                    if (ind.ndim !== 1 || ind.size !== this._shape[i]) {
                        throw new Error(`1D logic tensor of size ${this._shape[i]} expected for dimension ${i+1}.`);
                    }
                    iters.push(new ArrayBasedIndexIterator(DataHelper.findReal(ind.realData)));
                } else {
                    if (ind.ndim > 1) {
                        throw new Error(`1D vector of indices expected form for dimension ${i+1}.`
                            + `Got a ${ShapeHelper.shapeToString(ind.shape)} tensor.`);
                    }
                    iters.push(new ArrayBasedIndexIterator(this._convertSignedIndices(ind.realData, i)));
                }
            } else if (Array.isArray(ind)) {
                // no nested arrays allowed here
                if (ind.length > 0 && Array.isArray(ind[0])) {
                    throw new Error(`Expecting a 1D array of indices for ${i+1}. Got a nested array.`);
                }
                iters.push(new ArrayBasedIndexIterator(this._convertSignedIndices(<ArrayLike<number>>ind, i)));
            } else if (typeof ind === 'string') {
                iters.push(this._parseSlicingString(ind, i));
            } else {
                if (ind < 0) {
                    // convert negative index to positive before further checks
                    ind += this._shape[i];
                }
                if ((ind | 0) !== ind) {
                    throw new Error(`The index for dimension ${i+1} should be a integer.`);
                }
                if (ind >= this._shape[i]) {
                    throw new Error(`The index for dimension ${i+1} is out of bounds.`);
                }
                iters.push(new ConstantIndexIterator(<number>ind));
            }
        }
        return iters;
    }

    /**
     * Gets the index iteration definition for a string that represents a
     * slicing operation (in the syntax of Python).
     * @param str A string that represents a slicing operation in the following
     *  format: 'start:stop:step', where 'step' is optional.
     * @param dim The dimension to which slicing operation is applied.
     */
    private _parseSlicingString(str: string, dim?: number): IIndexIterator {
        // we adapt the syntax of Python
        let splits = str.trim().split(':');
        let start: number, stop: number, step: number;
        let max = dim == undefined ? this.size : this._shape[dim];
        switch (splits.length) {
            case 1:
                start = parseFloat(splits[0]);
                if (start < 0) start += max;
                this._checkIndex(start, dim);
                return new ConstantIndexIterator(start);
            case 2:
            case 3:
                if (splits.length === 3 && splits[2].length > 0) {
                    step = parseFloat(splits[2]);
                    if (Math.floor(step) !== step) {
                        throw new Error('Step must be an integer.');
                    }
                    if (step === 0) {
                        throw new Error('Step cannot be zero.');
                    }
                } else {
                    step = 1;
                }
                start = splits[0].length === 0
                    ? (step > 0 ? 0 : max - 1)
                    : parseFloat(splits[0]);
                if (start < 0) start += max;
                this._checkIndex(start, dim);
                if (splits[1].length === 0) {
                    if (step > 0) {
                        stop = max;
                    } else {
                        stop = -1; // this is a special one
                    }
                } else {
                    stop = parseFloat(splits[1]);
                    // unlike start, stop is allowed to be max
                    if (stop < 0) stop += max;
                    if ((stop | 0) !== stop) {
                        throw new Error('Index must be an integer.');
                    }
                    if (stop < 0 || stop > max) {
                        throw new Error(`Index ${stop} is out of bounds for dimension ${dim}.`);
                    }
                }
                return step > 0
                    ? new RangedIndexIterator(start, stop, step)
                    : new ReversedRangedIndexIterator(start, stop, -step);
            default:
                throw new Error(`Invalid slicing definition '${str}'.`);
        }
    }

    /**
     * Converts signed indices to the corresponding iteration definition.
     * Note: data copy only occurs when any of the indices is negative.
     * @param indices An array like object of indices.
     * @param dim (Optional) If specified, the given indices are for the
     *  specified dimension. Otherwise, the given indices are for flat indexing.
     */
    private _convertSignedIndices(indices: ArrayLike<number>, dim?: number): ArrayLike<number> {
        let ret: number[] | undefined = undefined;
        let i: number, cur: number;
        let max = dim == undefined ? this.size : this._shape[dim];
        for (i = 0;i < indices.length;i++) {
            if (indices[i] < 0) {
                // we need to make a copy
                ret = new Array<number>(indices.length);
                for (let j = 0;j < i;j++) {
                    ret[j] = indices[j];
                }
                break;
            }
            this._checkIndex(indices[i], dim);
        }
        for (;i < indices.length;i++) {
            cur = indices[i] < 0 ? max + indices[i] : indices[i];
            this._checkIndex(cur, dim);
            (<number[]>ret)[i] = cur;
        }
        return ret || indices;
    }

    /**
     * Checks if the given index is valid (is an integer and within bounds).
     * @param index Index to be checked.
     * @param dim (Optional) Dimension number. If specified, will check against
     *      the specified dimension. If omitted, will assume flat indexing over
     *      all the elements.
     */
    private _checkIndex(index: number, dim?: number): void {
        if ((index | 0) !== index) {
            throw new Error('Index must be an integer.');
        }
        let max = dim == undefined ? this.size : this._shape[dim];
        if (index < 0 || index >= max) {
            if (dim == undefined) {
                throw new Error(`Index ${index} is out of bounds.`);
            } else {
                throw new Error(`Index ${index} is out of bounds for dimension ${dim}.`);
            }
        }
    }

    private static _inferShapeFromIters(iters: IIndexIterator[]): [number[], number] {
        let shape: number[] = new Array(iters.length);
        let size = 1;
        for (let i = 0;i < iters.length;i++) {
            shape[i] = iters[i].count;
            size *= shape[i];
        }
        return [shape, size];
    }

    private _handleSetGet(args: IArguments, doSet: false, keepDims: boolean): Tensor | Scalar;
    private _handleSetGet(args: IArguments, doSet: true): void;
    private _handleSetGet(args: IArguments, doSet: boolean, keepDims: boolean = false): Tensor | Scalar | void {
        let arg0 = args[0];
        let tmp: Tensor | Scalar;
        let originalShape: number[];
        if (args.length === (doSet ? 2 : 1)) {
            // Cases 1, 2, 3
            if (arg0 instanceof Tensor) {
                // Case 1, 2 where indices are specified by a Tensor
                if (arg0.dtype === DType.LOGIC) {
                    // mask indexing
                    if (!ShapeHelper.compareShape(arg0._shape, this._shape)) {
                        throw new Error('When using a logic tensor for indexing, its shape must match that of the tensor being indexed.')
                    }
                    // no need to check indices here
                    if (doSet) {
                        this._setBatch(DataHelper.findReal(arg0.realData), args[1]);
                    } else {
                        // Because masked locations can be arbitrary, we cannot
                        // preserver the original shape and have to return a 
                        // 1D vector.
                        return this._getBatch(DataHelper.findReal(arg0.realData), keepDims);
                    }
                } else {
                    // indexing with signed integers
                    if (arg0.hasNonZeroComplexStorage()) {
                        throw new Error('Complex tensor cannot be used for indexing.');
                    }
                    if (doSet) {
                        // For tensors, we access it flattened version by directly
                        // reading realData.
                        this._setBatch(this._convertSignedIndices(arg0.realData), args[1]);
                    } else {
                        if (arg0.ndim === 1 || arg0.isScalar()) {
                            return this._getBatch(this._convertSignedIndices(arg0.realData), keepDims);
                        } else {
                            // we want to preserve the original shape for the
                            // get case
                            originalShape = arg0.shape;
                            tmp = <Tensor>this._getBatch(this._convertSignedIndices(arg0.realData), keepDims);
                            tmp._shape = originalShape;
                            tmp._updateStridesAndCalculator();
                            return tmp;
                        }
                    }
                }
            } else if (Array.isArray(arg0)) {
                // Case 1 where indices are specified by an Array
                if (doSet) {
                    if (Array.isArray(arg0)) {
                        // nested array detected, flatten it first and then
                        // index
                        originalShape = ShapeHelper.inferShapeFromArray(arg0);
                        ShapeHelper.validateArrayShape(arg0, originalShape);
                        this._setBatch(this._convertSignedNestedIndices(arg0, originalShape), args[1]);
                    } else {
                        this._setBatch(this._convertSignedIndices(arg0), args[1]);
                    }
                } else {
                    if (Array.isArray(arg0[0])) {
                        // nested array detected
                        originalShape = ShapeHelper.inferShapeFromArray(arg0);
                        ShapeHelper.validateArrayShape(arg0, originalShape);
                        tmp = this._getBatch(this._convertSignedNestedIndices(arg0, originalShape), keepDims);
                        // we want to preserve the original shape here
                        if (tmp instanceof Tensor) {
                            tmp._shape = originalShape;
                            tmp._updateStridesAndCalculator();
                        }
                        return tmp;
                    } else {
                        return this._getBatch(this._convertSignedIndices(arg0), keepDims);
                    }
                }
            } else if (arg0 instanceof Function) {
                // Case 3
                let indices = this.hasComplexStorage()
                    ? DataHelper.findWithCallbackComplex(this.realData, this.imagData, arg0)
                    : DataHelper.findWithCallbackReal(this.realData, arg0);
                // no need to check indices here
                if (doSet) {
                    this._setBatch(indices, args[1]);
                } else {
                    return this._getBatch(indices, keepDims);
                }
            } else if (typeof arg0 === 'string') {
                // Case 1 where indices are specified by a string
                if (doSet) {
                    this._setSubTensor([this._parseSlicingString(arg0)], args[1]);
                } else {
                    return this._getSubTensor([this._parseSlicingString(arg0)], keepDims);
                }
            } else {
                // Case 1 where index is just one number
                // Delegate to setEl()/getEl()
                let i = <number>arg0;
                if (i < 0) {
                    i += this.size;
                }
                if ((i | 0) !== i) {
                    throw new Error('Index should be an integer.');
                }
                if (doSet) {
                    let v: any = args[1];
                    if (v instanceof Tensor) {
                        // expecting a scalar tensor
                        if (v.size !== 1) {
                            throw new Error('Scalar tensor expected.');
                        }
                        this.setEl(i, v.getEl(0));
                    } else if (Array.isArray(v)) {
                        // expecting a one-element array
                        // we do not do further checks here
                        if (v.length !== 1) {
                            throw new Error('One-element numerical array expected.')
                        }
                        this.setEl(i, v[0]);
                    } else {
                        // expecting number | ComplexNumber
                        this.setEl(i, v);
                    }
                } else {
                    return this.getEl(i);
                }
            }
        } else {
            // (i1, i2, ...)
            if (doSet) {
                this._setSubTensor(this._parseIndexIterDefs(
                    Array.prototype.slice.call(args, 0, args.length - 1)), args[args.length - 1]);
            } else {
                return this._getSubTensor(this._parseIndexIterDefs(args), keepDims);
            }
        }
    }

    private _convertSignedNestedIndices(arr: any[], shape: number[]): number[] {
        let indices = new Array(ShapeHelper.getSizeFromShape(shape));
        let strides = ShapeHelper.computeStrides(shape);
        let _doParse = (arr: any[], level: number, offset: number): void => {
            if (level === shape.length - 1) {
                for (let i = 0;i < shape[level];i++) {
                    let cur = arr[i];
                    if (cur < 0) cur += this.size;
                    this._checkIndex(cur);
                    indices[offset + i] = cur;
                }
            } else {
                for (let i = 0;i < shape[level];i++) {
                    _doParse(arr[i], level + 1, offset);
                    offset += strides[level];
                }
            }
        };
        _doParse(arr, 0, 0);
        return indices;
    }

    /**
     * Sets the element at the specified index.
     */
    public setEl(...args: any[]);
    public setEl(): void {
        if (arguments.length < 2) {
            throw new Error('Too few arguments.');
        }
        if (arguments.length > this._shape.length + 1) {
            throw new Error('Too many arguments.')
        }
        this.ensureUnsharedLocalStorage();
        let offset = arguments.length > 2 ? this._offsetCalculator(arguments, this._strides) : <number>arguments[0],
            v = arguments[arguments.length - 1];
        if (offset < 0 || offset >= this.size) {
            throw new Error('Index out of bounds.');
        }
        if (v instanceof ComplexNumber) {
            if (this.dtype === DType.LOGIC) {
                throw new Error('Cannot store complex values in a logic tensor.');
            }
            this._re.data[offset] = v.re;
            if (v.im !== 0) {
                this.ensureComplexStorage();
                this._im.data[offset] = v.im;
            }
        } else {
            // we force convert v to number here
            let nv = typeof v === 'number' ? v : Number(v);
            if (this.dtype === DType.LOGIC) {
                this._re.setAsLogicAtUnchecked(offset, nv);
            } else {
                this._re.data[offset] = nv;
            }
        }
    }

    /**
     * Retrieves the element at the specified index.
     */
    public getEl(...args: any[]): Scalar;
    public getEl(): Scalar {
        let offset: number;
        if (arguments.length === 1) {
            offset = arguments[0];
        } else if (arguments.length === this._shape.length) {
            offset = this._offsetCalculator(arguments, this._strides);
        } else {
            throw new Error('Incorrect number of arguments.');
        }
        this._checkIndex(offset);
        return this.hasComplexStorage()
            ? new ComplexNumber(this._re.data[offset], this._im.data[offset])
            : this._re.data[offset];
    }

    /**
     * Ensures the underlying storage is a local copy.
     */
    public ensureUnsharedLocalStorage(): Tensor {
        if (this._re.refCount > 1) {
            this._re.refCount--;
            this._re = this._re.dataCopy();
            this._re.refCount++;
        }
        if (this.hasComplexStorage()) {
            if (this._im.refCount > 1) {
                this._im.refCount--;
                this._im = this._im.dataCopy();
                this._im.refCount++;
            }
        }
        return this;
    }

    /**
     * Ensures the underlying complex storage is not empty.
     */
    public ensureComplexStorage(): Tensor {
        if (!this.hasComplexStorage()) {
            if (this.dtype !== DType.LOGIC) {
                this._im = TensorStorage.create(this._re.data.length, this._re.dtype);
                this._im.refCount++;
            } else {
                throw new Error('Logic tensors cannot have a complex storage.')
            }
        }
        return this;
    }

    /**
     * Retrieves the real part.
     */
    public real(): Tensor {
        // Copy the shape array so that the modification of this tensor's shape
        // will not affected the new tensor's shape.
        return new Tensor(this._re, TensorStorage.Empty, this.shape);
    }

    /**
     * Retrieves the imaginary part.
     */
    public imag(): Tensor {
        if (this.hasComplexStorage()) {
            // Copy the shape array so that the modification of this tensor's shape
            // will not affected the new tensor's shape.
            return new Tensor(this._im, TensorStorage.Empty, this.shape);
        } else {
            return Tensor.zeros(this._shape, this.dtype);
        }
    }

    /**
     * Remove the imaginary part from this tensor.
     * Unlike real(), this is an in-place operation and the
     * **entire imaginary part will be discarded**.
     */
    public trimImaginaryPart(): Tensor {
        if (this.hasComplexStorage()) {
            this._im.refCount--;
            this._im = TensorStorage.Empty;
        }
        return this;
    }

    /**
     * Creates a array by apply the given map function to every element
     * in this tensor (ignoring the shape of this tensor).
     * @param f Map function.
     * @param dtype Data type of the output tensor.
     */
    public map<T>(f: (re: number, im: number) => T): T[] {
        let n = this.size;
        let result = new Array<T>(n);
        if (this.hasComplexStorage()) {
            for (let i = 0;i < n;i++) {
                result[i] = f(this._re.data[i], this._im.data[i]);
            }
        } else {
            for (let i = 0;i < n;i++) {
                result[i] = f(this._re.data[i], 0);
            }
        }
        return result;
    }

    /**
     * Applied the reduce function to each element in this function (ignoring
     * the shape of this tensor).
     * @param f Reduce function.
     * @param initialValue Initial value.
     */
    public reduce<T>(f: (re: number, im: number, result: T) => T, initialValue: T): T {
        let result = initialValue;
        if (this.hasComplexStorage()) {
            for (let i = 0, n = this.size;i < n;i++) {
                result = f(this._re.data[i], this._im.data[i], result);
            }
        } else {
            for (let i = 0, n = this.size;i < n;i++) {
                result = f(this._re.data[i], 0, result);
            }
        }
        return result;
    }

    /**
     * Prepends a new axis to this tensor.
     * This method modifies the shape in-place.
     */
    public prependAxis(): Tensor {
        this._shape.unshift(1);
        this._updateStridesAndCalculator();
        return this;
    }

    /**
     * Appends a new axis to this tensor.
     * This method modifies the shape in-place.
     */
    public appendAxis(): Tensor {
        this._shape.push(1);
        this._updateStridesAndCalculator();
        return this;
    }

    /**
     * Reshapes the tensor in-place.
     * @param newShape New shape.
     */
    public reshape(newShape: number[]): Tensor {
        this._shape = this._calculateNewShape(newShape);
        this._updateStridesAndCalculator();
        return this;
    }

    /**
     * Retrieves a reshaped copy.
     * @param newShape New shape of the returned copy.
     */
    public getReshapedCopy(newShape: number[]): Tensor {
        newShape = this._calculateNewShape(newShape);
        return new Tensor(this._re, this._im, newShape);
    }

    /**
     * Calculates the new shape for reshaping.
     * @param newShape New shape. If no changes are made, a copy of the original
     *                 shape array is returned.
     */
    private _calculateNewShape(newShape: number[]): number[] {
        // We make a copy here because it may be used elsewhere.
        newShape = newShape.slice();
        // Check if -1 exists.
        let idxM1 = -1,
            ns = 1;
        for (let i = 0;i < newShape.length;i++) {
            if ((newShape[i] | 0) !== newShape[i]) {
                throw new Error('Expecting a 32-bit integer.');
            }
            if (newShape[i] === -1) {
                if (idxM1 >= 0) {
                    throw new Error('Shape can only contain one unknown dimension.');
                } else {
                    idxM1 = i;
                }
            } else {
                ns *= newShape[i];
            }
        }
        if (idxM1 >= 0) {
            // -1 exists, infer the length of this dimension
            newShape[idxM1] = this.size / ns;
            if (!isFinite(newShape[idxM1]) || (newShape[idxM1] | 0) !== newShape[idxM1]) {
                throw new Error('The inferred length of the unknown dimension is not an integer.');
            }
        } else {
            ShapeHelper.validateShape(newShape);
            if (ns !== this.size) {
                throw new Error('The number of elements cannot change after reshaping.');
            }
        }
        return newShape;
    }

    /**
     * Retrieves a copy of this tensor.
     * @param copyStorageImmediately If set to true, the underlying tensor
     *      storage will be duplicated immediately. Default value is false.
     */
    public copy(copyStorageImmediately: boolean = false): Tensor {
        // We copy the shape array so that the modification of this tensor's
        // shape will not affected the new tensor's shape.
        if (copyStorageImmediately) {
            
            return new Tensor(this._re.dataCopy(), 
                this._im !== TensorStorage.Empty ? this._im.dataCopy() : TensorStorage.Empty,
                this.shape);
        } else {
            return new Tensor(this._re, this._im, this.shape);
        }
    }

    /**
     * Retrieves a new tensor by converting each element to the specified data
     * type. Loss of precision will occur during type conversion.
     * Note: you cannot convert complex values/nan values to logical values.
     * @param dtype Data type.
     * @param alwaysCopy By default, if the specified dtype is the same as the
     *  original one, a reference copy will be returned. Set this to true to
     *  ensure that a deep copy is always returned.
     */
    public asType(dtype: DType, alwaysCopy: boolean = false): Tensor {
        if (dtype === this._re.dtype) {
            // Just make a quick copy here.
            return this.copy(alwaysCopy);
        } else if (dtype === DType.LOGIC) {
            if (this.hasComplexStorage()) {
                throw new Error('Cannot convert a complex tensor to a logic tensor.');
            }
            // Remember to copy the shape.
            return new Tensor(this._re.copyAsType(DType.LOGIC), TensorStorage.Empty, this.shape);
        } else {
            let re = this._re.copyAsType(dtype),
                im = this._im === TensorStorage.Empty ? TensorStorage.Empty : this._im.copyAsType(dtype);
            // Remember to copy the shape.                
            return new Tensor(re, im, this.shape);
        }
    }

    /**
     * Converts this tensor to a JavaScript array, only including the real part.
     * @returns The converted array of the real part.
     */
    public toArray(realOnly: true): any[];
    /**
     * Converts this tensor to a JavaScript array, including both the real and
     * imaginary part.
     * @returns A 2-item tuple where the first item is the converted array
     *  of the real part, and the second item is the converted array of the
     *  imaginary part. If the tensor does not have the imaginary part, the
     *  second item will be an empty array.
     */
    public toArray(realOnly: false): [any[], any[]];
    public toArray(realOnly: boolean = false): any[] | [any[], any[]] {
        let reArr = Tensor._toArray(this._re.data, this._shape, this._strides, 0, 0);
        if (realOnly) {
            return reArr;
        } else {
            let imArr = this.hasComplexStorage()
                ? Tensor._toArray(this._im.data, this._shape, this._strides, 0, 0)
                : [];
            return [reArr, imArr];
        }
    }

    private static _toArray(storage: ArrayLike<number>, shape: number[],
                            strides: number[], level: number, offset: number): any[] {
        let arr = new Array(shape[level]);
        if (level === shape.length - 1) {
            for (let i = 0;i < shape[level];i++) {
                arr[i] = storage[offset + i];
            }
        } else {
            for (let i = 0;i < shape[level];i++) {
                arr[i] = Tensor._toArray(storage, shape, strides, level + 1, offset);
                offset += strides[level];
            }
        }
        return arr;
    }

    // TODO: simplify, too messy right now
    public toString(): string {
        const MAX_C = 3;
        const MAX_R = 4;
        let str: string;
        switch (this.ndim) {
            case 1:
                str = `[${this._elementsToString(0, this.size, 1, MAX_C)}]`;
                break;
            case 2:
                str = '[';
                if (this._shape[0] <= MAX_R + MAX_R) {
                    for (let i = 0;i < this._shape[0];i++) {
                        str += `[${this._elementsToString(i * this._shape[1], this._shape[1], 1, MAX_C)}]`;
                        if (i !== this._shape[0] - 1) {
                            str += ',\n ';
                        }
                    }
                } else {
                    for (let i = 0;i < MAX_R;i++) {
                        str += `[${this._elementsToString(i * this._shape[1], this._shape[1], 1, MAX_C)}],\n `;
                    }
                    str += '...\n ';
                    for (let i = this._shape[0] - MAX_R;i < this._shape[0];i++) {
                        str += `[${this._elementsToString(i * this._shape[1], this._shape[1], 1, MAX_C)}]`;
                        if (i !== this._shape[0] - 1) {
                            str += ',\n ';
                        }
                    }
                }
                str += ']';
                break;
            default:
                let strComplex = this.hasComplexStorage() ? 'complex ' : '';
                str = `[${this._shape.join('x')} ${DTypeHelper.dTypeToString(this.dtype)} ${strComplex} tensor]`;
        }
        return str;
    }

    private _elementsToString(offset: number, count: number, stride: number = 1, max: number = 2): string {
        let parts: string[] = [];
        if (count <= max + max) {
            for (let i = 0;i < count;i++) {
                parts.push(this._elementToString(offset + i * stride));
            }
        } else {
            for (let i = 0;i < max;i++) {
                parts.push(this._elementToString(offset + i * stride));
            }
            parts.push('...');
            for (let i = count - max;i < count;i++) {
                parts.push(this._elementToString(offset + i * stride));
            }
        }
        return parts.join(', ');
    }

    private _elementToString(offset: number): string {
        let re = this._re.data[offset];
        if (this.dtype === DType.LOGIC) {
            return re !== 0 ? ' true': 'false';
        } else {
            let str = re >= 0 ? ' ' : '';
            if (this.hasComplexStorage()) {
                let im = this._im.data[offset];
                if (this.dtype === DType.INT32) {
                    str += `${re.toString()} ${im >= 0 ? '+': '-'}${Math.abs(im).toString()}j`;
                } else {
                    str += `${re.toExponential(4)} ${im >= 0 ? '+': '-'} ${Math.abs(im).toExponential(4)}j`;
                }
            } else {
                if (this.dtype === DType.INT32) {
                    str += re.toString();
                } else {
                    str += re.toExponential(4);
                }
            }
            return str;
        }
    }

}
