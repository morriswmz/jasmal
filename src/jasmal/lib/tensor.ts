import { TensorStorage } from './storage';
import { DType, DTypeHelper } from './dtype';
import { ComplexNumber } from './complexNumber';
import { Scalar, OpInputInfo, OpInputType, OpInput, TypedArray, DataBlock } from './commonTypes';
import { ShapeHelper } from './helper/shapeHelper';
import { DataHelper } from './helper/dataHelper';
import { ObjectHelper } from './helper/objHelper';

type OffsetCalculator = (indices: ArrayLike<number>, strides: number[]) => number;

class OffsetCalculatorFactory {

    private static _cached: Array<OffsetCalculator> = [];

    public static create(dim: number): OffsetCalculator {
        // To comply with plain JavaScript arrays, we use row-major ordering.
        if (!OffsetCalculatorFactory._cached[dim]) {
            let funcBody = '\'use strict\'; return ';
            for (let i = 0; i < dim - 1;i++) {
                funcBody += `(indices[${i}]|0) * (strides[${i}]|0) + `
            }
            funcBody += `(indices[${dim - 1}]|0);`;
            OffsetCalculatorFactory._cached[dim] = <OffsetCalculator>(new Function('indices', 'strides', funcBody));
        }
        return OffsetCalculatorFactory._cached[dim];
    }

}

/**
 * Describes the type of iteration along some axis.
 */
const enum IndexIteratorType {
    Constant,
    Range,
    List
}

/**
 * Defines an iteration along some axis.
 */
interface IndexIteratorDefinition {
    type: IndexIteratorType;
    start?: number; // available only when type is IndexIterationType.Range
    stop?: number; // available only when type is IndexIterationType.Range
    step?: number; // available only when type is IndexIterationType.Range
    indices?: ArrayLike<number>; // available only when type is IndexIterationType.List
    index?: number; // available only when type is IndexIterationType.Constant
}

/**
 * Wraps information for indexing.
 */
interface IndexIteratorInfo {
    definitions: IndexIteratorDefinition[],
    areAllConstantType: boolean;
}

export class Tensor {

    private _re: TensorStorage;
    private _im: TensorStorage;
    private _shape: number[];
    private _strides: number[];
    private _offsetCalculator: OffsetCalculator;

    /**
     * Internal constructor for Tensor objects.
     * Note: the refCount of the input parameters will be automatically
     *       increased.
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
        return new Tensor(re._re, im._re, re._shape);
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
        let shape = Tensor._getShapeFromArray(re);
        Tensor._validateArrayShape(re, shape, 0);
        if (isComplex) Tensor._validateArrayShape(<any[] | TypedArray>im, shape, 0);
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

    private static _getShapeFromArray(arr: any[] | TypedArray): number[] {
        let shape: number[] = [];
        let curEl: any = arr;
        while (Array.isArray(curEl) || ObjectHelper.isTypedArray(curEl)) {
            if (curEl.length === 0) {
                throw new Error('Array cannot be empty.');
            }
            shape.push(curEl.length);
            curEl = curEl[0];
        }
        return shape;
    }

    private static _validateArrayShape(arr: any[] | TypedArray, shape: number[], level: number): void {
        if (arr.length !== shape[level]) {
            throw new Error('The structure of the input array does not match that of a tensor.');
        }
        if (level < shape.length - 1) {
            for (let i = 0;i < arr.length;i++) {
                if (Array.isArray(arr[i])) {
                    Tensor._validateArrayShape(arr[i], shape, level + 1);
                } else {
                    throw new Error('Cannot have mixed array and non-array elements at the same level.');
                }
            }
        }
    }

    /**
     * Creates a new tensor filled with zeros.
     * @param shape Shape of the tensor.
     * @param dtype Data type.
     */
    public static zeros(shape: number[], dtype: DType = DType.FLOAT64): Tensor {
        ShapeHelper.validateShape(shape);
        let re = TensorStorage.create(ShapeHelper.getSizeFromShape(shape), dtype);
        return new Tensor(re, TensorStorage.Empty, shape);
    }

    /**
     * Creates a new tensor filled with ones.
     * @param shape Shape of the tensor.
     * @param dtype Data type.
     */
    public static ones(shape: number[], dtype: DType = DType.FLOAT64): Tensor {
        ShapeHelper.validateShape(shape);
        let re = TensorStorage.create(ShapeHelper.getSizeFromShape(shape), dtype);
        for (let i = 0;i < re.data.length;i++) {
            re.data[i] = 1;
        }
        return new Tensor(re, TensorStorage.Empty, shape);
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
    public static toTensor(x: number | ComplexNumber | any[]): Tensor {
        if (Array.isArray(x)) {
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
            originalShape = value.shape;
            originalType = OpInputType.Tensor;
            originalDType = value.dtype;
        } else if (Array.isArray(value) || ObjectHelper.isTypedArray(value)) {
            if (value.length === 0) {
                throw new Array('Value cannot be an empty array.');
            }
            if (Array.isArray(value[0])) {
                // Detected a nested array, convert to Tensor.
                // This is not very efficient because we need to copy every
                // elements, but we have to do this because flat indexing is
                // used internally.
                let tmp = Tensor.analyzeOpInput(Tensor.fromArray(value));
                tmp.originalType = OpInputType.Array;
                return tmp;
            } else {
                // plain array
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
     */
    public get shape(): number[] {
        return this._shape.slice();
    }

    /**
     * Returns the strides of this tensor.
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

    private _setBatch(iterDef: IndexIteratorDefinition, value: number | ComplexNumber | number[] | Tensor): void {
        // _setBatch can be implemented by first flatten this tensor, and then
        // call _setSubTensor
        this._setSubTensor({
            definitions: [iterDef],
            areAllConstantType: false
        }, value);
    }
    
    private _setSubTensor(iterInfo: IndexIteratorInfo, value: number | ComplexNumber | any[] | Tensor): void {
        let iterDefs = iterInfo.definitions;
        let ndim: number;
        let strides: number[];
        let trailingOffset: number = 0;
        let finalStride: number = 1;
        // determine the shape of the sub tensor
        let [shapeSub, sizeSub] = Tensor._inferShapeFromIterDefs(iterDefs);
        if (sizeSub === 0) {
            throw new Error('The number of indices cannot be zero. No elements to update.')
        }
        let maxLevel = iterDefs.length - 1;
        if (iterDefs.length === 1) {
            ndim = 1;
            strides = [this.size];
        } else {
            ndim = this.ndim;
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
                switch (iterDefs[k].type) {
                    case IndexIteratorType.Constant:
                        trailingOffset += strides[k] * <number>iterDefs[k].index;
                        break;
                    case IndexIteratorType.List:
                        trailingOffset += strides[k] * (<ArrayLike<number>>iterDefs[k].indices)[0];
                        break;
                    case IndexIteratorType.Range:
                        trailingOffset += strides[k] * <number>iterDefs[k].start;
                        break;
                    default:
                        throw new Error('Should never reach here.');
                }
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
                this._setSubTensorC1(iterDefs, newRe, newIm, maxLevel, strides,
                    finalStride, trailingOffset, 0, 0);
            } else {
                this._setSubTensorR1(iterDefs, newRe, maxLevel, strides,
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
                this._setSubTensorCN(iterDefs, newRe, newIm, maxLevel, strides,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            } else {
                this._setSubTensorRN(iterDefs, newRe, maxLevel, strides,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            }
        }
    }

    private _setSubTensorR1(iterDefs: IndexIteratorDefinition[], newRe: number,
                            maxLevel: number, strides: number[], finalStride:number,
                            trailingOffset: number, level: number, offsetX: number): void {
        let ind = iterDefs[level];
        if (level === maxLevel) {
            // last level
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._re.data[offsetX + <number>ind.index * finalStride + trailingOffset] = newRe;
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe;
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe;
                        }
                    }
                    break;                                
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._re.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset] = newRe;
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        } else {
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._setSubTensorR1(iterDefs, newRe, maxLevel, strides,
                        finalStride, trailingOffset, level + 1,
                        offsetX + strides[level] * <number>ind.index);
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorR1(iterDefs, newRe, maxLevel, strides,
                                finalStride, trailingOffset, level + 1,
                                offsetX + strides[level] * i);
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorR1(iterDefs, newRe, maxLevel, strides,
                                finalStride, trailingOffset, level + 1,
                                offsetX + strides[level] * i);
                        }
                    }
                    break;
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._setSubTensorR1(iterDefs, newRe, maxLevel, strides,
                                finalStride, trailingOffset, level + 1,
                                offsetX + strides[level] * (<ArrayLike<number>>ind.indices)[i]);
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        }
    }

    private _setSubTensorC1(iterDefs: IndexIteratorDefinition[],
                            newRe: number, newIm: number, maxLevel: number,      
                            strides: number[], finalStride: number,
                            trailingOffset: number, level: number, offsetX: number): void {
        let ind = iterDefs[level];
        if (level === maxLevel) {
            // last level
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._re.data[offsetX + <number>ind.index * finalStride + trailingOffset] = newRe;
                    this._im.data[offsetX + <number>ind.index * finalStride + trailingOffset] = newIm;
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe;
                            this._im.data[offsetX + i * finalStride + trailingOffset] = newIm;
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe;
                            this._im.data[offsetX + i * finalStride + trailingOffset] = newIm;
                        }
                    }
                    break;                                
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._re.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset] = newRe;
                        this._im.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset] = newIm;
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        } else {
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._setSubTensorC1(iterDefs, newRe, newIm, maxLevel,
                        strides, finalStride, trailingOffset, level + 1,
                        offsetX + strides[level] * <number>ind.index);
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorC1(iterDefs, newRe, newIm, maxLevel,
                                strides, finalStride, trailingOffset,
                                level + 1, offsetX + strides[level] * i);
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorC1(iterDefs, newRe, newIm, maxLevel,
                                strides, finalStride, trailingOffset,
                                level + 1, offsetX + strides[level] * i);
                        }
                    }
                    break;
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._setSubTensorC1(iterDefs, newRe, newIm, maxLevel,
                            strides, finalStride, trailingOffset,
                            level + 1, offsetX + strides[level] * (<ArrayLike<number>>ind.indices)[i]);
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        }
    };

    private _setSubTensorRN(iterDefs: IndexIteratorDefinition[],
                            newRe: ArrayLike<number>, maxLevel: number,
                            strides: number[], stridesSub: number[],
                            finalStride: number, trailingOffset: number, level: number,
                            offsetX: number, offsetY: number): void {
        let ind = iterDefs[level];
        if (level === maxLevel) {
            // last level
            let j;
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._re.data[offsetX + <number>ind.index * finalStride + trailingOffset] = 
                        newRe[offsetY + <number>ind.index];
                    break;
                case IndexIteratorType.Range:
                    j = 0;
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe[offsetY + j];
                            j++;
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe[offsetY + j];
                            j++;
                        }
                    }
                    break;                                
                case IndexIteratorType.List:
                    j = 0;
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._re.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset] =
                            newRe[offsetY + j];
                        j++;
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        } else {
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._setSubTensorRN(iterDefs, newRe, maxLevel, strides,
                        stridesSub, finalStride, trailingOffset, level + 1,
                        offsetX + strides[level] * <number>ind.index, offsetY);
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorRN(iterDefs, newRe, maxLevel, strides,
                                stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + strides[level] * i, offsetY);
                            offsetY += stridesSub[level];
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorRN(iterDefs, newRe, maxLevel, strides,
                                stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + strides[level] * i, offsetY);
                            offsetY += stridesSub[level];
                        }
                    }
                    break;
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._setSubTensorRN(iterDefs, newRe, maxLevel, strides,
                            stridesSub, finalStride, trailingOffset, level + 1,
                            offsetX + strides[level] * (<ArrayLike<number>>ind.indices)[i], offsetY);
                        offsetY += stridesSub[level];
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        }
    }

    private _setSubTensorCN(iterDefs: IndexIteratorDefinition[],
                            newRe: ArrayLike<number>, newIm: ArrayLike<number>,
                            maxLevel: number, strides: number[], stridesSub: number[],
                            finalStride: number, trailingOffset: number, level: number,
                            offsetX: number, offsetY: number): void {
        let ind = iterDefs[level];
        if (level === maxLevel) {
            // last level
            let j;
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._re.data[offsetX + <number>ind.index * finalStride + trailingOffset] = 
                        newRe[offsetY + <number>ind.index];
                    this._im.data[offsetX + <number>ind.index * finalStride + trailingOffset] = 
                        newIm[offsetY + <number>ind.index];
                    break;
                case IndexIteratorType.Range:
                    j = 0;
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe[offsetY + j];
                            this._im.data[offsetX + i * finalStride + trailingOffset] = newIm[offsetY + j];
                            j++;
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._re.data[offsetX + i * finalStride + trailingOffset] = newRe[offsetY + j];
                            this._im.data[offsetX + i * finalStride + trailingOffset] = newIm[offsetY + j];
                            j++;
                        }
                    }
                    break;                                
                case IndexIteratorType.List:
                    j = 0;
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._re.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset] =
                            newRe[offsetY + j];
                        this._im.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset] =
                            newIm[offsetY + j];
                        j++;
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        } else {
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._setSubTensorCN(iterDefs, newRe, newIm, maxLevel,
                        strides, stridesSub, finalStride, trailingOffset, level + 1,
                        offsetX + strides[level] * <number>ind.index, offsetY);
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorCN(iterDefs, newRe, newIm, maxLevel,
                                strides, stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + strides[level] * i, offsetY);
                            offsetY += stridesSub[level];
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._setSubTensorCN(iterDefs, newRe, newIm, maxLevel,
                                strides, stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + strides[level] * i, offsetY);
                            offsetY += stridesSub[level];
                        }
                    }
                    break;
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._setSubTensorCN(iterDefs, newRe, newIm, maxLevel,
                            strides, stridesSub, finalStride, trailingOffset, level + 1,
                            offsetX + strides[level] * (<ArrayLike<number>>ind.indices)[i], offsetY);
                        offsetY += stridesSub[level];
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
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
     * By default, singleton dimension will be removed from the returned sub-
     * tensor. This in most cases will save many ugly squeeze() calls. However,
     * if you wish to prevent this behavior, set `keepDims` to false.
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

    private _getBatch(iterDefs: IndexIteratorDefinition, keepDims: boolean): Tensor | Scalar {
        return this._getSubTensor({
            definitions: [iterDefs],
            areAllConstantType: false
        }, keepDims);
    }

    private _getSubTensor(iterInfo: IndexIteratorInfo, keepDims: boolean): Tensor | Scalar {
        let iterDefs = iterInfo.definitions;
        let stridesX: number[];
        // determine the shape of the sub tensor
        let [shapeSub, sizeSub] = Tensor._inferShapeFromIterDefs(iterDefs);
        if (sizeSub === 0) {
            throw new Error('The number of indices cannot be zero. No elements to retrieve.')
        }
        let maxLevel = iterDefs.length - 1;
        if (iterDefs.length === 1) {
            stridesX = [this.size];
        } else {
            stridesX = this._strides;
        }
        if (iterInfo.areAllConstantType) {
            // ik are numbers for all k.
            // Result is a scalar if keepDims = false.
            let offset = 0;
            for (let i = 0;i < shapeSub.length;i++) {
                switch (iterDefs[i].type) {
                    case IndexIteratorType.Constant:
                        offset += stridesX[i] * <number>iterDefs[i].index;
                        break;
                    case IndexIteratorType.Range:
                        offset += stridesX[i] * <number>iterDefs[i].start;
                        break;
                    case IndexIteratorType.List:
                        offset += stridesX[i] * (<ArrayLike<number>>iterDefs[i].indices)[0];
                        break;
                    default:
                        throw new Error('Should never reach here.');
                }
            }
            let re = this._re.data[offset];
            let im = this.hasComplexStorage() ? this._im.data[offset] : 0;
            if (keepDims) {
                return Tensor.scalar(re, im, this.dtype, shapeSub.length);
            } else {
                return im === 0 ? re : new ComplexNumber(re, im);
            }
        } else {
            // The result is a Tensor
            let result = Tensor.zeros(shapeSub, this.dtype);
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
                switch (iterDefs[k].type) {
                    case IndexIteratorType.Constant:
                        trailingOffset += stridesX[k] * <number>iterDefs[k].index;
                        break;
                    case IndexIteratorType.List:
                        trailingOffset += stridesX[k] * (<ArrayLike<number>>iterDefs[k].indices)[0];
                        break;
                    case IndexIteratorType.Range:
                        trailingOffset += stridesX[k] * <number>iterDefs[k].start;
                        break;
                    default:
                        throw new Error('Should never reach here.');
                }
                finalStride *= stridesX[k - 1];
            }
            if (this.hasComplexStorage()) {
                result.ensureComplexStorage();
                let newIm = result._im.data;
                this._getSubTensorC(iterDefs, newRe, newIm, maxLevel, stridesX,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            } else {
                this._getSubTensorR(iterDefs, newRe, maxLevel, stridesX,
                    stridesSub, finalStride, trailingOffset, 0, 0, 0);
            }
            if (!keepDims) {
                // remove singleton dimensions along the axis where
                // IndexIterationType is Constant
                for (let i = 0, j = 0;i < shapeSub.length;i++, j++) {
                    if (iterDefs[j].type === IndexIteratorType.Constant) {
                        shapeSub.splice(i, 1);
                        i--;
                    }
                }
                result._shape = shapeSub;
                result._updateStridesAndCalculator();
            }
            return result;
        }
    }

    private _getSubTensorR(iterDefs: IndexIteratorDefinition[],
                           newRe: DataBlock, maxLevel: number,
                           stridesX: number[], stridesSub: number[],
                           finalStride: number, trailingOffset: number, level: number,
                           offsetX: number, offsetSub: number): void {
        let ind = iterDefs[level];
        if (level === maxLevel) {
            // last level
            let j = 0;
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    newRe[offsetSub] = this._re.data[offsetX + <number>ind.index * finalStride + trailingOffset];
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            newRe[offsetSub + j] = this._re.data[offsetX + i * finalStride + trailingOffset];
                            j++;
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            newRe[offsetSub + j] = this._re.data[offsetX + i * finalStride + trailingOffset];
                            j++;
                        }
                    }
                    break;                                
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        newRe[offsetSub + j] =
                            this._re.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset];
                        j++;
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        } else {
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._getSubTensorR(iterDefs, newRe, maxLevel, stridesX,
                        stridesSub, finalStride, trailingOffset, level + 1,
                        offsetX + stridesX[level] * <number>ind.index, offsetSub);
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._getSubTensorR(iterDefs, newRe, maxLevel, stridesX,
                                stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + stridesX[level] * i, offsetSub);
                            offsetSub += stridesSub[level];
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._getSubTensorR(iterDefs, newRe, maxLevel, stridesX,
                                stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + stridesX[level] * i, offsetSub);
                            offsetSub += stridesSub[level];
                        }
                    }
                    break;
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._getSubTensorR(iterDefs, newRe, maxLevel, stridesX,
                            stridesSub, finalStride, trailingOffset, level + 1,
                            offsetX + stridesX[level] * (<ArrayLike<number>>ind.indices)[i],
                            offsetSub);
                        offsetSub += stridesSub[level];
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        }
    }

    private _getSubTensorC(iterDefs: IndexIteratorDefinition[],
                           newRe: DataBlock, newIm: DataBlock, maxLevel: number,
                           stridesX: number[], stridesSub: number[],
                           finalStride: number, trailingOffset: number, level: number,
                           offsetX: number, offsetSub: number): void {
        let ind = iterDefs[level];
        if (level === maxLevel) {
            // last level
            let j = 0;
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    newRe[offsetSub] = this._re.data[offsetX + <number>ind.index * finalStride + trailingOffset];
                    newIm[offsetSub] = this._im.data[offsetX + <number>ind.index * finalStride + trailingOffset];
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            newRe[offsetSub + j] = this._re.data[offsetX + i * finalStride + trailingOffset];
                            newIm[offsetSub + j] = this._im.data[offsetX + i * finalStride + trailingOffset];
                            j++;
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            newRe[offsetSub + j] = this._re.data[offsetX + i * finalStride + trailingOffset];
                            newIm[offsetSub + j] = this._im.data[offsetX + i * finalStride + trailingOffset];
                            j++;
                        }
                    }
                    break;                                
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        newRe[offsetSub + j] =
                            this._re.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset];
                        newIm[offsetSub + j] =
                            this._im.data[offsetX + (<ArrayLike<number>>ind.indices)[i] * finalStride + trailingOffset];
                        j++;
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        } else {
            switch (ind.type) {
                case IndexIteratorType.Constant:
                    this._getSubTensorC(iterDefs, newRe, newIm, maxLevel,
                        stridesX, stridesSub, finalStride, trailingOffset, level + 1,
                        offsetX + stridesX[level] * <number>ind.index, offsetSub);
                    break;
                case IndexIteratorType.Range:
                    if (<number>ind.step > 0) {
                        for (let i = <number>ind.start;i < (<number>ind.stop);i += <number>ind.step) {
                            this._getSubTensorC(iterDefs, newRe, newIm, maxLevel,
                                stridesX, stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + stridesX[level] * i, offsetSub);
                            offsetSub += stridesSub[level];
                        }
                    } else {
                        for (let i = <number>ind.start;i > (<number>ind.stop);i += <number>ind.step) {
                            this._getSubTensorC(iterDefs, newRe, newIm, maxLevel,
                                stridesX, stridesSub, finalStride, trailingOffset, level + 1,
                                offsetX + stridesX[level] * i, offsetSub);
                            offsetSub += stridesSub[level];
                        }
                    }
                    break;
                case IndexIteratorType.List:
                    for (let i = 0;i < (<ArrayLike<number>>ind.indices).length;i++) {
                        this._getSubTensorC(iterDefs, newRe, newIm, maxLevel,
                            stridesX, stridesSub, finalStride, trailingOffset, level + 1,
                            offsetX + stridesX[level] * (<ArrayLike<number>>ind.indices)[i],
                            offsetSub);
                        offsetSub += stridesSub[level];
                    }
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
        }
    }

    
    private _parseIndexIterDefs(args: ArrayLike<any>): IndexIteratorInfo {
        let iterDefs: IndexIteratorDefinition[] = [];
        let allAreConstantType = true;
        for (let i = 0;i < args.length;i++) {
            // ik can be a integer, string, array/tensor of indices, logic
            // tensor as a mask. We unify them into list of index iterators.
            let ind = args[i];
            if (ind instanceof Tensor) {
                if (ind.hasNonZeroComplexStorage()) {
                    throw new Error('Indices cannot be complex');
                }
                if (ind.dtype === DType.LOGIC) {
                    if (ind.ndim !== 1 || ind.size !== this._shape[i]) {
                        throw new Error(`1D logic tensor of size ${this._shape[i]} expected for dimension ${i+1}.`);
                    }
                    iterDefs.push({
                        type: IndexIteratorType.List,
                        indices: DataHelper.findReal(ind.realData)
                    });
                } else {
                    iterDefs.push(this._parseSignedIndices(ind.realData, i));
                }
                allAreConstantType = false;
            } else if (Array.isArray(ind)) {
                // no nested arrays allowed here
                iterDefs.push(this._parseSignedIndices(<ArrayLike<number>>ind, i));
                allAreConstantType = false;
            } else if (typeof ind === 'string') {
                iterDefs.push(this._parseSlicingString(ind, i));
                allAreConstantType = false;
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
                iterDefs.push({
                    type: IndexIteratorType.Constant,
                    index: <number>ind
                });
            }
        }
        return { definitions: iterDefs, areAllConstantType: allAreConstantType };
    }

    /**
     * Gets the index iteration definition for a string that represents a
     * slicing operation (in the syntax of Python).
     * @param str A string that represents a slicing operation in the following
     *  format: 'start:stop:step', where 'step' is optional.
     * @param dim The dimension to which slicing operation is applied.
     */
    private _parseSlicingString(str: string, dim?: number): IndexIteratorDefinition {
        // we adapt the syntax of Python
        let splits = str.trim().split(':');
        let start: number, stop: number, step: number;
        let max = dim == undefined ? this.size : this._shape[dim];
        switch (splits.length) {
            case 1:
                start = parseFloat(splits[0]);
                if (start < 0) start += max;
                this._checkIndex(start, dim);
                return {
                    type: IndexIteratorType.Constant,
                    index: start
                }
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
                        stop = dim == undefined ? this.size : this._shape[dim];
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
                // stop cannot be equal to start
                if (stop === start) {
                    throw new Error('The stop index cannot be equal to the start index');
                }
                // consistency check
                if (start < stop && step < 0) {
                    throw new Error('The start index must be greater than the stop index when step is negative.');
                }
                if (start > stop && step > 0) {
                    throw new Error('The start index must be less than the stop index when step is positive.');
                }
                return {
                    type: IndexIteratorType.Range,
                    start: start,
                    stop: stop,
                    step: step
                };
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
    private _parseSignedIndices(indices: ArrayLike<number>, dim?: number): IndexIteratorDefinition {
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
        return {
            type: IndexIteratorType.List,
            indices: ret || indices
        };
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
            throw new Error(`Index ${index} is out of bounds for dimension ${dim}.`);
        }
    }

    private static _inferShapeFromIterDefs(iterDefs: IndexIteratorDefinition[]): [number[], number] {
        let shape: number[] = new Array(iterDefs.length);
        let size = 1;
        for (let i = 0;i < iterDefs.length;i++) {
            switch (iterDefs[i].type) {
                case IndexIteratorType.Constant:
                    shape[i] = 1;
                    break;
                case IndexIteratorType.List:
                    shape[i] = (<ArrayLike<number>>iterDefs[i].indices).length;
                    size *= shape[i];
                    break;
                case IndexIteratorType.Range:
                    let r = Math.abs(<number>iterDefs[i].stop - <number>iterDefs[i].start) - 1;
                    shape[i] = Math.max(0, Math.floor(r / Math.abs(<number>iterDefs[i].step))) + 1;
                    size *= shape[i];
                    break;
                default:
                    throw new Error('Should never reach here.');
            }
            if (size === 0) {
                throw new Error('Specified indices result in a empty tensor.');
            }
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
                        this._setBatch({
                            type: IndexIteratorType.List,
                            indices: DataHelper.findReal(arg0.realData)
                        }, args[1]);
                    } else {
                        // Because masked locations can be arbitrary, we cannot
                        // preserver the original shape and have to return a 
                        // 1D vector.
                        return this._getBatch({
                            type: IndexIteratorType.List,
                            indices: DataHelper.findReal(arg0.realData)
                        }, keepDims);
                    }
                } else {
                    // indexing with signed integers
                    if (arg0.hasNonZeroComplexStorage()) {
                        throw new Error('Complex tensor cannot be used for indexing.');
                    }
                    if (doSet) {
                        // For tensors, we access it flattened version by directly
                        // reading realData.
                        this._setBatch(this._parseSignedIndices(arg0.realData), args[1]);
                    } else {
                        if (arg0.ndim === 1 || arg0.isScalar()) {
                            return this._getBatch(this._parseSignedIndices(arg0.realData), keepDims);
                        } else {
                            // we want to preserve the original shape for the
                            // get case
                            originalShape = arg0.shape;
                            tmp = <Tensor>this._getBatch(this._parseSignedIndices(arg0.realData), keepDims);
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
                        originalShape = Tensor._getShapeFromArray(arg0);
                        Tensor._validateArrayShape(arg0, originalShape, 0);
                        this._setBatch(this._parseSignedNestedIndexArray(arg0, originalShape), args[1]);
                    } else {
                        this._setBatch(this._parseSignedIndices(arg0), args[1]);
                    }
                } else {
                    if (Array.isArray(arg0[0])) {
                        // nested array detected
                        originalShape = Tensor._getShapeFromArray(arg0);
                        Tensor._validateArrayShape(arg0, originalShape, 0);
                        tmp = this._getBatch(this._parseSignedNestedIndexArray(arg0, originalShape), keepDims);
                        // we want to preserve the original shape here
                        if (tmp instanceof Tensor) {
                            tmp._shape = originalShape;
                            tmp._updateStridesAndCalculator();
                        }
                        return tmp;
                    } else {
                        return this._getBatch(this._parseSignedIndices(arg0), keepDims);
                    }
                }
            } else if (arg0 instanceof Function) {
                // Case 3
                let indices = this.hasComplexStorage()
                    ? DataHelper.findWithCallbackComplex(this.realData, this.imagData, arg0)
                    : DataHelper.findWithCallbackReal(this.realData, arg0);
                // no need to check indices here
                if (doSet) {
                    this._setBatch({
                        type: IndexIteratorType.List,
                        indices: indices
                    }, args[1]);
                } else {
                    return this._getBatch({
                        type: IndexIteratorType.List,
                        indices: indices
                    }, keepDims);
                }
            } else if (typeof arg0 === 'string') {
                // Case 1 where indices are specified by a string
                if (doSet) {
                    this._setBatch(this._parseSlicingString(arg0), args[1]);
                } else {
                    return this._getBatch(this._parseSlicingString(arg0), keepDims);
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

    private _parseSignedNestedIndexArray(arr: any[], shape: number[]): IndexIteratorDefinition {
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
        return {
            type: IndexIteratorType.List,
            indices: indices
        };
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
        return new Tensor(this._re, TensorStorage.Empty, this._shape);
    }

    /**
     * Retrieves the imaginary part.
     */
    public imag(): Tensor {
        if (this.hasComplexStorage()) {
            return new Tensor(this._im, TensorStorage.Empty, this._shape);
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
     * Calculates the new shape for reshaping. If no modification is performed
     * on the input array, it will be returned. Otherwise a modified copy will
     * be returned.
     * @param newShape New shape.
     */
    private _calculateNewShape(newShape: number[]): number[] {
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
            // We do not want to modify the original array because it may be
            // used elsewhere.
            newShape = newShape.slice();
            newShape[idxM1] = this.size / ns;
            if ((newShape[idxM1] | 0) !== newShape[idxM1]) {
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
        if (copyStorageImmediately) {
            return new Tensor(this._re.dataCopy(), 
                this._im !== TensorStorage.Empty ? this._im.dataCopy() : TensorStorage.Empty,
                this._shape);
        } else {
            return new Tensor(this._re, this._im, this._shape);
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
            // just make a quick copy
            return this.copy(alwaysCopy);
        } else if (dtype === DType.LOGIC) {
            if (this.hasComplexStorage()) {
                throw new Error('Cannot convert a complex tensor to a logic tensor.');
            }
            return new Tensor(this._re.copyAsType(DType.LOGIC), TensorStorage.Empty, this._shape);
        } else {
            let re = this._re.copyAsType(dtype),
                im = this._im === TensorStorage.Empty ? TensorStorage.Empty : this._im.copyAsType(dtype);
            return new Tensor(re, im, this._shape);
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