import { ShapeHelper } from './helper/shapeHelper';
import { DType } from './dtype';
import { TypedArray } from './commonTypes';

/**
 * Represents a chunk of memory (supported by typed arrays).
 * This is a writable version of ArrayLike<number>.
 */
export interface DataBlock {
    [index: number]: number;
    length: number;
}

export class TensorStorage {

    private _data: DataBlock;
    private _dtype: DType;

    /**
     * A simple reference counter.
     * Since JavaScript object never know when they will be garbage collected,
     * this counter is not reliable. The number may be greater than 0 even if
     * no tensor is using this storage.
     * Consider the following case:
     *  x = Tensor.zeros([10, 10]); // refCount = 1
     *  x = x.copy(); // refCount = 2 when the new copy is made. However,
     *                // the original tensor may be garbage collected. There is
     *                // no way to tell that so refCount will not be decreased
     *                // by one.
     *  x.set(...); // This triggers an unnecessary copy because refCount > 1
     */
    public refCount: number;

    private constructor(data: DataBlock, dtype: DType) {
        this._data = data;
        this._dtype = dtype;
        this.refCount = 0;
    }

    /**
     * Checks typed array support.
     */
    public static HasTypedArraySupport = Float64Array && (typeof Float64Array === 'function');

    /**
     * A constant that represents an empty TensorStorage.
     */
    public static Empty: TensorStorage = TensorStorage.create(0, DType.LOGIC);

    /**
     * Validates if the given data type is supported. Throws if not supported.
     * @param dtype Data type.
     */
    public static ValidateDTypeSupport(dtype: DType): void {
        if (!TensorStorage.HasTypedArraySupport) {
            if (!((dtype === DType.FLOAT64) || (dtype === DType.LOGIC))) {
                throw new Error('When native typed arrays are not available, only DType.LOGIC and DType.FLOAT64 are supported.')
            }
        }
    }

    /**
     * Creates a TensorStorage filled with zeros.
     * @param size Maximum number of elements the TensorStorage can store.
     * @param dtype Element data type.
     */
    public static create(size: number, dtype: DType = DType.FLOAT64): TensorStorage {
        let data: DataBlock;
        if (TensorStorage.HasTypedArraySupport) {
            switch (dtype) {
                case DType.LOGIC:
                    data = new Uint8Array(size);
                    break;
                case DType.INT32:
                    data = new Int32Array(size);
                    break;
                case DType.FLOAT64:
                    data = new Float64Array(size);
                    break;
                default:
                    throw new Error(`Unknown dtype "${dtype}".`);
            }
        } else {
            TensorStorage.ValidateDTypeSupport(dtype);
            data = new Array<number>(size);
            // remember to fill it with zeros
            for (let i = 0;i < size;i++) {
                data[i] = 0;
            }
        }
        return new TensorStorage(data, dtype);
    }

    /**
     * Creates a TensorStorage from a multi-dimensional JavaScript array.
     * @param arr A multi-dimensional JavaScript array storing the tensor data.
     * @param shape Shape of the JavaScript array. This must match the actual
     *  shape of the input array.
     * @param dtype Data type.
     */
    public static fromArray(arr: any[] | TypedArray, shape: number[], dtype: DType = DType.FLOAT64): TensorStorage {
        TensorStorage.ValidateDTypeSupport(dtype);
        let strides = ShapeHelper.computeStrides(shape);
        let size = strides[0] * shape[0];
        let storage = TensorStorage.create(size, dtype);
        let copyFromArray = (arr: any[] | TypedArray, level: number, offset: number) => {
            if (level === shape.length - 1) {
                // final level
                if (storage.dtype === DType.LOGIC) {
                    for (let i = 0;i < shape[level];i++) {
                        storage.setAsLogicAtUnchecked(offset + i, arr[i]);
                    }
                } else {
                    // We rely on the type casting mechanism of typed arrays here.
                    for (let i = 0;i < shape[level];i++) {
                        storage.data[offset + i] = arr[i];
                    }
                }
            } else {
                for (let i = 0;i < shape[level];i++) {
                    copyFromArray(arr[i], level + 1, offset);
                    offset += strides[level];
                }
            }
        };
        copyFromArray(arr, 0, 0);
        return storage;
    }

    /**
     * Returns a copy.
     * Note: refCount will not be copied.
     */
    public dataCopy(): TensorStorage {
        let data: DataBlock;
        if (TensorStorage.HasTypedArraySupport) {
            switch (this.dtype) {
                case DType.LOGIC:
                    data = new Uint8Array(this.data);
                    break;
                case DType.INT32:
                    data = new Int32Array(this.data);
                    break;
                case DType.FLOAT64:
                    data = new Float64Array(this.data);
                    break;
                default:
                    throw new Error("This should never happen!");
            }
        } else {
            // without typed array support, the only possibility of the type of
            // data is Array<number>.
            data = (<Array<number>>this.data).slice();
        }
        return new TensorStorage(data, this.dtype);
    }

    /**
     * Retrieves the raw data.
     */
    public get data(): DataBlock {
        return this._data;
    }

    /**
     * Retrieves the data type.
     */
    public get dtype(): DType {
        return this._dtype;
    }

    /**
     * Sets the value at the specified offset as the casted logic value of the
     * given value without boundary check.
     * @param offset Offset.
     * @param v Value.
     */
    public setAsLogicAtUnchecked(offset: number, v: number): void {
        if (isNaN(v)) {
            throw new Error('Cannot convert NaN to logic values.');
        }
        this._data[offset] = !!v ? 1 : 0;
    }

    /**
     * Gets the value at the specified offset as a logic value without boundary
     * check.
     * @param offset Offset.
     */
    public getAsLogicAtUnchecked(offset: number): number {
        if (isNaN(this._data[offset])) {
            throw new Error('Cannot convert NaN to logic values.');
        }
        return this._data[offset] !== 0 ? 1 : 0;
    }

    /**
     * Return a copy whose elements are converted to the specified data type.
     * @param dtype Data type.
     */
    public copyAsType(dtype: DType): TensorStorage {
        let data: DataBlock;
        if (TensorStorage.HasTypedArraySupport) {
            switch (dtype) {
                case DType.LOGIC:
                    data = new Uint8Array(this.data.length);
                    for (let i = 0;i < this.data.length;i++) {
                        data[i] = this.getAsLogicAtUnchecked(i);
                    }
                    break;
                case DType.INT32:
                    data = new Int32Array(this.data);
                    break;
                case DType.FLOAT64:
                    data = new Float64Array(this.data);
                    break;
                default:
                    throw new Error("This should never happen!");
            }
        } else {
            TensorStorage.ValidateDTypeSupport(dtype);
            switch (dtype) {
                case DType.LOGIC:
                    data = new Array(this.data.length);
                    for (let i = 0;i < this.data.length;i++) {
                        data[i] = this.getAsLogicAtUnchecked(i);
                    }
                    break;
                case DType.FLOAT64:
                    data = (<Array<number>>this.data).slice();
                    break;
                default:
                    throw new Error("This should never happen!");
            }
        }
        return new TensorStorage(data, dtype);
    }

}