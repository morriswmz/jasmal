import { ComplexNumber } from './complexNumber';
import { Tensor } from './tensor';
import { DType } from './dtype';

export type TypedArray = Uint8Array | Uint16Array | Uint32Array | Int8Array | Int16Array | Int32Array | Float32Array | Float64Array;

/**
 * A number or a complex number.
 */
export type Scalar = number | ComplexNumber;

/**
 * An input can be a number, a complex number, an array (possibly nested),
 * or a tensor.
 */
export type OpInput = number | any[] | ComplexNumber | Tensor;

/**
 * An output can be a number, a complex number, or a tensor.
 */
export type OpOutput = number | ComplexNumber | Tensor;

/**
 * An output with both value and index.
 */
export type OpOutputWithIndex = [number | ComplexNumber, number] | [Tensor, number[]];

/**
 * Describes types of inputs.
 */
export const enum OpInputType {
    Number,
    ComplexNumber,
    Array,
    Tensor,
    Unknown
};

/**
 * Stores data unified from the following types of inputs:
 *  number, ComplexNumber, number[], n-d array, Tensor
 */
export interface OpInputInfo {
    /**
     * True if the original type is Number or ComplexNumber.
     */
    isInputScalar: boolean;
    /**
     * True if there is only one element.
     */
    hasOnlyOneElement: boolean;
    /**
     * True if there is any complex element.
     */
    isComplex: boolean;
    /**
     * Real part if isOriginalTypeScalar or hasOnlyOneElement is true.
     * Otherwise zero.
     */
    re: number;
    /**
     * Imaginary part if isOriginalTypeScalar or hasOnlyOneElement is true.
     * Otherwise zero.
     */
    im: number;
    /**
     * An array like object storing the flattened real part. Will be empty if
     * isOriginalTypeScalar is false.
     */
    reArr: ArrayLike<number>;
    /**
     * An array like object storing the flattened imaginary part. Will be empty
     * if isOriginalTypeScalar or isComplex is false.
     */
    imArr: ArrayLike<number>;
    /**
     * The shape of the original input. For consistency, if the input type is
     * number or ComplexNumber, the shape will be [1].
     */
    originalShape: number[];
    /**
     * The type of the original input.
     */
    originalType: OpInputType;
    /**
     * The data type (of each element) of the original input.
     */
    originalDType: DType;
}