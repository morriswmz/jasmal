import { Tensor } from '../tensor';
import { ComplexNumber } from '../complexNumber';
import { OpInput, OpOutput, OpOutputWithIndex, Scalar } from '../commonTypes';
import { DType } from '../dtype';

/**
 * General rules:
 * 1. If all the inputs are scalars, the output is scalar.
 * 2. If any of the input is an array/tensor, the output will be a tensor.
 */

/**
 * Represents a binary operation.
 */
export type GenericBinaryOp = (x: OpInput, y: OpInput, inPlace?: boolean) => OpOutput;
/**
 * Represents a unary operation.
 */
export type GenericUnaryOp = (x: OpInput, inPlace?: boolean) => OpOutput;
/**
 * Represents a unary operation with a parameter.
 */
export type OneParamUnaryOp = (x: OpInput, p: number, inPlace?: boolean) => OpOutput;

/**
 * Provides essential tensor operations.
 */
export interface ICoreOpProvider {

    reshape(x: OpInput, shape: number[]): Tensor;

    /**
     * Flattens the input into a 1D vector (with ndim = 1).
     */
    flatten(x: OpInput): Tensor;

    /**
     * Returns a new tensor with all singleton dimensions removed.
     */
    squeeze(x: OpInput): Tensor;

    /**
     * Flattens the input into a Nx1 vector (with ndim = 2).
     */
    vec(x: OpInput): Tensor;

    concat(inputs: OpInput[], axis?: number): Tensor;

    tile(x: OpInput, repeats: number[]): Tensor;

    prependAxis(x: OpInput): Tensor;

    appendAxis(x: OpInput): Tensor;

    real(x: OpInput): Tensor;

    imag(x: OpInput): Tensor;

    /**
     * Checks if every element of x is real.
     */
    isreal(x: OpInput): boolean;

    isnan(x: OpInput): OpOutput;

    isinf(x: OpInput): OpOutput;

    linspace(x1: number, x2: number, n: number): Tensor;

    logspace(x1: number, x2: number, n: number, base?: number): Tensor;

    find(x: OpInput, f?: (re: number, im: number) => boolean): number[];

}

export const enum MatrixModifier {
    None = 0,
    Transposed = 1,
    Hermitian = 2
}

export interface IMatrixOpProvider {

    eye(m: number, n?: number, dtype?: DType): Tensor;

    hilb(n: number): Tensor;

    diag(x: OpInput): Tensor;

    matmul(x: OpInput, y: OpInput, yModifier?: MatrixModifier): OpOutput;

    kron(x: OpInput, y: OpInput): Tensor;

    transpose(x: OpInput): Tensor;

    hermitian(x: OpInput): Tensor;

    trace(x: OpInput): Scalar;

    inv(x: OpInput): Tensor;

    det(x: OpInput): Scalar;

    norm(x: OpInput, p: number | 'fro'): number;

    lu(x: OpInput, compact: true): [Tensor, number[]];
    lu(x: OpInput, compact: false): [Tensor, Tensor, Tensor];

    svd(x: OpInput): [Tensor, Tensor, Tensor];

    rank(x: OpInput): number;

}

export interface IArithmeticOpProvider {

    add(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;
    
    sub(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    neg(x: OpInput, inPlace?: boolean): OpOutput;

    mul(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    div(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    reciprocal(x: OpInput, inPlace?: boolean): OpOutput;

    rem(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

}

export interface ILogicComparisonOpProvider {

    eq(x: OpInput, y: OpInput): OpOutput;

    neq(x: OpInput, y: OpInput): OpOutput;

    gt(x: OpInput, y: OpInput): OpOutput;

    ge(x: OpInput, y: OpInput): OpOutput;

    lt(x: OpInput, y: OpInput): OpOutput;

    le(x: OpInput, y: OpInput): OpOutput;

    and(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    or(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    xor(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    not(x: OpInput): OpOutput;

    all(x: OpInput): boolean;

    any(x: OpInput): boolean;            

}

export interface IMathOpProvider {

    abs(x: OpInput, inPlace?: boolean): OpOutput;

    sign(x: OpInput, inPlace?: boolean): OpOutput;

    min2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;
    
    max2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    conj(x: OpInput, inPlace?: boolean): OpOutput;

    angle(x: OpInput, inPlace?: boolean): OpOutput;

    sin(x: OpInput, inPlace?: boolean): OpOutput;

    cos(x: OpInput, inPlace?: boolean): OpOutput;

    tan(x: OpInput, inPlace?: boolean): OpOutput;

    cot(x: OpInput, inPlace?: boolean): OpOutput;

    sinh(x: OpInput, inPlace?: boolean): OpOutput;

    cosh(x: OpInput, inPlace?: boolean): OpOutput;

    tanh(x: OpInput, inPlace?: boolean): OpOutput;

    asin(x: OpInput, inPlace?: boolean): OpOutput;

    acos(x: OpInput, inPlace?: boolean): OpOutput;

    atan(x: OpInput, inPlace?: boolean): OpOutput;        

    sqrt(x: OpInput, inPlace?: boolean): OpOutput;

    pow2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    exp(x: OpInput, inPlace?: boolean): OpOutput;

    log(x: OpInput, inPlace?: boolean): OpOutput;

    floor(x: OpInput, inPlace?: boolean): OpOutput;

    ceil(x: OpInput, inPlace?: boolean): OpOutput;

    round(x: OpInput, inPlace?: boolean): OpOutput;

    rad2deg(x: OpInput, inPlace?: boolean): OpOutput;

    deg2rad(x: OpInput, inPlace?: boolean): OpOutput;

}

export interface IRandomOpProvider {

    seed(s: number): void;

    rand(): number;
    rand(shape: number[]): Tensor;

    randn(): number;
    randn(shape: number[]): Tensor;

    randi(high: number): number;
    randi(low: number, high: number): number;
    randi(low: number, high: number, shape: number[]): Tensor;

}

export interface IDataOpProvider {

    min(x: OpInput, y?: OpInput, keepDims?: boolean): OpOutputWithIndex;

    max(x: OpInput, y?: OpInput, keepDims?: boolean): OpOutputWithIndex;

    sum(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    prod(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    cumsum(x: OpInput, axis?: number): Tensor;

    mean(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    median(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    var(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    std(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

}