import { OpInput, OpOutput } from '../../commonTypes';
import { Tensor } from '../../tensor';

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

    /**
     * Permutes a tensor according to the specified order such that
     *  shapeOut[i] = shapeIn[order[i]]
     *  Y(i_{order[0]}, ..., i_{order[n-1]}) = X(i_0, ..., i_{n-1})
     * @example
     *  // shape is [1, 2, 3]
     *  x = T.ones([1, 2, 3]); 
     *  // shape is [3, 1, 2]
     *  y = T.permuteAxis(x, [2, 0, 1]);
     */
    permuteAxis(x: OpInput, order: number[]): Tensor;

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