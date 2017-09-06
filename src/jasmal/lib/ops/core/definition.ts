import { OpInput, OpOutput } from '../../commonTypes';
import { Tensor } from '../../tensor';

/**
 * Provides essential tensor operations.
 */
export interface ICoreOpProvider {

    /**
     * Reshapes the input tensor without changing its elements.
     * @param x
     * @param shape New shape. The number of elements should not change. It can
     *              contain up to one -1. If shape[j] = -1, then shape[j] will
     *              be computed automatically.
     */
    reshape(x: OpInput, shape: number[]): Tensor;

    /**
     * Flattens the input into a 1D vector (with ndim = 1).
     * Equivalent to `reshape(x, [-1])`.
     */
    flatten(x: OpInput): Tensor;

    /**
     * Returns a new tensor with all singleton dimensions removed.
     */
    squeeze(x: OpInput): Tensor;

    /**
     * Flattens the input into a Nx1 column vector (with ndim = 2).
     * Equivalent to `reshape(x, [-1,1])`.
     */
    vec(x: OpInput): Tensor;

    /**
     * Concatenates tensors along the specified axis.
     * @param inputs An array of tensor compatible objects.
     * @param axis (Optional) The axis along which the tensors will be
     *             concatenated. Default value is 0.
     * @example
     *  // shape is [1, 2]
     *  let x = T.fromArray([[1, 2]]);
     *  // shape is [2, 2]
     *  let y = [[3, 4], [5, 6]]
     *  // shape is now [3, 2]
     *  let z = T.concat([x, y], 0);
     */
    concat(inputs: OpInput[], axis?: number): Tensor;

    /**
     * Forms a new tensor by repeating the input along the specified axis.
     * Denote the length of `repeats` by r and the number of dimensions of `x`
     * by n. The number of dimensions of the output tensor will be max(n, r).
     * If r > n, new axes will be prepended `x` to promote it into a
     * r-dimensional tensor.
     * If r < n, ones will be prepended to `repeats` to promote it into a array
     * of length n.
     * @param x
     * @param repeats Number of repeats along each axis.
     * @example
     *  let x = T.fromArray([[1, 2], [3, 4]]);
     *  // y will be
     *  // [[1, 2, 1, 2],
     *  //  [3, 4, 3, 4],
     *  //  [1, 2, 1, 2],
     *  //  [3, 4, 3, 4]]
     *  let y = T.tile(x, [2, 2]);
     */
    tile(x: OpInput, repeats: number[]): Tensor;

    /**
     * Permutes a tensor according to the specified order such that
     *  shapeOut[i] = shapeIn[order[i]]
     *  Y(i_{order[0]}, ..., i_{order[n-1]}) = X(i_0, ..., i_{n-1})
     * @example
     *  // shape is [1, 2, 3]
     *  let x = T.ones([1, 2, 3]); 
     *  // shape is [3, 1, 2]
     *  let y = T.permuteAxis(x, [2, 0, 1]);
     */
    permuteAxis(x: OpInput, order: number[]): Tensor;

    /**
     * Prepend a new axis to the input.
     * @param x
     */
    prependAxis(x: OpInput): Tensor;

    /**
     * Append a new axis to the input.
     * @param x
     */
    appendAxis(x: OpInput): Tensor;

    /**
     * Retrieves the real part as a tensor.
     * @param x
     */
    real(x: OpInput): Tensor;

    /**
     * Retrieves the imaginary part as a tensor.
     * If the input tensor has no complex storage, a tensor filled with zeros
     * will be returned.
     * @param x
     */
    imag(x: OpInput): Tensor;

    /**
     * Checks if the input has any element.
     * @param x
     * @returns Returns true if x is an empty tensor, an array of length 0, or
     *          a nested array consists of zero numeric elements (e.g. [[], []]).
     */
    isempty(x: OpInput): boolean;
    
    /**
     * Returns true if every element is a real number (imaginary part is zero).
     * @param x
     */
    isreal(x: OpInput): boolean;

    /**
     * Applies `isNaN(x)` element-wise.
     * @param x
     */
    isnan(x: OpInput): OpOutput;

    /**
     * Applies `isFinite(x) && !isNaN(x)` element-wise.
     * @param x
     */
    isinf(x: OpInput): OpOutput;

    /**
     * Generates a 1D vector containing `n` evenly spaced points starting from
     * `x1` and ends at `x2`. If `n` is one, then `x1` will be the only point.
     * @param x1 Start.
     * @param x2 Stop.
     * @param n Number of points.
     */
    linspace(x1: number, x2: number, n: number): Tensor;

    /**
     * Generates a 1D vector containing `n` Logarithmically spaced points.
     * Equivalent to `T.pow(base, T.linspace(x1, x2, n))`.
     * @param x1 Start.
     * @param x2 Stop.
     * @param n Number of points.
     * @param base (Optional) Default value is 10.
     */
    logspace(x1: number, x2: number, n: number, base?: number): Tensor;

    /**
     * Returns the indices of the elements that satisfy the specified condition.
     * @param x
     * @param f (Optional) A predicate. Default value is
     *          `(re, im) => re !== 0 || im !== 0`.
     */
    find(x: OpInput, f?: (re: number, im: number) => boolean): number[];

}
