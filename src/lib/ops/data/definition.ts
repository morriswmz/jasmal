import { OpInput, OpOutputWithIndex, OpOutput } from '../../commonTypes';
import { Tensor } from '../../tensor';

// TODO: add sortRows
export interface IDataOpProvider {

    /**
     * Finds the minimum elements and their indices along the specified axis.
     * @param x
     * @param axis (Optional) Specifies the axis along which the operation is
     *             performed. This value must be either -1 or a nonnegative
     *             integer less than the number of dimensions in `x`.
     *             Specify -1 if you want to find the minimum among all the
     *             elements. Default value is -1.
     * @param keepDims (Optional) Specifies whether the dimension specified by
     *                 `axis` is kept in the results. Default value is false.
     * @returns A 2 element tuple [v, i] where v consists of the minimums and
     *          i consists of the indices of the minimums. v and i are scalars
     *          only when `axis` is set to -1 or `x` is a 1D vector, and
     *          `keepDims` is set to false. 
     */
    min(x: OpInput, axis?: number, keepDims?: boolean): OpOutputWithIndex;

    max(x: OpInput, axis?: number, keepDims?: boolean): OpOutputWithIndex;

    /**
     * Sums the elements along the specified axis.
     */
    sum(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    prod(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    cumsum(x: OpInput, axis?: number): Tensor;

    mean(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    median(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    var(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    std(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    sort(x: OpInput, dir: 'asc' | 'desc', outputIndices: false): Tensor;
    sort(x: OpInput, dir: 'asc' | 'desc', outputIndices: true): [Tensor, number[]];

    /**
     * Obtains the unique elements in the (flattened) input.
     */
    unique(x: OpInput, outputIndices: false): Tensor;
    /**
     * Obtains the unique elements in the (flattened) input.
     * Returns a tensor tuple [y, iy, ix] such that y = x(iy) and ix[j]
     * contains the indices in x such that all elements in x[ix[j]] equal to
     * y[j].
     */
    unique(x: OpInput, outputIndices: true): [Tensor, number[], number[][]];

    /**
     * Used for creating histograms. Bins the elements in the input into
     * specified number of bins. Returns the number of elements in each bin
     * and the edges of the bins.
     * All bins except the last one are left-closed and right-open.
     * @param x
     * @param nbin (Optional) Number of bins. Default value is 10.
     */
    hist(x: OpInput, nBins?: number): [Tensor, Tensor];

}