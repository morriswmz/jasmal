import { OpInput, OpOutputWithIndex, OpOutput } from '../../commonTypes';
import { Tensor } from '../../tensor';

export interface IDataOpProvider {

    /**
     * Finds the minimum elements and their indices along the specified axis.
     * Note: NaN is treated as the largest number (larger than Infinity).
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

    /**
     * Finds the maximum elements and their indices along the specified axis.
     * Note: NaN is treated as the largest number (larger than Infinity).
     */
    max(x: OpInput, axis?: number, keepDims?: boolean): OpOutputWithIndex;

    /**
     * Sums the elements along the specified axis.
     */
    sum(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    /**
     * Computes the products the elements along the specified axis.
     */
    prod(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    /**
     * Computes the cumulative sum of the elements along the specified axis. 
     */
    cumsum(x: OpInput, axis?: number): Tensor;

    /**
     * Computes the mean of the elements along the specified axis.
     */
    mean(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    /**
     * Computes the median of the elements along the specified axis.
     */
    median(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    /**
     * Computes the mode of the elements along the specified axis.
     * Note: NaNs will be ignored when determining the mode. If there are
     *       multiple modes, the smallest will be returned.
     */
    mode(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    /**
     * Computes the sample variance (divided by N - 1) of the elements along the
     * specified axis.
     */
    var(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    /**
     * Computes the standard deviation (divided by N - 1) of the elements along
     * the specified axis.
     */
    std(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    /**
     * Sorts the elements in the (flattened) input in the specified order and
     * return the result as a new tensor.
     * Note: NaN is treated as the largest number (larger than Infinity).
     */
    sort(x: OpInput, dir: 'asc' | 'desc', outputIndices: false): Tensor;
    /**
     * Sorts the elements in the (flattened) input in the specified order and
     * return the result as a new tensor y as well as the index map i such
     * that x.get(i) = y.
     * Note: NaN is treated as the largest number (larger than Infinity).
     */
    sort(x: OpInput, dir: 'asc' | 'desc', outputIndices: true): [Tensor, number[]];

    /**
     * Sorts the rows in the specified order and return the result as a new
     * tensor.
     * Note: NaN is treated as the largest number (larger than Infinity).
     */
    sortRows(x: OpInput, dir: 'asc' | 'desc', outputIndices: false): Tensor;
    /**
     * Sorts the rows in the specified order and return the result as a new
     * tensor y as well as the index map i such that x.get(i,':') = y.
     * Note: NaN is treated as the largest number (larger than Infinity).
     */
    sortRows(x: OpInput, dir: 'asc' | 'desc', outputIndices: true): [Tensor, number[]];

    /**
     * Obtains the unique elements in the (flattened) input.
     * Note: NaN is not equal to NaN. Therefore no NaN will be removed.
     */
    unique(x: OpInput, outputIndices: false): Tensor;
    /**
     * Obtains the unique elements in the (flattened) input.
     * Returns a tensor tuple [y, iy, ix] such that y = x[iy] and ix[j]
     * contains the indices in x such that all elements in x[ix[j]] equal to
     * y[j].
     * Note: NaN is not equal to NaN. Therefore no NaN will be removed.
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