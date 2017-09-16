import { OpInput, OpOutput, RealOpInput, RealOpOutputWithIndex, RealOpOutput } from '../../commonTypes';
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
    min(x: RealOpInput, axis?: number, keepDims?: boolean): RealOpOutputWithIndex;

    /**
     * Finds the maximum elements and their indices along the specified axis.
     * Note: NaN is treated as the largest number (larger than Infinity).
     */
    max(x: RealOpInput, axis?: number, keepDims?: boolean): RealOpOutputWithIndex;

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
    median(x: RealOpInput, axis?: number, keepDims?: boolean): RealOpOutput;

    /**
     * Computes the mode of the elements along the specified axis.
     * Note: NaNs will be ignored when determining the mode. If there are
     *       multiple modes, the smallest will be returned.
     */
    mode(x: RealOpInput, axis?: number, keepDims?: boolean): RealOpOutput;

    /**
     * Computes the sample variance (divided by N - 1) of the elements along the
     * specified axis.
     */
    var(x: OpInput, axis?: number, keepDims?: boolean): RealOpOutput;

    /**
     * Computes the standard deviation (divided by N - 1) of the elements along
     * the specified axis.
     */
    std(x: OpInput, axis?: number, keepDims?: boolean): RealOpOutput;

    /**
     * Estimates the (cross-)covariance matrix from samples. Assuming samples
     * are stored as column vectors, the cross-covariance matrix is computed
     * via:
     *  cov(X, Y) = E[(X - E[X])(Y - E[Y])^H]
     * @param x An 1D or 2D input of samples.
     * @param y (Optional) Another set of samples stored in the same format as
     *          x. If specified and different from x, the cross covariance
     *          matrix between x and y will be computed. Default value is x and
     *          the covariance matrix of x is computed.
     *          Note: x and y must share the same amount of samples.
     * @param samplesInColumns (Optional) If set to true, each column in x
     *          (or y) represents a sample. Otherwise each row in x (or y)
     *          represents a sample. Default value is true.
     * @return If each sample in x has dimension px, and each sample in y has
     *         dimension py, a px x py matrix will be returned.
     */
    cov(x: OpInput, y?: OpInput, samplesInColumns?: boolean): Tensor;

    /**
     * Computes the sample correlation coefficient matrix from samples. More
     * specifically
     *  corrcoef(x_i, y_j) = cov(x_i, y_j) / (cov(x_i, x_i) * cov(y_j, y_j))
     * @param x An 1D or 2D input of samples.
     * @param y (Optional) Another set of samples stored in the same format as
     *          x. If specified and different from x, the correlation
     *          coefficient matrix between x and y will be computed. Default
     *          value is x and the autocorrelation coefficient matrix is
     *          computed.
     *          Note: x and y must have the same shape.
     * @param samplesInColumns (Optional) If set to true, each column in x
     *          (or y) represents a sample. Otherwise each row in x (or y)
     *          represents a sample. Default value is true.
     * @return If each sample in x has dimension px, and each sample in y has
     *         dimension py, a px x py matrix will be returned.
     */
    corrcoef(x: OpInput, y?: OpInput, samplesInColumns?: boolean): Tensor;

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
     * Used for creating histograms. Bins the elements in the input into
     * specified number of bins. Returns the number of elements in each bin
     * and the edges of the bins.
     * All bins except the last one are left-closed and right-open.
     * @param x
     * @param nbin (Optional) Number of bins. Default value is 10.
     */
    hist(x: RealOpInput, nBins?: number): [Tensor, Tensor];

    /**
     * Fast Fourier transform.
     */
    fft(x: OpInput, axis?: number): Tensor;

    /**
     * Inverse fast Fouriert transform.
     */
    ifft(x: OpInput, axis?: number): Tensor;

}
