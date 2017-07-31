import { OpInput, OpOutputWithIndex, OpOutput } from '../../commonTypes';
import { Tensor } from '../../tensor';

// TODO: add sortRows, unique, hist
export interface IDataOpProvider {

    min(x: OpInput, axis?: number, keepDims?: boolean): OpOutputWithIndex;

    max(x: OpInput, axis?: number, keepDims?: boolean): OpOutputWithIndex;

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