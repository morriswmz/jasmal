import { Tensor } from '../../tensor';
import { OpInput } from '../../commonTypes';

export interface ISetOpProvider {

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

}
