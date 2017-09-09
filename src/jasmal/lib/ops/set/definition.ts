import { Tensor } from '../../tensor';
import { OpInput, Scalar, NonScalarOpInput } from '../../commonTypes';

export interface ISetOpProvider {

    /**
     * Obtains the unique elements in the flattened input.
     * Note: NaN is not equal to NaN. Therefore no NaN will be removed.
     * @returns An 1D tensor object containing the unique elements sorted in
     * ascending order.
     */
    unique(x: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains the unique elements in the flattened input and returns relevant
     * indices.
     * Note: NaN is not equal to NaN. Therefore no NaN will be removed.
     * @returns A tuple [y, iy, ix], where y contains the unique elements
     * sorted in ascending order, iy is an array of indices such that y can be
     * formed by picking elements in x according to the indices in iy, and ix[j]
     * contains the indices of elements in x that are equal to the j-th element
     * in y.
     */
    unique(x: OpInput, outputIndices: true): [Tensor, number[], number[][]];

    /**
     * Checks if x is an element in y.
     */
    isin(x: Scalar, y: OpInput, outputIndices?: false): boolean;
    /**
     * Checks if x is an element in y.
     * @returns A tuple [m, i], where m is a boolean indicating whether x is in
     * y, and i is the index of x in y (-1 if x is not in y).
     */
    isin(x: Scalar, y: OpInput, outputIndices: true): [boolean, number];
    /**
     * For every element in x, checks if it is an element in y.
     */
    isin(x: NonScalarOpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * For every element in x, checks if it is an element in y.
     * @returns A tuple [M, I]. M is a LOGIC tensor having the same shape with
     * x, masking the elements in x that appear in y. I is a tensor storing the
     * indices such that
     * 1. If the k-th element of x, x[k], appears in y, then the k-th element of
     *    I is the index of x[k] in y.
     * 2. If the k-th element of x, x[k], is not in y, then the k-th element of
     *    I is -1.
     */
    isin(x: NonScalarOpInput, y: OpInput, outputIndices: true): [Tensor, Tensor];

    /**
     * Obtains the union of flattened x and y.
     * @returns An 1D tensor object representing the union of x and y, whose
     * elements are sorted in ascending order.
     */
    union(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains the union of flattened x and y. Also returns relevant indices.
     * @returns A tuple [z, ix, iy], where z is an 1D tensor object being the
     * union of flattened x and y, whose elements are sorted in ascending order.
     * ix and iy contains the indices of elements picked from x and y to form z.
     * If an element appears in both x and y, then the index of this element in
     * x is returned in ix. If there are repeated elements in x or y, the index
     * of the first occurrence of these repeated elements is used when forming
     * ix and iy.
     */
    union(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];
    
    /**
     * Obtains the intersection of flattened x and y. 
     * @returns An 1D tensor object representing the intersection of x and y,
     * whose elements are sorted in ascending order.
     */
    intersect(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains the intersection of flattened x and y.
     * @returns A tuple [z, ix, iy], where z is an 1D tensor object representing
     * the intersection of x and y, whose elements are sorted in ascending
     * order. ix and iy are both arrays of indices such that z can be formed
     * by either picking elements in x according to ix, or picking elements in
     * y according to iy.
     */
    intersect(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];

    /**
     * Obtains a new set by removing elements that appear in y from x. Both
     * x and y are flattened before perform the operation.
     * @returns An 1D tensor object storing the result of the set difference
     * operation.
     */
    setdiff(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains a new set by removing elements that appear in y from x. Both
     * x and y are flattened before perform the operation.
     * @returns A tuple [z, ix], where z is an 1D tensor object storing the
     * result of the set difference operation, and ix is an array of indices
     * such that z can be formed by picking elements in x according to ix.
     */
    setdiff(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[]];
}
