import { OpInput, OpOutput } from '../../commonTypes';

export interface IPolynomialOpProvider {

    /**
     * Evaluates the polynomial:
     *  p[0]*x^N + p[1]*x^{N-1} + ... + p[N]
     * at x.
     * @param p A vector of coefficients.
     * @param x
     * @returns A scalar if x is a scalar. Otherwise a tensor object is
     *          returned.
     */
    polyval(p: OpInput, x: OpInput): OpOutput;

    /**
     * Evaluates the matrix polynomial:
     *  p[0]*x^N + p[1]*x^{N-1} + ... + p[N]*I
     * at x.
     * @param p A vector of coefficients.
     * @param x A scalar or a square matrix.
     * @returns A scalar if x is a scalar. Otherwise a tensor object is
     *          returned.
     */
    polyvalm(p: OpInput, x: OpInput): OpOutput;

}