import { OpInput, OpOutput } from '../../commonTypes';
import { Tensor } from '../../tensor';

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

    /**
     * Least-squares polynomial fit.
     *  min \sum_i [y_i - (p[0]*x_i^N + p[1]*x_i^{N-1} + ... + p[N])]^2
     * Note: it is possible that the least-squares problem is ill-conditioned,
     *       leading to inaccurate estimates.
     * @param x A vector consists of the x-coordinates of the samples points.
     * @param y A vector consists of the y-coordinates of the samples points.
     * @param n Degree of the fitting polynomial.
     * @returns The coefficients of the polynomial.
     */
    polyfit(x: OpInput, y: OpInput, n: number): Tensor;

}