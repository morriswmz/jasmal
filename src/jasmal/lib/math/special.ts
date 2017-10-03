import { LOGE2 } from '../constant';

export class FactorialTable {

    /**
     * The max integer whose factorial is not Infinity in double format.
     */
    public static readonly MAX_INTEGER = 170;

    private static _table: number[] = new Array<number>(FactorialTable.MAX_INTEGER + 1);
    private static _inited: boolean = false;

    private static _initTable(): void {
        FactorialTable._table[0] = 1;
        for (let i = 1;i <= FactorialTable.MAX_INTEGER;i++) {
            FactorialTable._table[i] = FactorialTable._table[i - 1] * i;
        }
    }

    public static get(x: number): number {
        if (!FactorialTable._inited) {
            FactorialTable._initTable();
            FactorialTable._inited = true;
        }
        let res = FactorialTable._table[x];
        return res == undefined ? NaN : res;
    }

}

export class SpecialFunction {

    /**
     * Checks if a 32-bit integer is a power of two.
     * @param n 
     */
    public static isPowerOfTwoN(n: number): boolean {
        return ((n !== 0) && !(n & (n - 1)));
    }

    /**
     * Finds the next power of two that is closet to x.
     * @param x 
     */
    public static nextPowerOfTwo(x: number): number {
        if (x < 0) {
            return 0;
        } else {
            return Math.pow(2, Math.ceil(Math.log(x) / LOGE2));
        }
    }
    
    /**
     * Computes log(gamma(x)) for a nonnegative x.
     * 
     * This function is based on Netlib's Fortran routine.
     * References:
     * 
     *   1) W. J. Cody and K. E. Hillstrom, 'Chebyshev Approximations for
     *      the Natural Logarithm of the Gamma Function,' Math. Comp. 21,
     *      1967, pp. 198-203.
     *   2) K. E. Hillstrom, ANL/AMD Program ANLC366S, DGAMMA/DLGAMA, May,
     *      1969.
     *   3) Hart, Et. Al., Computer Approximations, Wiley and sons, New
     *      York, 1968.

     *   Authors: W. J. Cody and L. Stoltz
     *            Argonne National Laboratory
     * @param x 
     */
    public static gammaln(x: number): number {
        const D1 = -5.772156649015328605195174e-1;
        const P1 = [4.945235359296727046734888e0, 2.018112620856775083915565e2,
                    2.290838373831346393026739e3,1.131967205903380828685045e4,
                    2.855724635671635335736389e4,3.848496228443793359990269e4,
                    2.637748787624195437963534e4,7.225813979700288197698961e3];
        const Q1 = [6.748212550303777196073036e1,1.113332393857199323513008e3,
                    7.738757056935398733233834e3,2.763987074403340708898585e4,
                    5.499310206226157329794414e4,6.161122180066002127833352e4,
                    3.635127591501940507276287e4,8.785536302431013170870835e3];
        const D2 = 4.227843350984671393993777e-1;
        const P2 = [4.974607845568932035012064e0,5.424138599891070494101986e2,
                    1.550693864978364947665077e4,1.847932904445632425417223e5,
                    1.088204769468828767498470e6,3.338152967987029735917223e6,
                    5.106661678927352456275255e6,3.074109054850539556250927e6];
        const Q2 = [1.830328399370592604055942e2,7.765049321445005871323047e3,
                    1.331903827966074194402448e5,1.136705821321969608938755e6,
                    5.267964117437946917577538e6,1.346701454311101692290052e7,
                    1.782736530353274213975932e7,9.533095591844353613395747e6];
        const D4 = 1.791759469228055000094023;
        const P4 = [1.474502166059939948905062e4,2.426813369486704502836312e6,
                    1.214755574045093227939592e8,2.663432449630976949898078e9,
                    2.940378956634553899906876e10,1.702665737765398868392998e11,
                    4.926125793377430887588120e11,5.606251856223951465078242e11];
        const Q4 = [2.690530175870899333379843e3,6.393885654300092398984238e5,
                    4.135599930241388052042842e7,1.120872109616147941376570e9,
                    1.488613728678813811542398e10,1.016803586272438228077304e11,
                    3.417476345507377132798597e11,4.463158187419713286462081e11];
        const C = [-1.910444077728e-03,8.4171387781295e-04,
                   -5.952379913043012e-04,7.93650793500350248e-04,
                   -2.777777777777681622553e-03,8.333333333333333331554247e-02,
                    5.7083835261e-03];
        let res: number, corr: number, tmp1: number, tmp2: number, xDen: number, xNum: number, i: number;
        if (x > 0 && x < 2.55e305) {
            if (x <= 2.22e-16) {
                // (0, eps]
                res = -Math.log(x);
            } else if (x < 1.5) {
                // (eps, 1.5]
                if (x < 0.6796875) {
                    corr = -Math.log(x);
                    tmp1 = x;
                } else {
                    corr = 0;
                    tmp1 = (x - 0.5) - 0.5;
                }
                if (x <= 0.5 || x >= 0.6796875) {
                    xDen = 1;
                    xNum = 0;
                    for (i = 0;i < 8;i++) {
                        xNum = xNum * tmp1 + P1[i];
                        xDen = xDen * tmp1 + Q1[i];
                    }
                    res = corr + (tmp1 * (D1 + tmp1 * (xNum / xDen)));
                } else {
                    tmp1 = (x - 0.5) - 0.5;
                    xDen = 1;
                    xNum = 0;
                    for (i = 0;i < 8;i++) {
                        xNum = xNum * tmp1 + P2[i];
                        xDen = xDen * tmp1 + Q2[i];
                    }
                    res = corr + tmp1 * (D2 + tmp1 * (xNum / xDen));
                }
            } else if (x <= 4) {
                // (1.5, 4]
                tmp2 = x - 2;
                xDen = 1;
                xNum = 0;
                for (i = 0;i < 8;i++) {
                    xNum = xNum * tmp2 + P2[i];
                    xDen = xDen * tmp2 + Q2[i];
                }
                res = tmp2 * (D2 + tmp2 * (xNum / xDen));
            } else if (x <= 12) {
                // (4, 12]
                tmp1 = x - 4;
                xDen = -1;
                xNum = 0;
                for (i = 0;i < 8;i++) {
                    xNum = xNum * tmp1 + P4[i];
                    xDen = xDen * tmp1 + Q4[i];
                }
                res = D4 + tmp1 * (xNum / xDen);
            } else {
                // (12, Inf)
                res = 0;
                if (x <= 2.25e76) {
                    res = C[6];
                    tmp2 = x * x;
                    for (i = 0;i < 6;i++) {
                        res = res / tmp2 + C[i];
                    }
                }
                res = res / x;
                corr = Math.log(x);
                res = res + 0.9189385332046727417803297 - 0.5 * corr;
                res = res + x * (corr - 1);
            }
        } else {
            if (x < 0) {
                throw new Error('Input must be nonnegative.');
            } else {
                res = Infinity;
            }
        }
        return res;
    }

    /**
     * Gamma function.
     * 
     * This function is based on Netlib's Fortran routine.
     * References: "An Overview of Software Development for Special
     *              Functions", W. J. Cody, Lecture Notes in Mathematics,
     *              506, Numerical Analysis Dundee, 1975, G. A. Watson
     *              (ed.), Springer Verlag, Berlin, 1976.
     *
     *              Computer Approximations, Hart, Et. Al., Wiley and
     *              sons, New York, 1968.
     *
     *  Latest modification: October 12, 1989
     *
     *  Authors: W. J. Cody and L. Stoltz
     *           Applied Mathematics Division
     *           Argonne National Laboratory
     *           Argonne, IL 60439
     * @param x 
     */
    public static gamma(x: number): number {
        const P = [-1.71618513886549492533811e+0, 2.47656508055759199108314e+1,
                   -3.79804256470945635097577e+2, 6.29331155312818442661052e+2,
                    8.66966202790413211295064e+2,-3.14512729688483675254357e+4,
                   -3.61444134186911729807069e+4, 6.64561438202405440627855e+4];
        const Q = [-3.08402300119738975254353e+1, 3.15350626979604161529144e+2,
                   -1.01515636749021914166146e+3,-3.10777167157231109440444e+3,
                    2.25381184209801510330112e+4, 4.75584627752788110767815e+3,
                   -1.34659959864969306392456e+5,-1.15132259675553483497211e+5];
        const C = [-1.910444077728e-3,8.4171387781295e-4,
                   -5.952379913043012e-4,7.93650793500350248e-4,
                   -2.777777777777681622553e-3,8.333333333333333331554247e-2,
                    5.7083835261e-3];
        let parity = false;
        let fact = 1;
        let n = 0;
        let tmp: number, res: number, z: number, i: number;
        if (x <= 0) {
            // x <= 0
            x = -x;
            tmp = Math.floor(x);
            res = x - tmp;
            if (res !== 0) {
                if (tmp !== Math.floor(tmp * 0.5) * 2) {
                    parity = true;
                }
                fact = -Math.PI / Math.sin(Math.PI * res);
                x += 1;
            } else {
                return Infinity;
            }
        }
        if (x < 2.22e-16) {
            // (0, eps)
            if (x >= 2.23e-308) {
                res = 1 / x;
            } else {
                return Infinity;
            }
        } else if (x < 12) {
            tmp = x;
            if (x < 1) {
                // [eps, 1)
                z = x;
                x += 1;
            } else {
                // [1, 12)
                n = Math.floor(x) - 1;
                x -= n;
                z = x - 1;
            }
            // evaluate approximation for the interval (1, 2)
            let xNum = 0;
            let xDen = 1;
            for (i = 0;i < 8;i++) {
                xNum = (xNum + P[i]) * z;
                xDen = xDen * z + Q[i];
            }
            res = xNum / xDen + 1;
            if (tmp < x) {
                // adjust result for the interval (0, 1)
                res /= tmp;
            } else if (tmp > x) {
                // adjust result for the interval (2, 12)
                for (i = 0;i < n;i++) {
                    res *= x;
                    x += 1;
                }
            }
        } else {
            // [12, Inf)
            if (x <= 171.624) {
                tmp = x * x;
                z = C[6];
                for (i = 0;i < 6;i++) {
                    z = z / tmp + C[i];
                }
                z = z / x - x + 0.9189385332046727417803297;
                z = z + (x - 0.5) * Math.log(x);
                res = Math.exp(z);
            } else {
                return Infinity;
            }
        }
        // final adjustments
        if (parity) {
            res = -res;
        }
        if (fact !== 1) {
            res = fact / res;
        }
        return res;
    }

    public static factorial(x: number): number {
        if (isNaN(x) || x < 0 || Math.floor(x) !== x) {
            throw new Error('Input must be a nonnegative integer.');
        }
        if (x <= FactorialTable.MAX_INTEGER) {
            return FactorialTable.get(x);
        } else {
            return Infinity;
        }
    }

    /**
     * Error functions.
     * 
     * This function is based on Netlib's Fortran routine.
     * Author: W. J. Cody
     *         Mathematics and Computer Science Division
     *         Argonne National Laboratory
     *         Argonne, IL 60439
     * @param x 
     * @param type 0 - erf(x), 1 - erfc(x), 2 - exp(x*x)*erfc(x)
     */
    public static calerf(x: number, type: number) {
        const A = [3.16112374387056560   ,1.13864154151050156e2,
                   3.77485237685302021e02,3.20937758913846947e3,
                   1.85777706184603153e-1];
        const B = [2.36012909523441209e1,2.44024637934444173e2,
                   1.28261652607737228e3,2.84423683343917062e3];
        const C = [5.64188496988670089e-1,8.88314979438837594,
                   6.61191906371416295e01,2.98635138197400131e2,
                   8.81952221241769090e02,1.71204761263407058e3,
                   2.05107837782607147e03,1.23033935479799725e3,
                   2.15311535474403846e-8];
        const D = [1.57449261107098347e1,1.17693950891312499e2,
                   5.37181101862009858e2,1.62138957456669019e3,
                   3.29079923573345963e3,4.36261909014324716e3,
                   3.43936767414372164e3,1.23033935480374942e3];
        const P = [3.05326634961232344e-1,3.60344899949804439e-1,
                   1.25781726111229246e-1,1.60837851487422766e-2,
                   6.58749161529837803e-4,1.63153871373020978e-2];
        const Q = [2.56852019228982242   ,1.87295284992346047,
                   5.27905102951428412e-1,6.05183413124413191e-2,
                   2.33520497626869185e-3];
        const SQRPI = 0.56418958354775628695;
        let y = Math.abs(x);
        let flag = false;
        let i: number, res: number, del: number;
        let ySq: number, xNum: number, xDen: number;
        if (y <= 0.46875) {
            // |x| <= 0.46875
            ySq = y > 1.11e-16 ? y * y : 0;
            xNum = A[4] * ySq;
            xDen = ySq;
            for (i = 0;i < 3;i++) {
                xNum = (xNum + A[i]) * ySq;
                xDen = (xDen + B[i]) * ySq;
            }
            res = x * (xNum + A[3]) / (xDen + B[3]);
            if (type !== 0) res = 1 - res;
            if (type === 2) res = Math.exp(ySq) * res;
            return res;
        } else if (y <= 4) {
            // 0.46875 < x <= 4
            xNum = C[8] * y;
            xDen = y;
            for (i = 0;i < 7;i++) {
                xNum = (xNum + C[i]) * y;
                xDen = (xDen + D[i]) * y;
            }
            res = (xNum + C[7]) / (xDen + D[7]);
            if (type !== 2) {
                ySq = Math.floor(y * 16) / 16;
                del = (y - ySq) * (y + ySq);
                res = Math.exp(-ySq * ySq) * Math.exp(-del) * res;
            }
        } else {
            // x > 4
            res = 0;
            if (y >= 26.543) {
                if (type !== 2 || y >= 2.53e307) {
                    flag = true;
                } else if (y >= 6.71e7) {
                    res = SQRPI / y;
                    flag = true;
                }
            }
            if (!flag) {
                ySq = 1 / (y * y);
                xNum = P[5] * ySq;
                xDen = ySq;
                for (i = 0;i < 4;i++) {
                    xNum = (xNum + P[i]) * ySq;
                    xDen = (xDen + Q[i]) * ySq;
                }
                res = ySq * (xNum + P[4]) / (xDen + Q[4]);
                res = (SQRPI - res) / y;
                if (type !== 2) {
                    ySq = Math.floor(y * 16) / 16;
                    del = (y - ySq) * (y + ySq);
                    res = Math.exp(-ySq * ySq) * Math.exp(-del) * res;
                }
            }
        }
        // fix up for negative argument, erf, etc.
        if (type === 0) {
            res = (0.5 - res) + 0.5;
            if (x < 0) res = -res;
        } else if(type === 1) {
            if (x < 0) res = 2 - res;
        } else {
            if (x < 0) {
                if (x < -26.628) {
                    res = Infinity;
                } else {
                    ySq = Math.ceil(x * 16) / 16;
                    del = (x - ySq) * (x + ySq);
                    y = Math.exp(ySq * ySq) * Math.exp(del);
                    res = (y + y) - res;
                }
            }
        }
        return res;
    }

    /**
     * Error function.
     * @param x 
     */
    public static erf(x: number): number {
        return SpecialFunction.calerf(x, 0);
    }

    /**
     * Complementary error function.
     * @param x 
     */
    public static erfc(x: number): number {
        return SpecialFunction.calerf(x, 1);
    }

    /**
     * Scaled complementary error function: exp(x^2) * erfc(x).
     * @param x 
     */
    public static erfcx(x: number): number {
        return SpecialFunction.calerf(x, 2);
    }

}
