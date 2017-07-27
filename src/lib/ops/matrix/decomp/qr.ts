// TODO: add QR decomposition

import { DataBlock } from '../../../storage';
import { EPSILON } from '../../../constant';

export class QR {

    /**
     * QR decomposition with pivoting using Householder reflections.
     * See Algorithm 5.4.1 in Matrix Computations (4th edition) for details.
     * @param m 
     * @param n 
     * @param a (Input/Output) Input matrix. The lower triangular part
     *          (including the main diagonal) will be overwritten with the
     *          Householder vector info and the remaining upper triangular part
     *          corresponds the elements in R.
     * @param d (Output) Diagonal elements of R.
     * @param ind (Output) Stores permutation information. ind[i] stores the
     *            index of the column being swapped with the column being worked
     *            on at the i-th step.
     */
    public static qrp(m: number, n: number, a: DataBlock, d: DataBlock, ind: DataBlock): void {
        let i: number, j: number, k: number, r: number, l: number;
        let s: number, t: number, f: number, g: number, h: number;
        let tau = 0;
        l = Math.min(m, n);
        let c = new Array(n);
        for (i = 0;i < l;i++) {
            d[i] = 0;
        }
        k = 0;
        for (j = 0;j < n;j++) {
            s = 0;
            for (i = 0;i < m;i++) {
                s += a[i * n + j] * a[i * n + j];
            }
            c[j] = s;
            if (s > tau) {
                tau = s;
                k = j;
            }
        }
        r = 0;
        
        while (tau > 0 && r < l) {
            if (k !== r) {
                // swap columns
                for (let i = 0;i < m;i++) {
                    t = a[i * n + r];
                    a[i * n + r] = a[i * n + k];
                    a[i * n + k] = t;
                }
                // record which column is swapped with the current column
                ind[r] = k;
                t = c[r];
                c[r] = c[k];
                c[k] = t;
            } else {
                ind[r] = r;
            }
            // apply Householder transform to the r-th column
            s = 0.0;
            // s = ||A[r:end,r]||_2^2
            for (i = r;i < m;i++) {
                s += a[i * n + r] * a[i * n + r];
            }
            if (s) {
                f = a[r * n + r]; // f <- A[r,r]
                // g <- sqrt(s)
                g = f >= 0 ? -Math.sqrt(s) : Math.sqrt(s);
                // h <- - 1/\beta
                h = f * g - s;
                // store the Householder vector
                // w <- A[r,r] >= 0 ? a + ||a||e_1 : a - ||a||e_1 
                a[r * n + r] = f - g;
                // apply Householder transform to A[r:end,r+1:end]
                //  (I - \beta ww^T) A = A - (\beta w)(w^T [a_1, a_2, ..., a_n])
                for (j = r + 1;j < n;j++) {
                    s = 0.0
                    // w^T a_j
                    for (k = r;k < m;k++) {
                        s += a[k * n + r] * a[k * n + j];
                    }
                    // -\beta w^T a_j
                    f = s / h;
                    //  a_j + (-\beta w^T a_j) w
                    for (k = r;k < m;k++) {
                        a[k * n + j] += f * a[k * n + r];
                    }
                }
                d[r] = g;
            } else {
                d[r] = 0.0;
            }
            // update c
            for (j = r + 1;j < n;j++) {
                // Remove the square of the first element in the j-th column
                // so that c[j] now stores ||A[r+1:end,j]||_2^2. 
                c[j] -= a[r * n + j] * a[r * n + j];
            }
            // update tau
            tau = 0.0;
            for (j = r + 1;j < n;j++) {
                if (c[j] > tau) {
                    tau = c[j];
                    k = j;
                }
            }
            r++;
        }
        // set remaining ind
        for (;r < n;r++) {
            ind[r] = r;
        }
    }

    public static qrtrans(m: number, n: number, a: DataBlock, d: ArrayLike<number>, q: DataBlock): void {
        let i: number, j: number, k: number;
        let g: number, h: number, s: number;
        // init q: m x m
        for (i = 0;i < m;i++) {
            q[i * m + i] = 1.0;
        }
        // accumulate transforms
        for (i = Math.min(m, n) - 1;i >= 0;i--) {
            g = d[i];
            if (g) {
                // (I - \beta vv^T) Q = Q - (\beta v)(v^T [q_1 q_2 ...])
                // h <- \beta
                // Note that \beta = 2/v^v = 1/(||a||^2 - a_1 ||a||) = -1/g/a[i,i]
                h = - 1.0 / g / a[i * n + i];
                for (j = i;j < m;j++) {
                    // compute v^T q_j
                    s = 0.0;
                    for (k = i;k < m;k++) {
                        s += a[k * n + i] * q[k * m + j];
                    }
                    s *= h;
                    // update q_j <- q_j - \beta v v^T q_j
                    for (k = i;k < m;k++) {
                        q[k * m + j] -= a[k * n + i] * s;
                    }
                }
            }
            // update a
            a[i * n + i] = g;
            for (k = i + 1;k < m;k++) {
                a[k * n + i] = 0.0;
            }
        }
    }

    /**
     * Performs QR decomposition with column pivoting such that AP = QR.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param a (Input/Output) Matrix A. Will be overwritten with R.
     * @param q (Output) Matrix Q. Must be all zeros.
     * @param p (Output) Matrix P. Must be all zeros.
     */
    public static qrpf(m: number, n: number, a: DataBlock, q: DataBlock, p: DataBlock): void {
        let d = new Array(Math.min(m, n));
        let ind = new Array(n);
        QR.qrp(m, n, a, d, ind);
        QR.qrtrans(m, n, a, d, q);
        // fill p
        let i: number, t: number;
        let perm = new Array(n);
        for (i = 0;i < n;i++) {
            perm[i] = i;
        }
        for (i = 0;i < n;i++) {
            if (ind[i] !== i) {
                t = perm[i];
                perm[i] = perm[ind[i]];
                perm[ind[i]] = t;
            }
        }
        for (let i = 0;i < n;i++) {
            p[perm[i] * n + i] = 1.0;
        }
    }

    /**
     * Obtains the least square solution using QR decomposition such that
     * ||A X - B|| is minimized.
     * When A is rank deficient, there are infinitely many solutions. This
     * function will only return one solution satisfying the normal equation
     * (assuming free variables are all zeros).
     * @param m 
     * @param n 
     * @param p 
     * @param a (Input/Destroyed) Output from qrp().
     * @param d (Input) Output from qrp().
     * @param ind (Input) Output from qrp().
     * @param b (Input/Destroyed) m x p matrix. 
     * @param x (Output) n x p matrix. Must be initialized to zeros.
     * @returns The estimated rank of A.
     */
    public static qrpsol(m: number, n: number, p: number, a: DataBlock,
                         d: ArrayLike<number>, ind: ArrayLike<number>,
                         b: DataBlock, x: DataBlock): number {
        let i: number, j: number, k: number, l: number;
        let f: number, g: number, h: number, r: number, s: number, t: number;
        let tol: number;
        // determine the rank of a
        // Here we just test if the diagonal element in R is sufficiently small
        // compared with the largest diagonal element.
        l = Math.min(m, n);
        tol = Math.abs(d[0]) * EPSILON;
        if (tol === 0) {
            // all zeros
            return 0;
        }
        r = l;
        for (i = 1;i < l;i++) {
            if (Math.abs(d[i]) < tol) {
                r = i;
                break;
            }
        }
        // apply transforms to b
        // C1 = Q1^T B1
        for (i = 0;i < l;i++) {
            g = d[i];
            if (g) {
                // (I - \beta vv^T) B = B - (\beta v)(v^T [b_1 b_2 ...])
                // h <- \beta
                // Note that \beta = 2/v^v = 1/(||a||^2 - a_1 ||a||) = -1/g/a[i,i]
                h = - 1.0 / g / a[i * n + i];
                for (j = 0;j < p;j++) {
                    // compute v^T b_j
                    s = 0.0;
                    for (k = i;k < m;k++) {
                        s += a[k * n + i] * b[k * p + j];
                    }
                    s *= h;
                    // update b_j <- b_j - \beta v v^T b_j
                    for (k = i;k < m;k++) {
                        b[k * p + j] -= a[k * n + i] * s;
                    }
                }
            }
        }
        // solve R1^-1 Q1^T B1 using back substitution
        for (j = 0;j < p;j++) {
            for (i = r - 1;i >= 0;i--) {
                s = b[i * p + j];
                for (k = i + 1;k < r;k++) {
                    s -= a[i * n + k] * x[k * p + j];
                }
                // diagonal elements of R are stored in d
                x[i * p + j] = s / d[i];
            }
        }
        // apply permutation
        // Note that ind[] records the column swapping for a, we need to convert
        // it to row swapping records for x.
        let ind2 = new Array(n);
        for (i = 0;i < n;i++) {
            ind2[i] = i;
        }
        for (i = 0;i < n;i++) {
            if (ind[i] !== i) {
                for (j = 0;j < p;j++) {
                    t = x[ind2[i] * p + j];
                    x[ind2[i] * p + j] = x[ind2[ind[i]] * p + j];
                    x[ind2[ind[i]] * p + j] = t;
                }
                t = ind2[i];
                ind2[i] = ind2[ind[i]];
                ind2[ind[i]] = t;
            }
        }
        return r;
    }

}