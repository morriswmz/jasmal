import { DataBlock } from '../../../storage';
import { EPSILON } from '../../../constant';
import { CMathHelper } from '../../../helper/mathHelper';

export class QR {

    /**
     * QR decomposition with pivoting using Householder reflections.
     * See Algorithm 5.4.1 in Matrix Computations (4th edition) for details.
     * @param m 
     * @param n 
     * @param a (Input/Output) Input matrix. The lower triangular part
     *          (including the main diagonal) will be overwritten with the
     *          Householder vector info and the remaining upper triangular part
     *          corresponds to the elements in R.
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
                    s = 0.0;
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

    public static ind2p(n: number, ind: ArrayLike<number>, p: DataBlock): void {
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
     * Performs QR decomposition with column pivoting such that AP = QR.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param a (Input/Output) Matrix A. Will be overwritten with R.
     * @param q (Output) Matrix Q. Must be initialized with zeros.
     * @param p (Output) Matrix P. Must be initialized with zeros.
     */
    public static qrpf(m: number, n: number, a: DataBlock, q: DataBlock, p: DataBlock): void {
        let d = new Array(Math.min(m, n));
        let ind = new Array(n);
        QR.qrp(m, n, a, d, ind);
        QR.qrtrans(m, n, a, d, q);
        QR.ind2p(n, ind, p);
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
     * @param b (Input/Destroyed) m x p matrix B. 
     * @param x (Output) n x p matrix X. Must be initialized to zeros.
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

    /**
     * Complex QR decomposition with pivoting using Householder reflections.
     * See Algorithm 5.4.1 in Matrix Computations (4th edition) for details.
     * @param m 
     * @param n 
     * @param ar (Input/Output) Real part of the input matrix. The lower
     *           triangular part (including the main diagonal) will be
     *           overwritten with the Householder vector info and the remaining
     *           upper triangular part corresponds to the real part of the
     *           elements in R.
     * @param ai (Input/Output) Imaginary part of the input matrix. The lower
     *           triangular part (including the main diagonal) will be
     *           overwritten with the Householder vector info and the remaining
     *           upper triangular part corresponds to the imaginary part of the
     *           elements in R.
     * @param d (Output) Magnitudes of the diagonal elements of R.
     * @param phr (Output) Phase info of the diagonal elements of R.
     * @param phi (Output) Phase info of the diagonal elements of R.
     * @param ind (Output) Stores permutation information. ind[i] stores the
     *            index of the column being swapped with the column being worked
     *            on at the i-th step.
     */
    public static cqrp(m: number, n: number, ar: DataBlock, ai: DataBlock,
                       d: DataBlock, phr: DataBlock, phi: DataBlock, ind: DataBlock): void {
        let i: number, j: number, k: number, r: number, l: number;
        let s: number, si: number, sr: number, t: number, f: number, g: number, h: number;
        let tau = 0;
        l = Math.min(m, n);
        let c = new Array(n);
        for (i = 0;i < l;i++) {
            d[i] = 0;
            phr[i] = 1.0;
            phi[i] = 0.0;
        }
        k = 0;
        for (j = 0;j < n;j++) {
            s = 0;
            for (i = 0;i < m;i++) {
                s += ar[i * n + j] * ar[i * n + j] + ai[i * n + j] * ai[i * n + j];
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
                    t = ar[i * n + r];
                    ar[i * n + r] = ar[i * n + k];
                    ar[i * n + k] = t;
                    t = ai[i * n + r];
                    ai[i * n + r] = ai[i * n + k];
                    ai[i * n + k] = t;
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
            // s = ||A[r:end,r]||_2^2 or ||a||_2^2
            for (i = r;i < m;i++) {
                s += ar[i * n + r] * ar[i * n + r] + ai[i * n + r] * ai[i * n + r];
            }
            if (s) {
                // f <- |A[r,r]|
                f = CMathHelper.length2(ar[r * n + r], ai[r * n + r]);
                // g <- sqrt(s) = ||a||
                g = Math.sqrt(s);
                // \beta = 2/(w^H w) = 1/(s + |A[r,r]| * sqrt(s))
                // h <- - 1/\beta
                h = - f * g - s;
                // Update A[r,r] according to Householder refection:
                //  w <- a + exp(j\theta) ||a|| e_1, then
                //  H a <- - exp(j\theta) ||a|| e_1
                if (f) {
                    phr[r] = ar[r * n + r] / f;
                    phi[r] = ai[r * n + r] / f;
                    ar[r * n + r] += phr[r] * g;
                    ai[r * n + r] += phi[r] * g;
                } else {
                    phr[r] = 1.0;
                    phi[r] = 0.0;
                    ar[r * n + r] = g;
                    ai[i * n + r] = 0.0;
                }
                // apply Householder transform to A[r:end,r+1:end]
                //  (I - \beta ww^H) A = A - (\beta w)(w^H [a_1, a_2, ..., a_n])
                for (j = r + 1;j < n;j++) {
                    sr = 0.0;
                    si = 0.0;
                    // w^H a_j
                    for (k = r;k < m;k++) {
                        sr += ar[k * n + r] * ar[k * n + j] + ai[k * n + r] * ai[k * n + j];
                        si += ar[k * n + r] * ai[k * n + j] - ai[k * n + r] * ar[k * n + j];
                    }
                    // -\beta w^H a_j
                    sr /= h;
                    si /= h;
                    //  a_j + (-\beta w^H a_j) w
                    for (k = r;k < m;k++) {
                        ar[k * n + j] += sr * ar[k * n + r] - si * ai[k * n + r];
                        ai[k * n + j] += sr * ai[k * n + r] + si * ar[k * n + r];
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
                c[j] -= ar[r * n + j] * ar[r * n + j] + ai[r * n + j] * ai[r * n + j];
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

    /**
     * Accumulates complex Householder transforms and form matrices Q, R.
     * @param m 
     * @param n 
     * @param ar (Input/Output)
     * @param ai (Input/Output)
     * @param d (Input)
     * @param phr (Input)
     * @param phi (Input)
     * @param qr (Output) Real part of Q. Must be initialized with zeros.
     * @param qi (Output) Imaginary part of Q. Must be initialized with zeros.
     */
    public static cqrtrans(m: number, n: number, ar: DataBlock, ai: DataBlock,
                           d: ArrayLike<number>, phr: ArrayLike<number>,
                           phi: ArrayLike<number>, qr: DataBlock, qi: DataBlock): void {
        let i: number, j: number, k: number;
        let g: number, h: number, si: number, sr: number;
        // init q: m x m
        for (i = 0;i < m;i++) {
            qr[i * m + i] = 1.0;
        }
        // accumulate transforms
        for (i = Math.min(m, n) - 1;i >= 0;i--) {
            g = d[i];
            if (g) {
                // (I - \beta vv^H) Q = Q - (\beta v)(v^H [q_1 q_2 ...])
                // h <- -\beta
                // Note that \beta = 2/v^Hv = 1/(||a||^2 + |a_1| ||a||) = 1/g/a[i,i]
                h = - 1.0 / g / CMathHelper.length2(ar[i * n + i], ai[i * n + i]);
                for (j = i;j < m;j++) {
                    // compute v^H q_j
                    sr = 0.0;
                    si = 0.0;
                    for (k = i;k < m;k++) {
                        sr += ar[k * n + i] * qr[k * m + j] + ai[k * n + i] * qi[k * m + j];
                        si += ar[k * n + i] * qi[k * m + j] - ai[k * n + i] * qr[k * m + j];
                    }
                    sr *= h;
                    si *= h;
                    // update q_j <- q_j - \beta v v^H q_j
                    for (k = i;k < m;k++) {
                        qr[k * m + j] += sr * ar[k * n + i] - si * ai[k * n + i];
                        qi[k * m + j] += sr * ai[k * n + i] + si * ar[k * n + i];
                    }
                }
            }
            // update a
            //  v <- a + exp(j\theta) ||a|| e_1, then
            //  H a <- - exp(j\theta) ||a|| e_1
            ar[i * n + i] = -g * phr[i];
            ai[i * n + i] = -g * phi[i];
            for (k = i + 1;k < m;k++) {
                ar[k * n + i] = 0.0;
                ai[k * n + i] = 0.0;
            }
        }
    }

    /**
     * Performs complex QR decomposition with column pivoting such that AP = QR.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param ar (Input/Output) Real part of A. Will be overwritten with the
     *                          real part of R.
     * @param ai (Input/Output) Imaginary part of A. Will be overwritten with
     *                          the imaginary part of R.
     * @param qr (Output) Real part of Q. Must be initialized with zeros.
     * @param qi (Output) Imaginary part of Q. Must be initialized with zeros.
     * @param p (Output) Matrix P. Must be initialized with zeros.
     */
    public static cqrpf(m: number, n: number, ar: DataBlock, ai: DataBlock,
                        qr: DataBlock, qi: DataBlock, p: DataBlock): void {
        let l = Math.min(m, n);
        let d = new Array(l);
        let phr = new Array(l);
        let phi = new Array(l);
        let ind = new Array(n);
        QR.cqrp(m, n, ar, ai, d, phr, phi, ind);
        QR.cqrtrans(m, n, ar, ai, d, phr, phi, qr, qi);
        QR.ind2p(n, ind, p);
    }

    /**
     * Obtains the least square solution using QR decomposition such that
     * ||A X - B|| is minimized (complex case).
     * When A is rank deficient, there are infinitely many solutions. This
     * function will only return one solution satisfying the normal equation
     * (assuming free variables are all zeros).
     * @param m 
     * @param n 
     * @param p 
     * @param ar (Input/Destroyed) Output from cqrp().
     * @param ai (Input/Destroyed) Output from cqrp().
     * @param d (Input) Output from cqrp().
     * @param phr (Input) Output from cqrp().
     * @param ind (Input) Output from cqrp().
     * @param br (Input/Destroyed) Real part of the m x p matrix B. 
     * @param bi (Input/Destroyed) Imaginary part of the m x p matrix B. 
     * @param xr (Output) Real part of the n x p matrix X. Must be initialized
     *           to zeros.
     * @param xi (Output) Imaginary part of the n x p matrix X. Must be
     *           initialized to zeros.
     * @returns The estimated rank of A.
     */
    public static cqrpsol(m: number, n: number, p: number, ar: DataBlock,
                          ai: DataBlock, d: ArrayLike<number>, ind: ArrayLike<number>,
                          phr: ArrayLike<number>, phi: ArrayLike<number>,
                          br: DataBlock, bi: DataBlock, xr: DataBlock, xi: DataBlock): number {
        let i: number, j: number, k: number, l: number;
        let f: number, g: number, h: number, r: number, si: number, sr: number, t: number;
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
                // (I - \beta vv^H) B = B - (\beta v)(v^H [b_1 b_2 ...])
                // h <- -\beta
                // Note that -\beta = 2/v^v = 1/(||a||^2 + a_1 ||a||) = -1/g/a[i,i]
                h = - 1.0 / g / CMathHelper.length2(ar[i * n + i], ai[i * n + i]);
                for (j = 0;j < p;j++) {
                    // compute v^H b_j
                    sr = 0.0;
                    si = 0.0;
                    for (k = i;k < m;k++) {
                        sr += ar[k * n + i] * br[k * p + j] + ai[k * n + i] * bi[k * p + j];
                        si += ar[k * n + i] * bi[k * p + j] - ai[k * n + i] * br[k * p + j];
                    }
                    sr *= h;
                    si *= h;
                    // update b_j <- b_j - v (\beta v^H b_j)
                    for (k = i;k < m;k++) {
                        br[k * p + j] += ar[k * n + i] * sr - ai[k * n + i] * si;
                        bi[k * p + j] += ar[k * n + i] * si + ai[k * n + i] * sr;
                    }
                }
            }
        }
        // solve R1^-1 Q1^T B1 using back substitution
        for (j = 0;j < p;j++) {
            for (i = r - 1;i >= 0;i--) {
                sr = br[i * p + j];
                si = bi[i * p + j];
                for (k = i + 1;k < r;k++) {
                    sr -= ar[i * n + k] * xr[k * p + j] - ai[i * n + k] * xi[k * p + j];
                    si -= ar[i * n + k] * xi[k * p + j] + ai[i * n + k] * xr[k * p + j]
                }
                // diagonal elements of R are stored in d
                [xr[i * p + j], xi[i * p + j]] = CMathHelper.cdivCC(sr, si, -d[i] * phr[i], -d[i] * phi[i]);
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
                    t = xr[ind2[i] * p + j];
                    xr[ind2[i] * p + j] = xr[ind2[ind[i]] * p + j];
                    xr[ind2[ind[i]] * p + j] = t;
                    t = xi[ind2[i] * p + j];
                    xi[ind2[i] * p + j] = xi[ind2[ind[i]] * p + j];
                    xi[ind2[ind[i]] * p + j] = t;
                }
                t = ind2[i];
                ind2[i] = ind2[ind[i]];
                ind2[ind[i]] = t;
            }
        }
        return r;
    }

}