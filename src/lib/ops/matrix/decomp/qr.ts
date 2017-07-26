// TODO: add QR decomposition

import { DataBlock } from '../../../storage';

export class QR {

    public static qrp(m: number, n: number, a: DataBlock, d: DataBlock, p: DataBlock): void {
        let i: number, j: number, k: number, r: number, l: number;
        let s: number, t: number, scale: number, f: number, g: number, h: number;
        let tau = 0;
        l = Math.min(m, n);
        let c = new Array(n);
        for (i = 0;i < l;i++) {
            d[i] = 0;
        }
        for (j = 0;j < n;j++) {
            p[j] = j;
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
                t = p[r];
                p[r] = p[k];
                p[k] = t;
                t = c[r];
                c[r] = c[k];
                c[k] = t;
            }
            // apply Householder transform to the r-th column
            // calculate scaling factor
            scale = 1.0;
            for (i = r;i < m;i++) {
                scale += Math.abs(a[i * n + r]);
            }
            if (scale) {
                s = 0.0;
                // s = ||A[r:end,r]||_2^2
                for (i = r;i < m;i++) {
                    a[i * n + r] /= scale;
                    s += a[i * n + r] * a[i * n + r];
                }
                f = a[r * n + r]; // f <- A[r,r]
                // g <- sqrt(s)
                g = Math.sqrt(s);
                // h <- - 1/\beta
                h = f * g - s;
                // store the Householder vector
                // w <- a - ||a||e_1
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
                // scale back
                for (i = r;i < m;i++) {
                    a[i * n + r] *= scale;
                }
                d[r] = scale * g;
                // update c
                for (j = r + 1;j < n;j++) {
                    c[j] -= a[r * n + j] * a[r * n + j];
                }
            } else {
                d[r] = 0.0;
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
        let perm = new Array(n);
        QR.qrp(m, n, a, d, perm);
        QR.qrtrans(m, n, a, d, q);
        // fill p
        for (let i = 0;i < n;i++) {
            p[i * n + perm[i]] = 1.0;
        }
    }

    public static qrpsol(m: number, n: number, a: DataBlock, b: DataBlock, x: DataBlock): void {

    }

}