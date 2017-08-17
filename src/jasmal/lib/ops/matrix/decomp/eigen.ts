/**
 * The methods in the Eigen class are ported from the Fortran 77 version of
 * EISPACK: http://www.netlib.org/eispack/.
 * All rights reserved by the authors of EISPACK.
 * http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg01379.html
 */
import { DataBlock } from '../../../commonTypes';
import { CMath } from '../../../complexNumber';
import { DataHelper } from '../../../helper/dataHelper';

export class Eigen {

    /**
     * See tred2 in EISPACK.
     * @param n 
     * @param reA 
     * @param diag 
     * @param sub 
     * @param reE 
     */
    public static tred2(n: number, reA: ArrayLike<number>, diag: DataBlock,
                        sub: DataBlock, reE: DataBlock): void {
        let i: number, j: number, k: number, l: number, ii: number, jp1: number;
        let f: number, g: number, h: number, hh: number, scale: number;
        for (i = 0;i < n;i++) {
            for (j = 0;j <= i;j++) {
                reE[i * n + j] = reA[i * n + j];
            }
            diag[i] = reA[(n - 1) * n + i];
        }
        if (n !== 1) {
            for (ii = 1;ii < n;ii++) {
                i = n - ii;
                l = i - 1;
                h = 0.0;
                scale = 0.0;
                if (l >= 1) {
                    // scale rows
                    for (k = 0;k <= l;k++) {
                        scale += Math.abs(diag[k]);
                    }
                }
                if (scale === 0.0) {
                    sub[i] = diag[l];
                    for (j = 0;j <= l;j++) {
                        diag[j] = reE[l * n + j];
                        reE[i * n + j] = 0.0;
                        reE[j * n + i] = 0.0;
                    }
                } else {
                    for (k = 0;k <= l;k++) {
                        diag[k] /= scale;
                        h = h + diag[k] * diag[k];
                    }
                    f = diag[l];
                    g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
                    sub[i] = scale * g;
                    h = h - f * g;
                    diag[l] = f - g;
                    // a * u
                    for (j = 0;j <= l;j++) {
                        sub[j] = 0.0;
                    }
                    for (j = 0;j <= l;j++) {
                        f = diag[j];
                        reE[j * n + i] = f;
                        g = sub[j] + reE[j * n + j] * f;
                        jp1 = j + 1;
                        if (l >= jp1) {
                            for (k = jp1;k <= l;k++) {
                                g += reE[k * n + j] * diag[k];
                                sub[k] += reE[k * n + j] * f;
                            }
                        }
                        sub[j] = g;
                    }
                    // form p
                    f = 0.0;
                    for (j = 0;j <= l;j++) {
                        sub[j] /= h;
                        f += sub[j] * diag[j];
                    }
                    hh = f / (h + h);
                    // form q
                    for (j = 0;j <= l;j++) {
                        sub[j] -= hh * diag[j];
                    }
                    // form reduced a
                    for (j = 0;j <= l;j++) {
                        f = diag[j];
                        g = sub[j];
                        for (k = j;k <= l;k++) {
                            reE[k * n + j] -= f * sub[k] + g * diag[k];
                        }
                        diag[j] = reE[l * n + j];
                        reE[i * n + j] = 0.0;
                    }
                }
                diag[i] = h;
            }
            // accumulation of transformation matrices
            for (i = 1;i < n;i++) {
                l = i - 1;
                reE[(n - 1) * n + l] = reE[l * n + l];
                reE[l * n + l] = 1.0;
                h = diag[i];
                if (h !== 0.0) {
                    for (k = 0;k <= l;k++) {
                        diag[k] = reE[k * n + i] / h;
                    }
                    for (j = 0;j <= l;j++) {
                        g = 0.0;
                        for (k = 0;k <= l;k++) {
                            g += reE[k * n + i] * reE [k * n + j];
                        }
                        for (k = 0;k <= l;k++) {
                            reE[k * n + j] -= g * diag[k];
                        }
                    }
                }
                for (k = 0;k <= l;k++) {
                    reE[k * n + i] = 0.0;
                }
            }
        }
        for (i = 0;i < n;i++) {
            diag[i] = reE[(n - 1) * n + i];
            reE[(n - 1) * n + i] = 0.0;
        }
        reE[(n - 1) * n + n - 1] = 1.0;
        sub[0] = 0.0;
    }

    /**
     * See tql2 in EISPACK.
     * @param n 
     * @param diag 
     * @param sub 
     * @param reE 
     */
    public static tql2(n: number, diag: DataBlock, sub: DataBlock, reE: DataBlock): void {
        let i: number, j: number, k: number, l: number, l1: number, l2: number, m: number;
        let ii: number, mml: number;
        let f: number, g: number, h: number, tst1: number, tst2: number;
        let s: number = 0, s2: number = 0;
        let c: number, c2: number, c3: number = 0, p: number, r: number, dl1: number, el1: number;
        if (n !== 1) {
            for (i = 1;i < n;i++) {
                sub[i - 1] = sub[i];
            }
            f = 0.0;
            tst1 = 0.0;
            sub[n - 1] = 0.0;
            for (l = 0;l < n;l++) {
                j = 0;
                h = Math.abs(diag[l]) + Math.abs(sub[l]);
                if (tst1 < h) {
                    tst1 = h;
                }
                // look for small sub-diagonal element
                for (m = l;m < n;m++) {
                    tst2 = tst1 + Math.abs(sub[m]);
                    // e(n) is always zero, so there is no exit through the bottom
                    // of the loop
                    if (tst2 === tst1) {
                        break;
                    }
                }
                if (m !== l) {
                    do {
                        if (j === 30) {
                            throw new Error('Failed to converge!');
                        }
                        j++;
                        // form shift
                        l1 = l + 1;
                        l2 = l1 + 1;
                        g = diag[l];
                        p = (diag[l1] - g) / (2.0 * sub[l]);
                        r = CMath.length2(p, 1.0);
                        diag[l] = sub[l] / (p + (p >= 0 ? r : -r));
                        diag[l1] = sub[l] * (p + (p >= 0 ? r : -r));
                        dl1 = diag[l1];
                        h = g - diag[l];
                        if (l2 < n) {
                            for (i = l2;i < n;i++) {
                                diag[i] -= h;
                            }
                        }
                        f += h;
                        // ql transform
                        p = diag[m];
                        c = 1.0;
                        c2 = c;
                        el1 = sub[l1];
                        s = 0.0;
                        mml = m - l;
                        for (i = m - 1;i >= l;i--) {
                            c3 = c2;
                            c2 = c;
                            s2 = s;
                            g = c * sub[i];
                            h = c * p;
                            r = CMath.length2(p, sub[i]);
                            sub[i + 1] = s * r;
                            s = sub[i] / r;
                            c = p / r;
                            p = c * diag[i] - s * g;
                            diag[i + 1] = h + s * (c * g + s * diag[i]);
                            // form vector
                            for (k = 0;k < n;k++) {
                                h = reE[k * n + i + 1];
                                reE[k * n + i + 1] = s * reE[k * n + i] + c * h;
                                reE[k * n + i] = c * reE[k * n + i] - s * h;
                            }
                        }
                        p = -s * s2 * c3 * el1 * sub[l] / dl1;
                        sub[l] = s * p;
                        diag[l] = c * p;
                        tst2 = tst1 + Math.abs(sub[l]);
                    } while (tst2 > tst1);
                }
                diag[l] += f;
            }
            // order eigenvalues and eigenvectors in ascending order
            for (ii = 1;ii < n;ii++) {
                i = ii - 1;
                k = i;
                p = diag[i];
                for (j = ii;j < n;j++) {
                    if (diag[j] < p) {
                        k = j;
                        p = diag[j];
                    }
                }
                if (k !== i) {
                    diag[k] = diag[i];
                    diag[i] = p;
                    for (j = 0;j < n;j++) {
                        p = reE[j * n + i];
                        reE[j * n + i] = reE[j * n + k];
                        reE[j * n + k] = p;
                    }
                }
            }
        }
    }

    
    /**
     * Performs eigendecomposition for real symmetric matrices.
     * @param n Dimension of the matrix.
     * @param reA (Input) Matrix data.
     * @param lambda (Output) Eigenvalues.
     * @param reE (Output) Eigenvectors.
     */
    public static eigSym(n: number, reA: ArrayLike<number>, lambda: DataBlock, reE: DataBlock): void {
        let tmpArr = DataHelper.allocateFloat64Array(n);
        Eigen.tred2(n, reA, lambda, tmpArr, reE);
        Eigen.tql2(n, lambda, tmpArr, reE);
    }

    /**
     * Reduces a complex Hermitian matrix to a real symmetric tridiagonal
     * matrix.
     * See htridi.f in EISPACK for details.
     * @param n Dimension of the matrix.
     * @param ar (Input/Output) Real part of the input matrix (n^2). Its lower
     *           triangle will be overwritten with the information about the
     *           unitary transforms.
     * @param ai (Input/Output) Imaginary part of the input matrix (n^2). Its
     *           lower triangle will be overwritten with the information about
     *           the unitary transforms.
     * @param d (Output) Main diagonal elements (n).
     * @param e (Output) Sub diagonal elements (n).
     * @param e2 (Output) Squares of the sub diagonal elements (n).
     * @param tau (Output) Contains information about the transformations, used
     *                     to reconstruct the eigenvectors later (2n).
     */
    public static htridi(n: number, ar: DataBlock, ai: DataBlock, d: DataBlock,
                         e: DataBlock, e2: DataBlock, tau: DataBlock): void {
        let i: number, j: number, k: number, l: number, jp1: number;
        let f: number, fi: number, g: number, gi: number, h: number, hh: number;
        let scale: number, si: number;
        tau[n - 1] = 1.0;
        tau[n + n - 1] = 0.0;
        for (i = 0;i < n;i++) {
            d[i] = ar[i * n + i];
        }
        for (i = n - 1;i >= 0;i--) {
            l = i - 1;
            h = 0.0;
            scale = 0.0;
            if (l >= 0) {
                // scale rows
                for (k = 0;k <= l;k++) {
                    scale += Math.abs(ar[i * n + k]) + Math.abs(ai[i * n + k]);
                }
                if (scale === 0) {
                    tau[l] = 1.0;
                    tau[n + l] = 0.0;
                    e[i] = 0.0;
                    e2[i] = 0.0;
                } else {
                    for (k = 0;k <= l;k++) {
                        ar[i * n + k] /= scale;
                        ai[i * n + k] /= scale;
                        h += ar[i * n + k] * ar[i * n + k] + ai[i * n + k] * ai[i * n + k];
                    }

                    e2[i] = scale * scale * h;
                    g = Math.sqrt(h);
                    e[i] = scale * g;
                    f = CMath.length2(ar[i * n + l], ai[i * n + l]);
                    // form next diagonal element of matrix t
                    if (f !== 0) {
                        tau[l] = (ai[i * n + l] * tau[n + i] - ar[i * n + l] * tau[i]) / f;
                        si = (ar[i * n + l] * tau[n + i] + ai[i * n + l] * tau[i]) / f;
                        h += f * g;
                        g = 1.0 + g / f;
                        ar[i * n + l] *= g;
                        ai[i * n + l] *= g;
                        
                    } else {
                        tau[l] = -tau[i];
                        si = tau[n + i];
                        ar[i * n + l] = g;
                    }
                    if (l !== 0) {
                        f = 0.0;
                        for (j = 0;j <= l;j++) {
                            g = 0.0;
                            gi = 0.0;
                            // form element of a*u
                            for (k = 0;k <= j;k++) {
                                g = g + ar[j * n + k] * ar[i * n + k] + ai[j * n + k] * ai[i * n + k];
                                gi = gi - ar[j * n + k] * ai[i * n + k] + ai[j * n + k] * ar[i * n + k];
                            }
                            jp1 = j + 1;
                            if (l >= jp1) {
                                for (k = jp1;k <= l;k++) {
                                    g = g + ar[k * n + j] * ar[i * n + k] - ai[k * n + j] * ai[i * n + k];
                                    gi = gi - ar[k * n + j] * ai[i * n + k] - ai[k * n + j] * ar[i * n + k];
                                }
                            }
                            // form element of p
                            e[j] = g / h;
                            tau[n + j] = gi / h;
                            f += e[j] * ar[i * n + j] - tau[n + j] * ai[i * n + j];
                        }
                        hh = f / (h + h);
                        // form reduced a
                        for (j = 0;j <= l;j++) {
                            f = ar[i * n + j];
                            g = e[j] - hh * f;
                            e[j] = g;
                            fi = -ai[i * n + j];
                            gi = tau[n + j] - hh * fi;
                            tau[n + j] = -gi;
                            for (k = 0;k <= j;k++) {
                                ar[j * n + k] += -f * e[k] - g * ar[i * n + k]
                                    + fi * tau[n + k] + gi * ai[i * n + k];
                                ai[j * n + k] += -f * tau[n + k] - g * ai[i * n + k]
                                    - fi * e[k] - gi * ar[i * n + k];
                            }
                        }
                    }
                    for (k = 0;k <= l;k++) {
                        ar[i * n + k] *= scale;
                        ai[i * n + k] *= scale;
                    }
                    tau[n + l] = -si;
                }
            } else {
                e[i] = 0.0;
                e2[i] = 0.0;
            }
            hh = d[i];
            d[i] = ar[i * n + i];
            ar[i * n + i] = hh;
            ai[i * n + i] = scale * Math.sqrt(h);
        }
    }

    /**
     * Forms the eigenvectors via back transform.
     * See htribk.f in EISPACK for details.
     * @param n Dimension of the matrix.
     * @param ar (Input) Real part of the corresponding output from
     *           cTridiagonalize().
     * @param ai (Input) Imaginary part of the corresponding output from
     *           cTridiagonalize().
     * @param tau (Input) Further information about the transforms.
     * @param m Number of eigenvectors to be back transformed.
     * @param zr (Input/Output) Eigenvectors to be back transformed in its first
     *           m columns. Will be overwritten with the real part of the
     *           eigenvectors.
     * @param zi (Output) Imaginary part of the eigenvectors.
     */
    public static htribk(n: number, ar: DataBlock, ai: DataBlock, tau: DataBlock,
                         m: number, zr: DataBlock, zi: DataBlock): void {
        let i: number, j: number, k: number, l: number;
        let h: number, s: number, si: number;
        if (m > 0) {
            // recover and apply the Householder transforms
            for (k = 0;k < n;k++) {
                for (j = 0;j < m;j++) {
                    zi[k * n + j] = -zr[k * n + j] * tau[n + k];
                    zr[k * n + j] = zr[k * n + j] * tau[k];
                }
            }
            if (n === 1) {
                return;
            }
            for (i = 1;i < n;i++) {
                l = i - 1;
                h = ai[i * n + i];
                if (h === 0) {
                    continue;
                }
                for (j = 0;j < m;j++) {
                    s = 0.0;
                    si = 0.0;
                    for (k = 0;k <= l;k++) {
                        s += ar[i * n + k] * zr[k * n + j] - ai[i * n + k] * zi[k * n + j];
                        si += ar[i * n + k] * zi[k * n + j] + ai[i * n + k] * zr[k * n + j];
                    }
                    s = (s / h) / h;
                    si = (si / h) / h;
                    for (k = 0;k <= l;k++) {
                        zr[k * n + j] += -s * ar[i * n + k] - si * ai[i * n + k];
                        zi[k * n + j] += -si * ar[i * n + k] + s * ai[i * n + k];
                    }
                }
            }
        }
    }

    /**
     * 
     * @param n Dimension of the matrix.
     * @param ar (Input/Destroyed) Real part of the input matrix. Will be
     *           destroyed.
     * @param ai (Input/Destroyed) Imaginary part of the input matrix. Will be
     *           destroyed.
     * @param w (Output) Eigenvalues.
     * @param zr (Output) Real part of the eigenvectors.
     * @param zi (Output) Imaginary part of the eigenvectors.
     */
    public static eigHermitian(n: number, ar: DataBlock, ai: DataBlock,
                               w: DataBlock, zr: DataBlock, zi: DataBlock): void {
        let tmpArr1 = DataHelper.allocateFloat64Array(n);
        let tmpArr2 = DataHelper.allocateFloat64Array(n);
        let tmpArr3 = DataHelper.allocateFloat64Array(2 * n);
        Eigen.htridi(n, ar, ai, w, tmpArr1, tmpArr2, tmpArr3);
        for (let i = 0;i < n;i++) {
            zr[i * n + i] = 1.0;
        }
        Eigen.tql2(n, w, tmpArr1, zr);
        Eigen.htribk(n, ar, ai, tmpArr3, n, zr, zi);
    }

    /**
     * Balances a general real matrix.
     * See balanc.f in EISPACK.
     * @param n Dimension of the matrix.
     * @param a (Input/Output) Matrix data. Will be overwritten.
     * @param scale (Output) Scale data.
     */
    public static balanc(n: number, a: DataBlock, scale: DataBlock): [number, number] {
        let radix = 16.0;
        let b2 = radix * radix;
        let k = 0;
        let l = n - 1;
        let i: number, j: number;
        let m = 0, c: number, r: number, f: number, g: number, s: number;
        let flag = true;
        // search for rows isolating an eigenvalue and push them down
        while (flag) {
            flag = false;
            rowLoop: for (j = l;j >= 0;j--) {
                for (i = 0;i <= l;i++) {
                    if (i === j) {
                        continue;
                    }
                    if (a[j * n + i] !== 0.0) {
                        continue rowLoop;
                    }
                }
                // row/column exchange
                m = l;
                scale[m] = j;
                if (j !== m) {
                    for (i = 0;i <= l;i++) {
                        f = a[i * n + j];
                        a[i * n + j] = a[i * n + m];
                        a[i * n + m] = f;
                    }
                    for (i = k;i < n;i++) {
                        f = a[j * n + i];
                        a[j * n + i] = a[m * n + i];
                        a[m * n + i] = f;
                    }
                }
                if (l === 0) {
                    // return directly
                    return [k, l];
                }
                l--;
                flag = true;
                break;
            }
        }
        // search for columns isolating an eigenvalue and push them left
        flag = true;
        while (flag) {
            flag = false;
            colLoop: for (j = k;j <= l;j++) {
                for (i = k;i <= l;i++) {
                    if (i === j) {
                        continue;
                    }
                    if (a[i * n + j] !== 0.0) {
                        continue colLoop;
                    }
                }
                m = k;
                scale[m] = j;
                if (j !== m) {
                    for (i = 0;i <= l;i++) {
                        f = a[i * n + j];
                        a[i * n + j] = a[i * n + m];
                        a[i * n + m] = f;
                    }
                    for (i = k;i < n;i++) {
                        f = a[j * n + i];
                        a[j * n + i] = a[m * n + i];
                        a[m * n + i] = f;
                    }
                }
                k++;
                flag = true;
                break;
            }
        }
        // now balance the submatrix in rows k to l
        for (i = k;i <= l;i++) {
            scale[i] = 1.0;
        }
        // iterative loop for norm reduction
        do {
            flag = false;
            for (i = k;i <= l;i++) {
                c = 0.0;
                r = 0.0;
                for (j = k;j <= l;j++) {
                    if (j === i) {
                        continue;
                    }
                    c += Math.abs(a[j * n + i]);
                    r += Math.abs(a[i * n + j]);
                }
                // guard against zero c or r due to underflow
                if (c === 0.0 || r === 0.0) {
                    continue;
                }
                g = r / radix;
                f = 1.0;
                s = c + r;
                while (c < g) {
                    f *= radix;
                    c *= b2;
                }
                g = r * radix;
                while (c >= g) {
                    f /= radix;
                    c /= b2;
                }
                // now balance
                if ((c + r) / f >= 0.95 * s) {
                    continue;
                }
                g = 1.0 / f;
                scale[i] *= f;
                flag = true;
                for (j = k;j < n;j++) {
                    a[i * n + j] *= g;
                }
                for (j = 0;j <= l;j++) {
                    a[j * n + i] *= f;
                }
            }
        } while (flag);
        return [k, l];
    }

    /**
     * Converts to upper Hessenberg form.
     * See elmhes.f in EISPACK.
     * @param n Dimension of the matrix.
     * @param low Parameter determined by balanc().
     * @param igh Parameter determined by balanc().
     * @param a (Input/Output) Matrix data. Will be overwritten.
     * @param int (Output) Additional information for the transforms.
     */
    public static elmhes(n: number, low: number, igh: number, a: DataBlock, int: DataBlock): void {
        // to upper Hessenberg
        let i: number, j: number, m: number, la: number, kp1: number, mm1: number, mp1: number;
        let x: number, y: number;
        la = igh - 1;
        kp1 = low + 1;
        if (la < kp1) {
            return;
        }
        for (m = kp1;m <= la;m++) {
            mm1 = m - 1;
            x = 0.0;
            i = m;
            for (j = m;j <= igh;j++) {
                if (Math.abs(a[j * n + mm1]) > Math.abs(x)) {
                    x = a[j * n + mm1];
                    i = j;
                }
            }
            int[m] = i;
            if (i !== m) {
                // interchange rows and columns of a
                for (j = mm1;j < n;j++) {
                    y = a[i * n + j];
                    a[i * n + j] = a[m * n + j];
                    a[m * n + j] = y;
                }
                for (j = 0;j <= igh;j++) {
                    y = a[j * n + i];
                    a[j * n + i] = a[j * n + m];
                    a[j * n + m] = y;
                }
            }
            if (x === 0.0) {
                continue;
            }
            mp1 = m + 1;
            for (i = mp1;i <= igh;i++) {
                y = a[i * n + mm1];
                if (y === 0.0) {
                    continue;
                }
                y = y / x;
                a[i * n + mm1] = y;
                for (j = m;j < n;j++) {
                    a[i * n + j] -= y * a[m * n + j];
                }
                for (j = 0;j <= igh;j++) {
                    a[j * n + m] += y * a[j * n + i];
                }
            }
        }
    }

    /**
     * Accumulates the stabilized elementary similarity transformations used in
     * the reduction of a real general matrix to upper hessenberg form by
     * elmhes().
     * See eltran.f in EISPACK.
     * @param n Dimension of the matrix.
     * @param low Parameter determined by balanc().
     * @param igh Parameter determined by balanc().
     * @param a (Input) Output from elmhes().
     * @param int (Input) Output from elmhes().
     * @param z (Output) Contains the transformation matrix produced in
     *          the reduction by elmhes().
     */
    public static eltran(n: number, low: number, igh: number, a: ArrayLike<number>,
                         int: ArrayLike<number>, z: DataBlock): void {
        let i: number, j: number, kl: number, mp: number, mp1: number;
        // initialize z to identity matrix
        for (i = 0;i < n;i++) {
            for (j = 0;j < n;j++) {
                z[i * n + j] = 0.0;
            }
            z[i * n + i] = 1.0;
        }
        kl = igh - low + 1;
        if (kl < 1) {
            return;
        }
        for (mp = igh - 1;mp >= low + 1;mp--) {
            mp1 = mp + 1;
            for (i = mp1;i <= igh;i++) {
                z[i * n + mp] = a[i * n + mp - 1];
            }
            i = int[mp];
            if (i !== mp) {
                for (j = mp;j <= igh;j++) {
                    z[mp * n + j] = z[i * n + j];
                    z[i * n + j] = 0.0;
                }
                z[i * n + mp] = 1.0;
            }
        }
    }

    /**
     * Finds the eigenvalues and eigenvectors of a real upper Hessenberg matrix
     * by the QR iterations.
     * See hqr2.f in EISPACK.
     * @param n 
     * @param low 
     * @param igh 
     * @param h (Input/Destroyed)
     * @param wr (Output) Real part of the eigenvalues.
     * @param wi (Output) Imaginary part of the eigenvalues.
     * @param z (Output) Contains the real and imaginary parts of the
     *           eigenvectors. If the i-th eigenvalue is real, the i-th column
     *           of z contains its eigenvector.  if the i-th eigenvalue is
     *           complex with positive imaginary part, the i-th and (i+1)-th 
     *           columns of z contain the real and imaginary parts of its
     *           eigenvector. The eigenvectors are unnormalized.
     */
    public static hqr2(n: number, low: number, igh: number, h: DataBlock,
                       wr: DataBlock, wi: DataBlock, z: DataBlock): void {
        let i: number, j: number, k: number, l: number, m: number, en: number;
        let na: number = 0, itn: number, its: number = 0;
        let mp2: number, enm2: number = 0;
        let p: number = 0, q: number = 0, r: number = 0, s: number = 0, t: number, w: number;
        let x: number, y: number, ra: number, sa: number, vi: number, vr: number;
        let zz: number = 0, norm: number, tst1: number, tst2: number;
        let notlas: boolean, skip: boolean = false;
        norm = 0.0;
        k = 0;
        // store roots isolated by balanc and compute matrix norm
        for (i = 0;i < n;i++) {
            for (j = k;j <= n;j++) {
                norm += Math.abs(h[i * n + j]);
            }
            k = i;
            if (i < low || i >  igh) {
                wr[i] = h[i * n + i];
                wi[i] = 0.0;
            }
        }

        en = igh;
        t = 0.0;
        itn = 30 * n;
        while (true) {
            // search for next eigenvalues
            if (!skip) {
                if (en < low) {
                    break;
                }
                its = 0;
                na = en - 1;
                enm2 = na - 1;
            }
            // look for single small sub-diagonal element
            for (l = en;l > low;l--) {
                s = Math.abs(h[(l - 1) * n + l - 1]) + Math.abs(h[l * n + l]);
                if (s === 0.0) {
                    s = norm;
                }
                tst1 = s;
                tst2 = tst1 + Math.abs(h[l * n + l - 1]);
                if (tst2 === tst1) {
                    break;
                }
            }
            // form shift
            x = h[en * n + en];
            if (l !== en) {
                y = h[na * n + na];
                w = h[en * n + na] * h[na * n + en];
                if (l !== na) {
                    if (itn === 0) {
                        throw new Error('Maximum number of iterations reached.');
                    }
                    if (its === 10 || its === 20) {
                        // form exceptional shift
                        t += x;
                        for (i = low;i <= en;i++) {
                            h[i * n + i] -= x;
                        }
                        s = Math.abs(h[en * n + na]) + Math.abs(h[na * n + enm2]);
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its++;
                    itn--;
                    // look for two consecutive small sub-diagonal elements
                    for (m = en - 2;m >= l;m--) {
                        zz = h[m * n + m];
                        r = x - zz;
                        s = y - zz;
                        p = (r * s - w) / h[(m + 1) * n + m] + h[m * n + m + 1];
                        q = h[(m + 1) * n + (m + 1)] - zz - r - s;
                        r = h[(m + 2) * n + (m + 1)];
                        s = Math.abs(p) + Math.abs(q) + Math.abs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m === l) {
                            break;
                        }
                        tst1 = Math.abs(p) * (Math.abs(h[(m - 1) * n + (m - 1)]) + Math.abs(zz)
                            + Math.abs(h[(m + 1) * n + (m + 1)]));
                        tst2 = tst1 + Math.abs(h[m * n + (m - 1)]) * (Math.abs(q) + Math.abs(r));
                        if (tst2 === tst1) {
                            break;
                        }
                    }
                    mp2 = m + 2;
                    for (i = mp2;i <= en;i++) {
                        h[i * n + (i - 2)] = 0.0;
                        if (i === mp2) {
                            continue;
                        }
                        h[i * n + (i - 3)] = 0.0;
                    }
                    // double qr step involving rows 1 to en and columns m to en
                    for (k = m;k <= na;k++) {
                        notlas = k !== na;
                        if (k !== m) {
                            p = h[k * n + (k - 1)];
                            q = h[(k + 1) * n + (k - 1)];
                            r = notlas ? h[(k + 2) * n + (k - 1)] : 0.0;
                            x = Math.abs(p) + Math.abs(q) + Math.abs(r);
                            if (x === 0) {
                                continue;
                            }
                            p /= x;
                            q /= x;
                            r /= x;
                        }
                        s = p >= 0 ? Math.sqrt(p * p + q * q + r * r) : -Math.sqrt(p * p + q * q + r * r);
                        if (k !== m) {
                            h[k * n + (k - 1)] = -s * x;
                        } else {
                            if (l !== m) {
                                h[k * n + (k - 1)] = -h[k * n + (k - 1)];
                            }
                        }
                        p += s;
                        x = p / s;
                        y = q / s;
                        zz = r / s;
                        q = q / p;
                        r = r / p;
                        if (notlas) {
                            // row modification
                            for (j = k;j < n;j++) {
                                p = h[k * n + j] + q * h[(k + 1) * n + j] + r * h[(k + 2) * n + j];
                                h[k * n + j] -= p * x;
                                h[(k + 1) * n + j] -= p * y;
                                h[(k + 2) * n + j] -= p * zz;
                            }
                            j = Math.min(en, k + 3);
                            // column modification
                            for (i = 0;i <= j;i++) {
                                p = x * h[i * n + k] + y * h[i * n + (k + 1)] + zz * h[i * n + (k + 2)];
                                h[i * n + k] -= p;
                                h[i * n + (k + 1)] -= p * q;
                                h[i * n + (k + 2)] -= p * r;
                            }
                            // accumulate transforms
                            for (i = low;i <= igh;i++) {
                                p = x * z[i * n + k] + y * z[i * n + (k + 1)] + zz * z[i * n + (k + 2)];
                                z[i * n + k] -= p;
                                z[i * n + (k + 1)] -= p * q;
                                z[i * n + (k + 2)] -= p * r;
                            }
                        } else {
                            // row modification
                            for (j = k;j < n;j++) {
                                p = h[k * n + j] + q * h[(k + 1) * n + j];
                                h[k * n + j] -= p * x;
                                h[(k + 1) * n + j] -= p * y;
                            }
                            j = Math.min(en, k + 3);
                            // column modification
                            for (i = 0;i <= j;i++) {
                                p = x * h[i * n + k] + y * h[i * n + (k + 1)];
                                h[i * n + k] -= p;
                                h[i * n + (k + 1)] -= p * q;
                            }
                            // accumulate transforms
                            for (i = low;i <= igh;i++) {
                                p = x * z[i * n + k] + y * z[i * n + (k + 1)];
                                z[i * n + k] -= p;
                                z[i * n + (k + 1)] -= p * q;
                            }
                        }
                    }
                    skip = true;
                } else {
                    // two roots found
                    p = (y - x) / 2.0;
                    q = p * p + w;
                    zz = Math.sqrt(Math.abs(q));
                    h[en * n + en] = x + t;
                    x = h[en * n + en];
                    h[na * n + na] = y + t;
                    if (q >= 0.0) {
                        // real pair
                        zz = p + (p >= 0 ? Math.abs(zz) : -Math.abs(zz));
                        wr[na] = x + zz;
                        wr[en] = wr[na];
                        if (zz !== 0) {
                            wr[en] = x - w / zz;
                        }
                        wi[na] = 0.0;
                        wi[en] = 0.0;
                        x = h[en * n + na];
                        s = Math.abs(x) + Math.abs(zz);
                        p = x / s;
                        q = zz / s;
                        r = Math.sqrt(p * p + q * q);
                        p /= r;
                        q /= r;
                        // row modification
                        for (j = na;j < n;j++) {
                            zz = h[na * n + j];
                            h[na * n + j] = q * zz + p * h[en * n + j];
                            h[en * n + j] = q * h[en * n + j] - p * zz;
                        }
                        // column modification
                        for (i = 0;i <= en;i++) {
                            zz = h[i * n + na];
                            h[i * n + na] = q * zz + p * h[i * n + en];
                            h[i * n + en] = q * h[i * n + en] - p * zz;
                        }
                        // accumulate transforms
                        for (i = low;i <= igh;i++) {
                            zz = z[i * n + na];
                            z[i * n + na] = q * zz + p * z[i * n + en];
                            z[i * n + en] = q * z[i * n + en] - p * zz;
                        }
                        en = enm2;
                        skip = false;
                    } else {
                        wr[na] = x + p;
                        wr[en] = x + p;
                        wi[na] = zz;
                        wi[en] = -zz;
                        en = enm2;
                        skip = false;
                    }
                }
            } else {
                // one root found
                h[en * n + en] = x + t;
                wr[en] = h[en * n + en];
                wi[en] = 0.0;
                en = na;
                skip = false;
            }
        }
        // all roots found, backsubstitute to find vectors of upper triangular
        // form
        if (norm === 0) {
            return;
        }
        for (en = n - 1;en >= 0;en--) {
            p = wr[en];
            q = wi[en];
            na = en - 1;
            if (q < 0.0) {
                // complex vector
                m = na;
                // last vector component chosen imaginary so that eigenvector
                // matrix is triangular
                if (Math.abs(h[en * n + na]) <= Math.abs(h[na * n + en])) {
                    [h[na * n + na], h[na * n + en]] = CMath.cdivCC(
                        0.0, -h[na * n + en], h[na * n + na] - p, q);
                } else {
                    h[na * n + na] = q / h[en * n + na];
                    h[na * n + en] = -(h[en * n + en] - p) / h[en * n + na];
                }
                h[en * n + na] = 0.0;
                h[en * n + en] = 1.0;
                enm2 = na - 1;
                if (enm2 === -1) {
                    continue;
                }
                for (i = en - 2;i >= 0;i--) {
                    w = h[i * n + i] - p;
                    ra = 0.0;
                    sa = 0.0;
                    for (j = m;j <= en;j++) {
                        ra += h[i * n + j] * h[j * n + na];
                        sa += h[i * n + j] * h[j * n + en];
                    }
                    if (wi[i] >= 0.0) {
                        m = i;
                        if (wi[i] !== 0.0) {
                            // solve complex equations
                            x = h[i * n + (i + 1)];
                            y = h[(i + 1) * n + i];
                            vr = (wr[i] - p) * (wr[i] - p) + wi[i] * wi[i] - q * q;
                            vi = (wr[i] - p) * 2 * q;
                            if (vr === 0.0 && vi === 0.0) {
                                tst1 = norm * (Math.abs(w) + Math.abs(q) + Math.abs(x)
                                    + Math.abs(y) + Math.abs(zz));
                                vr = tst1;
                                do {
                                    vr = 0.01 * vr;
                                    tst2 = tst1 + vr;
                                } while (tst2 > tst1);
                            }
                            [h[i * n + na], h[i * n + en]] = CMath.cdivCC(
                                x * r - zz * ra + q * sa,
                                x * s - zz * sa - q * ra,
                                vr, vi);
                            if (Math.abs(x) <= Math.abs(zz) + Math.abs(q)) {
                                [h[(i + 1) * n + na], h[(i + 1) * n + en]] = CMath.cdivCC(
                                    -r - y * h[i * n + na], -s - y * h[i * n + en], zz, q);
                            } else {
                                h[(i + 1) * n + na] = (-ra - w * h[i * n + na] + q * h[i * n + en]) / x;
                                h[(i + 1) * n + en] = (-sa - w * h[i * n + en] - q * h[i * n + na]) / x;
                            }
                        } else {
                            [h[i * n + na], h[i * n + en]] = CMath.cdivCC(-ra, -sa, w, q);
                        }
                        // overflow control
                        t = Math.max(Math.abs(h[i * n + na]), Math.abs(h[i * n + en]));
                        if (t === 0.0) {
                            continue;
                        }
                        tst1 = t;
                        tst2 = tst1 + 1.0 / tst1;
                        if (tst2 <= tst1) {
                            for (j = i;j <= en;j++) {
                                h[j * n + na] /= t;
                                h[j * n + en] /= t;
                            }
                        }
                    } else {
                        zz = w;
                        r = ra;
                        s = sa;
                    }
                }
            } else if (q === 0.0) {
                // real vector
                m = en;
                h[en * n + en] = 1.0;
                if (na < 0) {
                    continue;
                }
                for (i = en - 1;i >= 0;i--) {
                    w = h[i * n + i] - p;
                    r = 0.0;
                    for (j = m;j <= en;j++) {
                        r += h[i * n + j] * h[j * n + en];
                    }
                    if (wi[i] < 0.0) {
                        zz = w;
                        s = r;
                        continue;
                    }
                    m = i;
                    if (wi[i] === 0.0) {
                        t = w;
                        if (t === 0.0) {
                            tst1 = norm;
                            t = tst1;
                            do {
                                t = 0.01 * t;
                                tst2 = norm + t;
                            } while (tst2 > tst1);
                        }
                        h[i * n + en] = -r / t;
                    } else {
                        // wi[i] !== 0
                        // solve real equations
                        x = h[i * n + (i + 1)];
                        y = h[(i + 1) * n + i];
                        q = (wr[i] - p) * (wr[i] -  p) + wi[i] * wi[i];
                        t = (x * s - zz * r) / q;
                        h[i * n + en] = t;
                        if (Math.abs(x) <= Math.abs(zz)) {
                            h[(i + 1) * n + en] = (-s - y * t) / zz;
                        } else {
                            h[(i + 1) * n + en] = (-r - w * t) / x;
                        }
                    }
                    // overflow control
                    t = Math.abs(h[i * n + en]);
                    if (t === 0.0) {
                        continue;
                    }
                    tst1 = t;
                    tst2 = tst1 + 1.0 / tst1;
                    if (tst2 <= tst1) {
                        for (j = i;j <= en;j++) {
                            h[j * n + en] /= t;
                        }
                    }
                }
            } else {
                continue;
            }
        }
        // vectors of isolated roots
        for (i = 0;i < n;i++) {
            if (i >= low && i <= igh) {
                continue;
            }
            for (j = i;j < n;j++) {
                z[i * n + j] = h[i * n + j];
            }
        }
        // multiply by transformation matrix to give vectors of original full
        // matrix
        for (j = n - 1;j >= 0;j--) {
            m = Math.min(j, igh);
            for (i = low;i <= igh;i++) {
                zz = 0.0;
                for (k = low;k <= m;k++) {
                    zz += z[i * n + k] * h[k * n + j];
                }
                z[i * n + j] = zz;
            }
        }
    }

    /**
     * Forms the eigenvectors of a real general matrix by back transforming
     * those of the corresponding balanced matrix determined by balanc().
     * See balbak.f in EISPACK.
     * @param n 
     * @param low 
     * @param igh 
     * @param scale (Input)
     * @param m 
     * @param z (Input/Output)
     */
    public static balbak(n: number, low: number, igh: number, scale: ArrayLike<number>, m: number, z: DataBlock): void {
        let i: number, j: number, k: number;
        let s: number;
        if (m === 0) {
            return;
        }
        if (igh !== low) {
            for (i = low;i <= igh;i++) {
                s = scale[i];
                // left hand eigenvectors are back transformed if the foregoing
                // statement is replaced by s = 1.0/scale[i]
                for (j = 0;j < m;j++) {
                    z[i * n + j] *= s;
                }
            }
        }
        for (i = low - 1;i >= 0;i--) {
            k = scale[i];
            if (k !== i) {
                for (j = 0;j < m;j++) {
                    s = z[i * n + j];
                    z[i * n + j] = z[k * n + j];
                    z[j * n + j] = s;
                }
            }
        }
        for (i = igh + 1;i < n;i++) {
            k = scale[i];
            if (k !== i) {
                for (j = 0;j < m;j++) {
                    s = z[i * n + j];
                    z[i * n + j] = z[k * n + j];
                    z[j * n + j] = s;
                }
            }
        }
    }

    /**
     * Performs eigendecomposition of general real matrices.  The
     * eigenvectors are unnormalized.
     * @param n 
     * @param a (Input/Destroyed)
     * @param wr (Output)
     * @param wi (Output)
     * @param zr (Output)
     * @param zi (Output)
     */
    public static eigRealGeneral(n: number, a: DataBlock, wr: DataBlock,
                                 wi: DataBlock, zr: DataBlock, zi: DataBlock): void {
        let i: number, j: number;
        let tmpArr1 = DataHelper.allocateFloat64Array(n);
        let [low, igh] = Eigen.balanc(n, a, tmpArr1);
        let tmpArr2 = DataHelper.allocateFloat64Array(igh + 1);
        Eigen.elmhes(n, low, igh, a, tmpArr2);
        Eigen.eltran(n, low, igh, a, tmpArr2, zr);
        Eigen.hqr2(n, low, igh, a, wr, wi, zr);
        Eigen.balbak(n, low, igh, tmpArr1, n, zr);
        // fill zi
        for (i = 0;i < n;) {
            if (wi[i] === 0) {
                // real eigenvalue, set column to zero
                for (j = 0;j < n;j++) {
                    zi[j * n + i] = 0.0;
                }
                i++;
            } else if (wi[i] > 0) {
                // complex eigenvalue, set imaginary part and update real part
                for (j = 0;j < n;j++) {
                    zi[j * n + i] = zr[j * n + (i + 1)];
                    zi[j * n + (i + 1)] = -zr[j * n + (i + 1)];
                    zr[j * n + (i + 1)] = zr[j * n + i];
                }
                i += 2;
            } else {
                throw new Error('This should never happen.');
            }
        }
    }

    public static cbal(n: number, ar: DataBlock, ai: DataBlock, scale: DataBlock): [number, number] {
        let i: number, j: number, k: number, l: number, m: number;
        let c: number, f: number, g: number, r: number, s: number, b2: number;
        let flag = true;
        let radix = 16.0;
        b2 = radix * radix;
        k = 0;
        l = n - 1;
        // search for rows isolating an eigenvalue and push them down
        while (flag) {
            flag = false;
            rowLoop: for (j = l;j >= 0;j--) {
                for (i = 0;i <= l;i++) {
                    if (i === j) {
                        continue;
                    }
                    if (ar[j * n + i] !== 0.0 || ai[j * n + i] !== 0.0) {
                        continue rowLoop;
                    }
                }
                m = l;
                scale[m] = j;
                if (j !== m) {
                    for (i = 0;i <= l;i++) {
                        f = ar[i * n + j];
                        ar[i * n + j] = ar[i * n + m];
                        ar[i * n + m] = f;
                        f = ai[i * n + j];
                        ai[i * n + j] = ai[i * n + m];
                        ai[i * n + m] = f;
                    }
                    for (i = k;i < n;i++) {
                        f = ar[j * n + i];
                        ar[j * n + i] = ar[m * n + i];
                        ar[m * n + i] = f;
                        f = ai[j * n + i];
                        ai[j * n + i] = ai[m * n + i];
                        ai[m * n + i] = f;
                    }
                }
                if (l === 0) {
                    return [k, l];
                }
                l--;
                flag = true;
                break;
            }
        }
        // search for columns isolating an eigenvalue and push them left
        flag = true;
        while (flag) {
            flag = false;
            colLoop: for (j = k;j <= l;j++) {
                for (i = k;i <= l;i++) {
                    if (i === j) {
                        continue;
                    }
                    if (ar[i * n + j] !== 0 || ai[i * n + j] !== 0.0) {
                        continue colLoop;
                    }
                }
                m = k;
                scale[m] = j;
                if (j !== m) {
                    for (i = 0;i <= l;i++) {
                        f = ar[i * n + j];
                        ar[i * n + j] = ar[i * n + m];
                        ar[i * n + m] = f;
                        f = ai[i * n + j];
                        ai[i * n + j] = ai[i * n + m];
                        ai[i * n + m] = f;
                    }
                    for (i = k;i < n;i++) {
                        f = ar[j * n + i];
                        ar[j * n + i] = ar[m * n + i];
                        ar[m * n + i] = f;
                        f = ai[j * n + i];
                        ai[j * n + i] = ai[m * n + i];
                        ai[m * n + i] = f;
                    }
                }
                k++;
                flag = true;
                break;
            }
        }
        // now balance the submatrix in rows k to l
        for (i = k;i <= l;i++) {
            scale[i] = 1.0;
        }
        do {
            flag = false; // noconv = false
            for (i = k;i <= l;i++) {
                c = 0.0;
                r = 0.0;
                for (j = k;j <= l;j++) {
                    if (j !== i) {
                        c += Math.abs(ar[j * n + i]) + Math.abs(ai[j * n + i]);
                        r += Math.abs(ar[i * n + j]) + Math.abs(ai[i * n + j]);
                    }
                }
                // guard against zero c or r due to underflow
                if (c === 0.0 || r === 0.0) {
                    continue;
                }
                g = r / radix;
                f = 1.0;
                s = c + r;
                while (c < g) {
                    f *= radix;
                    c *= b2;
                }
                g = r * radix;
                while (c >= g) {
                    f /= radix;
                    c /= b2;
                }
                // now balance
                if ((c + r) / f >= 0.95 * s) {
                    continue;
                }
                g = 1.0 / f;
                scale[i] *= f;
                flag = true;
                
                for (j = k;j < n;j++) {
                    ar[i * n + j] *= g;
                    ai[i * n + j] *= g;
                }
                for (j = 0;j <= l;j++) {
                    ar[j * n + i] *= f;
                    ai[j * n + i] *= f;
                }
            }
        } while (flag);
        return [k, l];
    }

    public static corth(n: number, low: number, igh: number, ar: DataBlock,
                        ai: DataBlock, ortr: DataBlock, orti: DataBlock): void {
        let i: number, j: number, m: number, la: number, mp: number, kp1: number;
        let f: number, g: number, h: number, fi: number, fr: number, scale: number;
        la = igh - 1;
        kp1 = low + 1;
        if (la < kp1) {
            return;
        }
        for (m = kp1;m <= la;m++) {
            h = 0.0;
            ortr[m] = 0.0;
            orti[m] = 0.0;
            scale = 0.0;
            // scale column (algol tol then not needed)
            for (i = m;i <= igh;i++) {
                scale += Math.abs(ar[i * n + (m - 1)]) + Math.abs(ai[i * n + (m - 1)]);
            }
            if (scale === 0.0) {
                continue;
            }
            mp = m + igh;
            for (i = igh;i >= m;i--) {
                ortr[i] = ar[i * n + (m - 1)] / scale;
                orti[i] = ai[i * n + (m - 1)] / scale;
                h += ortr[i] * ortr[i] + orti[i] * orti[i];
            }

            g = Math.sqrt(h);
            f = CMath.length2(ortr[m], orti[m]);
            if (f !== 0.0) {
                h += f * g;
                g /= f;
                ortr[m] *= 1.0 + g;
                orti[m] *= 1.0 + g; 
            } else {
                ortr[m] = g;
                ar[m * n + (m - 1)] = scale;
            }
            // form (i - (u * ut)/h) * a
            for (j = m;j < n;j++) {
                fr = 0.0;
                fi = 0.0;
                for (i = igh;i >= m;i--) {
                    fr += ortr[i] * ar[i * n + j] + orti[i] * ai[i * n + j];
                    fi += ortr[i] * ai[i * n + j] - orti[i] * ar[i * n + j];
                }
                fr /= h;
                fi /= h;
                for (i = m;i <= igh;i++) {
                    ar[i * n + j] += -fr * ortr[i] + fi * orti[i];
                    ai[i * n + j] += -fr * orti[i] - fi * ortr[i];
                }
            }
            // form (i - (u * ut)/h) * a * (i - (u * ut)/h)
            for (i = 0;i <= igh;i++) {
                fr = 0.0;
                fi = 0.0;
                for (j = igh;j >= m;j--) {
                    fr += ortr[j] * ar[i * n + j] - orti[j] * ai[i * n + j];
                    fi += ortr[j] * ai[i * n + j] + orti[j] * ar[i * n + j];
                }
                fr /= h;
                fi /= h;
                for (j = m;j <= igh;j++) {
                    ar[i * n + j] += -fr * ortr[j] - fi * orti[j];
                    ai[i * n + j] += fr * orti[j] - fi * ortr[j];
                }
            }
            ortr[m] *= scale;
            orti[m] *= scale;
            ar[m * n + (m - 1)] *= -g;
            ai[m * n + (m - 1)] *= -g;
        }
    }

    public static comqr2(n: number, low: number, igh: number, ortr: DataBlock,
                         orti: DataBlock, hr: DataBlock, hi: DataBlock,
                         wr: DataBlock, wi: DataBlock, zr: DataBlock, zi: DataBlock): void {
        let i: number, j: number, k: number, l: number, ll: number, m: number, en: number;
        let ip1: number, itn: number, its: number = 0, lp1: number, enm1: number = 0, iend: number;
        let si: number, sr: number, ti: number, tr: number, xi: number, xr: number;
        let yi: number, yr: number, zzi: number, zzr: number, norm: number;
        let tst1: number, tst2: number;
        let skip: boolean = false;
        // initialize eigenvector matrix
        for (i = 0;i < n;i++) {
            for (j = 0;j < n;j++) {
                zr[i * n + j] = 0.0;
                zi[i * n + j] = 0.0;
            }
            zr[i * n + i] = 1.0;
        }
        // form the matrix of accumulated transforms from the information left
        // by corth
        iend = igh - low - 1;
        //hr[5] = 1.5; hi[5] = -1.5;
        if (iend > 0) {
            // iend > 0
            for (i = igh - 1;i >= low + 1;i--) {
                if (ortr[i] === 0.0 && orti[i] === 0.0) {
                    continue;
                }
                if (hr[i * n + (i - 1)] === 0.0 && hi[i * n + (i - 1)] === 0.0) {
                    continue;
                }
                // norm below is negative of h formed in corth
                norm = hr[i * n + (i - 1)] * ortr[i] + hi[i * n + (i - 1)] * orti[i];
                ip1 = i + 1;

                for (k = ip1;k <= igh;k++) {
                    ortr[k] = hr[k * n + (i - 1)];
                    orti[k] = hi[k * n + (i - 1)];
                }

                for (j = i;j <= igh;j++) {
                    sr = 0.0;
                    si = 0.0;
                    for (k = i;k <= igh;k++) {
                        sr += ortr[k] * zr[k * n + j] + orti[k] * zi[k * n + j];
                        si += ortr[k] * zi[k * n + j] - orti[k] * zr[k * n + j];
                    }
                    sr /= norm;
                    si /= norm;
                    for (k = i;k <= igh;k++) {
                        zr[k * n + j] += sr * ortr[k] - si * orti[k];
                        zi[k * n + j] += sr * orti[k] + si * ortr[k];
                    }
                }
            }
        }
        // create real subdiagonal elements
        if (iend >= 0) {
            l = low + 1;
            for (i = l;i <= igh;i++) {
                ll = Math.min(i + 1, igh);
                if (hi[i * n + (i - 1)] === 0.0) {
                    continue;
                }
                norm = CMath.length2(hr[i * n + (i - 1)], hi[i * n + (i - 1)]);
                yr = hr[i * n + (i - 1)] / norm;
                yi = hi[i * n + (i - 1)] / norm;
                hr[i * n + (i - 1)] = norm;
                hi[i * n + (i - 1)] = 0.0;

                for (j = i;j < n;j++) {
                    si = yr * hi[i * n + j] - yi * hr[i * n + j];
                    hr[i * n + j] = yr * hr[i * n + j] + yi * hi[i * n + j];
                    hi[i * n + j] = si;
                }
                for (j = 0;j <= ll;j++) {
                    si = yr * hi[j * n + i] + yi * hr[j * n + i];
                    hr[j * n + i] = yr * hr[j * n + i] - yi * hi[j * n + i];
                    hi[j * n + i] = si;
                }
                for (j = low;j <= igh;j++) {
                    si = yr * zi[j * n + i] + yi * zr[j * n + i];
                    zr[j * n + i] = yr * zr[j * n + i] - yi * zi[j * n + i];
                    zi[j * n + i] = si;
                }
            }
        }
        // store roots isolated by cbal
        for (i = 0;i < n;i++) {
            if (i >= low && i <= igh) {
                continue;
            }
            wr[i] = hr[i * n + i];
            wi[i] = hi[i * n + i];
        }

        en = igh;
        tr = 0.0;
        ti = 0.0;
        itn = 30 * n;
        // search for next eigenvalue
        while (true) {
            if (!skip) {
                if (en < low) {
                    break;
                }
                its = 0;
                enm1 = en - 1;
            }
            // look for single small sub-diagonal element
            for (l = en;l > low;l--) {
                tst1 = Math.abs(hr[(l - 1) * n + (l - 1)]) + Math.abs(hi[(l - 1) * n + (l - 1)])
                    + Math.abs(hr[l * n + l]) + Math.abs(hi[l * n + l]);
                tst2 = tst1 + Math.abs(hr[l * n + (l - 1)]);
                if (tst2 === tst1) {
                    break;
                }
            }
            // form shift
            if (l === en) {
                // a root found
                hr[en * n + en] += tr;
                wr[en] = hr[en * n + en];
                hi[en * n + en] += ti;
                wi[en] = hi[en * n + en];
                en = enm1;
                skip = false;
            } else {
                if (itn === 0) {
                    throw new Error('Maximum allowed iterations reached.');
                }
                if (its === 10 || its === 20) {
                    // form exceptional shift
                    sr = Math.abs(hr[en * n + enm1]) + Math.abs(hr[enm1 * n + (en - 2)]);
                    si = 0.0;
                } else {
                    sr = hr[en * n + en];
                    si = hi[en * n + en];
                    xr = hr[enm1 * n + en] * hr[en * n + enm1];
                    xi = hi[enm1 * n + en] * hr[en * n + enm1];
                    if (xr !== 0.0 || xi !== 0.0) {
                        yr = (hr[enm1 * n + enm1] - sr) / 2.0;
                        yi = (hi[enm1 * n + enm1] - si) / 2.0;
                        [zzr, zzi] = CMath.csqrt((yr + yi) * (yr - yi) + xr, 2.0 * yr * yi + xi);
                        if (yr * zzr + yi * zzi < 0.0) {
                            zzr = -zzr;
                            zzi = -zzi;
                        }
                        [xr, xi] = CMath.cdivCC(xr, xi, yr + zzr, yi + zzi);
                        sr -= xr;
                        si -= xi;
                    }
                }

                for (i = low;i <= en;i++) {
                    hr[i * n + i] -= sr;
                    hi[i * n + i] -= si;
                }

                tr += sr;
                ti += si;
                its++;
                itn--;
                // reduce to triangle (rows)
                lp1 = l + 1;
                for (i = lp1;i <= en;i++) {
                    sr = hr[i * n + (i - 1)];
                    hr[i * n + (i - 1)] = 0.0;
                    norm = CMath.length2(
                        CMath.length2(hr[(i - 1) * n + (i - 1)], hi[(i - 1) * n + (i - 1)]), sr);
                    xr = hr[(i - 1) * n + (i - 1)] / norm;
                    wr[i - 1] = xr;
                    xi = hi[(i - 1) * n + (i - 1)] / norm;
                    wi[i - 1] = xi;
                    hr[(i - 1) * n + (i - 1)] = norm;
                    hi[(i - 1) * n + (i - 1)] = 0.0;
                    hi[i * n + (i - 1)] = sr / norm;

                    for (j = i;j < n;j++) {
                        yr = hr[(i - 1) * n + j];
                        yi = hi[(i - 1) * n + j];
                        zzr = hr[i * n + j];
                        zzi = hi[i * n + j];
                        hr[(i - 1) * n + j] = xr * yr + xi * yi + hi[i * n + (i - 1)] * zzr;
                        hi[(i - 1) * n + j] = xr * yi - xi * yr + hi[i * n + (i - 1)] * zzi;
                        hr[i * n + j] = xr * zzr - xi * zzi - hi[i * n + (i - 1)] * yr;
                        hi[i * n + j] = xr * zzi + xi * zzr - hi[i * n + (i - 1)] * yi;
                    }
                }

                si = hi[en * n + en];
                if (si !== 0.0) {
                    norm = CMath.length2(hr[en * n + en], si);
                    sr = hr[en * n + en] / norm;
                    si /= norm;
                    hr[en * n + en] = norm;
                    hi[en * n + en] = 0.0;
                    if (en < n - 1) {
                        ip1 = en + 1;
                        for (j = ip1;j < n;j++) {
                            yr = hr[en * n + j];
                            yi = hi[en * n + j];
                            hr[en * n + j] = sr * yr + si * yi;
                            hi[en * n + j] = sr * yi - si * yr;
                        }
                    }
                }
                // inverse operation (columns)
                for (j = lp1;j <= en;j++) {
                    xr = wr[j - 1];
                    xi = wi[j - 1];
                    for (i = 0;i <= j;i++) {
                        yr = hr[i * n + (j - 1)];
                        yi = 0.0;
                        zzr = hr[i * n + j];
                        zzi = hi[i * n + j];
                        if (i !== j) {
                            yi = hi[i * n + (j - 1)];
                            hi[i * n + (j - 1)] = xr * yi + xi * yr + hi[j * n + (j - 1)] * zzi;
                        }
                        hr[i * n + (j - 1)] = xr * yr - xi * yi + hi[j * n + (j - 1)] * zzr;
                        hr[i * n + j] = xr * zzr + xi * zzi - hi[j * n + (j - 1)] * yr;
                        hi[i * n + j] = xr * zzi - xi * zzr - hi[j * n + (j - 1)] * yi;
                    }
                    for (i = low;i <= igh;i++) {
                        yr = zr[i * n + (j - 1)];
                        yi = zi[i * n + (j - 1)];
                        zzr = zr[i * n + j];
                        zzi = zi[i * n + j];
                        zr[i * n + (j - 1)] = xr * yr - xi * yi + hi[j * n + (j - 1)] * zzr;
                        zi[i * n + (j - 1)] = xr * yi + xi * yr + hi[j * n + (j - 1)] * zzi;
                        zr[i * n + j] = xr * zzr + xi * zzi - hi[j * n + (j - 1)] * yr;
                        zi[i * n + j] = xr * zzi - xi * zzr - hi[j * n + (j - 1)] * yi;
                    }
                }
                if (si === 0.0) {
                    skip = true;
                    continue;
                }
                for (i = 0;i <= en;i++) {
                    yr = hr[i * n + en];
                    yi = hi[i * n + en];
                    hr[i * n + en] = sr * yr - si * yi;
                    hi[i * n + en] = sr * yi + si * yr;
                }
                for (i = low;i <= igh;i++) {
                    yr = zr[i * n + en];
                    yi = zi[i * n + en];
                    zr[i * n + en] = sr * yr - si * yi;
                    zi[i * n + en] = sr * yi + si * yr;
                }
                skip = true;
            }
        }
        // all roots found. backsubstitute to find vectors of upper triangular
        // form
        norm = 0.0;
        for (i = 0;i < n;i++) {
            for (j = i;j < n;j++) {
                tr = Math.abs(hr[i * n + j]) + Math.abs(hi[i * n + j]);
                if (tr > norm) {
                    norm = tr;
                }
            }
        }
        if (n === 1 || norm === 0.0) {
            return;
        }
        for (en = n - 1;en >= 1;en--) {
            xr = wr[en];
            xi = wi[en];
            hr[en * n + en] = 1.0;
            hi[en * n + en] = 0.0;
            enm1 = en - 1;
            for (i = en - 1;i >= 0;i--) {
                zzr = 0.0;
                zzi = 0.0;
                ip1 = i + 1;
                for (j = ip1;j <= en;j++) {
                    zzr += hr[i * n + j] * hr[j * n + en] - hi[i * n + j] * hi[j * n + en];
                    zzi += hr[i * n + j] * hi[j * n + en] + hi[i * n + j] * hr[j * n + en];
                }
                yr = xr - wr[i];
                yi = xi - wi[i];
                if (yr === 0.0 && yi === 0.0) {
                    tst1 = norm;
                    yr = tst1;
                    do {
                        yr *= 0.01;
                        tst2 = norm + yr;
                    } while (tst2 > tst1);
                }
                [hr[i * n + en], hi[i * n + en]] = CMath.cdivCC(zzr, zzi, yr, yi);
                // overflow control
                tr = Math.abs(hr[i * n + en]) + Math.abs(hi[i * n + en]);
                if (tr === 0.0) {
                    continue;
                }
                tst1 = tr;
                tst2 = tst1 + 1.0 / tst1;
                if (tst2 <= tst1) {
                    for (j = i;j <= en;j++) {
                        hr[j * n + en] /= tr;
                        hi[j * n + en] /= tr;
                    }
                }
            }
        }
        // end backsubstitution
        // vectors of isolated roots
        for (i = 0;i < n;i++) {
            if (i >= low && i <= igh) {
                continue;
            }
            for (j = i;j < n;j++) {
                zr[i * n + j] = hr[i * n + j];
                zi[i * n + j] = hi[i * n + j];
            }
        }
        // multiply by transformation matrix to give vectors of original full
        // matrix
        for (j = n - 1;j >= low;j--) {
            m = Math.min(j, igh);
            for (i = low;i <= igh;i++) {
                zzr = 0.0;
                zzi = 0.0;
                for (k = low;k <= m;k++) {
                    zzr += zr[i * n + k] * hr[k * n + j] - zi[i * n + k] * hi[k * n + j];
                    zzi += zr[i * n + k] * hi[k * n + j] + zi[i * n + k] * hr[k * n + j];
                }
                zr[i * n + j] = zzr;
                zi[i * n + j] = zzi;
            }
        }
    }

    public static cbabk2(n: number, low: number, igh: number, scale: ArrayLike<number>,
                         m: number, zr: DataBlock, zi: DataBlock): void {
        let i: number, j: number, k: number;
        let s: number;
        if (m === 0) {
            return;
        }
        if (igh !== low) {
            for (i = low;i <= igh;i++) {
                s = scale[i];
                // left hand eigenvectors are back transformed if the foregoing statement
                // is replaced by s = 1.0/scale[i]
                for (j = 0;j < m;j++) {
                    zr[i * n + j] *= s;
                    zi[i * n + j] *= s;
                }
            }
        }
        for (i = low - 1;i >= 0;i--) {
            k = scale[i];
            if (k !== i) {    
                for (j = 0;j < m;j++) {
                    s = zr[i * n + j];
                    zr[i * n + j] = zr[k * n + j];
                    zr[k * n + j] = s;
                    s = zi[i * n + j];
                    zi[i * n + j] = zi[k * n + j];
                    zi[k * n + j] = s;
                }
            }
        }
        for (i = igh + 1;i < n;i++) {
            k = scale[i];
            if (k !== i) {    
                for (j = 0;j < m;j++) {
                    s = zr[i * n + j];
                    zr[i * n + j] = zr[k * n + j];
                    zr[k * n + j] = s;
                    s = zi[i * n + j];
                    zi[i * n + j] = zi[k * n + j];
                    zi[k * n + j] = s;
                }
            }
        }
    }

    /**
     * Performs eigendecomposition for general complex matrices. The
     * eigenvectors are unnormalized.
     * @param n 
     * @param ar (Input/Destroyed)
     * @param ai (Input/Destroyed)
     * @param wr (Output)
     * @param wi (Output)
     * @param zr (Output)
     * @param zi (Output)
     */
    public static eigComplexGeneral(n: number, ar: DataBlock, ai: DataBlock,
                                    wr: DataBlock, wi: DataBlock, zr: DataBlock,
                                    zi: DataBlock): void {
        let scale = DataHelper.allocateFloat64Array(n);
        let ortr = DataHelper.allocateFloat64Array(n);
        let orti = DataHelper.allocateFloat64Array(n);
        for (let i = 0;i < n;i++) {
            scale[i] = 0.0;
            ortr[i] = 0.0;
            orti[i] = 0.0;
        }
        let [low, igh] = Eigen.cbal(n, ar, ai, scale);
        Eigen.corth(n, low, igh, ar, ai, ortr, orti);
        Eigen.comqr2(n, low, igh, ortr, orti, ar, ai, wr, wi, zr, zi);
        Eigen.cbabk2(n, low, igh, scale, n, zr, zi);
    }

}