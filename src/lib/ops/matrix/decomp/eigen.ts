import { DataBlock } from "../../../storage";
import { MathHelper } from "../../../helper/mathHelper";

export class Eigen {

    /**
     * See tred2 in ESIPACK.
     * @param n 
     * @param reA 
     * @param diag 
     * @param sub 
     * @param reE 
     */
    public static tridiagonalize(n: number, reA: DataBlock, diag: DataBlock,
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
     * See tql2 in ESIPACK.
     * @param n 
     * @param diag 
     * @param sub 
     * @param reE 
     */
    public static triQL(n: number, diag: DataBlock, sub: DataBlock, reE: DataBlock): void {
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
                        r = MathHelper.length2(p, 1.0);
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
                            r = MathHelper.length2(p, sub[i]);
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

    public static eigSym(n: number, reA: DataBlock, lambda: DataBlock, reE: DataBlock): void {
        let tmpArr = new Array(n);
        Eigen.tridiagonalize(n, reA, lambda, tmpArr, reE);
        Eigen.triQL(n, lambda, tmpArr, reE);
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
    public static cTridiagonalize(n: number, ar: DataBlock, ai: DataBlock, d: DataBlock,
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
                    f = MathHelper.length2(ar[i * n + l], ai[i * n + l]);
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
    public static cTriBackTransform(n: number, ar: DataBlock, ai: DataBlock,
                                    tau: DataBlock, m: number, zr: DataBlock,
                                    zi: DataBlock): void {
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

    public static eigHermitian(n: number, ar: DataBlock, ai: DataBlock,
                               w: DataBlock, zr: DataBlock, zi: DataBlock): void {
        let tmpArr1 = new Array(n);
        let tmpArr2 = new Array(n);
        let tmpArr3 = new Array(2 * n);
        Eigen.cTridiagonalize(n, ar, ai, w, tmpArr1, tmpArr2, tmpArr3);
        for (let i = 0;i < n;i++) {
            zr[i * n + i] = 1.0;
        }
        Eigen.triQL(n, w, tmpArr1, zr);
        Eigen.cTriBackTransform(n, ar, ai, tmpArr3, n, zr, zi);
    }

}