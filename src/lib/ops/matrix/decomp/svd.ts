import { DataBlock } from '../../../storage';
import { CMathHelper } from '../../../helper/mathHelper';

/**
 * Singular value decomposition.
 */
export class SVD {

    /**
     * Singular value decomposition for a real matrix A such that A = USV^T.
     * Adapted from Numerical Recipes.
     * @param m Number of rows in A.
     * @param n Number of columns in A.
     * @param computeUV If set to false, only singular values will be computed.
     *                  In this case, A will still be overwritten, but V will be
     *                  untouched.
     * @param reA (Input/Output) m x n Matrix A. Will be overwritten as U.
     * @param reS (Output) A n-element vector of singular values sorted in
     *            descending order.
     * @param reV (Output) n x n matrix V.
     */
    public static svd(m: number, n: number, computeUV: boolean, reA: DataBlock, reS: DataBlock, reV: DataBlock) {
        let flag: number, i: number, j: number, jj: number, k: number, l: number = 0;
        let its: number, nm: number = 0;
        let f: number, g: number = 0, h: number, c: number;
        let x: number, y: number, z: number;
        let rv1 = new Array(n);
        // 1. Use Householder reflections to reduce A into bidiagonal form.
        let scale = 0, anorm = 0;
        let s: number = 0; // s acts a an accumulator in various places
        for (i = 0;i < n;i++) {
            l = i + 1;
            // Left side Householder reflections.
            // Update the i-th super diagonal element to ||A[i,i+1:end]||.
            // Note that rv1[0] is always 0
            rv1[i] = scale * g;
            g = 0;
            s = 0;
            scale = 0;
            if (i < m) {
                // Compute scaling factor for the i-th column to be reduced.
                // Note: scaling a vector does not affect the resulting Householder
                // transform matrix.
                for (k = i;k < m;k++) {
                    scale += Math.abs(reA[k * n + i]);
                }
                if (scale) {
                    // Scale is non-zero. If scale is zero, we do not need to do
                    // anything.
                    for (k = i;k < m;k++) {
                        reA[k * n + i] /= scale; // do scaling
                        s += reA[k * n + i] * reA[k * n + i];
                    }
                    // now s = ||A[i,i:end]||_2^2 / scale^2
                    // f <- A[i,i]
                    f = reA[i * n + i];
                    // g <- - sgn(f) * sqrt(s)
                    g = f >= 0 ? -Math.sqrt(s) : Math.sqrt(s);
                    // \beta = 2/(w^T w) = 1/(s + sgn(f) * f * sqrt(s))
                    // h is actually -1/\beta
                    h = f * g - s;
                    // Update A[i,i] according to Householder refection:
                    //  w <- a \pm ||a||e_1
                    // The Householder vector is stored exactly in the i-th column.
                    reA[i * n + i] = f - g;
                    // Apply Householder reflection to the remaining elements of A
                    //  (I - \beta ww^T) A = A - (\beta w)(w^T [a_1, a_2, ..., a_n])
                    for (j = l;j < n;j++) {
                        s = 0.0
                        // w^T a_j
                        for (k = i;k < m;k++) {
                            s += reA[k * n + i] * reA[k * n + j];
                        }
                        // -\beta w^T a_j
                        f = s / h;
                        //  a_j + (-\beta w^T a_j) w
                        for (k = i;k < m;k++) {
                            reA[k * n + j] += f * reA[k * n + i];
                        }
                    }
                    // scale back
                    for (k = i;k < m;k++) {
                        reA[k * n + i] *= scale;
                    }
                }
            }
            // Update the i-th diagonal element to ||A[i:end,i]||.
            reS[i] = scale * g;
            // Right side Householder reflections.
            g = 0;
            s = 0;
            scale = 0;
            // note that we do not need to work on the last column
            if (i < m && i !== n - 1) {
                for (k = l;k < n;k++) {
                    scale += Math.abs(reA[i * n + k]);
                }
                if (scale) {
                    for (k = l;k < n;k++) {
                        reA[i * n + k] /= scale;
                        s += reA[i * n + k] * reA[i * n + k];
                    }
                    // same as above
                    f = reA[i * n + l];
                    g = f >= 0 ? -Math.sqrt(s) : Math.sqrt(s);
                    h = f * g - s;
                    reA[i * n + l] = f - g;
                    // A[i,i+1:end] *= -\beta (i.e., rv1 <- -\beta w)
                    for (k = l;k < n;k++) {
                        rv1[k] = reA[i * n + k] / h;
                    }
                    // Apply Householder reflection to the remaining elements
                    //  A (I - \beta ww^T) = A - (Av) (\beta w^T)
                    for (j = l;j < m;j++) {
                        s = 0;
                        // s <- - A[j,i+1:end] * w
                        for (k = l;k < n;k++) {
                            s += reA[j * n + k] * reA[i * n + k];
                        }
                        // A[j,i+1:end] += (A[j,i+1:end] * w) (- \beta w^T)
                        for (k = l;k < n;k++) {
                            reA[j * n + k] += s * rv1[k];
                        }
                    }
                    // scale back
                    for (k = l;k < n;k++) {
                        reA[i * n + k] *= scale;
                    }
                }
            }
            anorm = Math.max(anorm, Math.abs(reS[i]) + Math.abs(rv1[i]));
        }
        if (computeUV) {
            // 2. Accumulate right side Householder transforms H_1^R H_2^R ...
            // Q_i Q = (I - \beta w_i w_i^T) Q = Q - (\beta w_i) (w_i^T Q)
            for (i = n - 1;i >= 0;i--) {
                if (i < n - 1) {
                    if (g) {
                        // V[i+1:end,i] <- w_i / w_i(1) / g
                        // Note that w_i(1) = f - g. Then w_i(1) * g = f * g - s = h
                        // which is -1/\beta. Then V[i+1:end,i] <- -\beta * w_i
                        for (j = l;j < n;j++) {
                            reV[j * n + i] = (reA[i * n + j] / reA[i * n + l]) / g;
                        }
                        for (j = l;j < n;j++) {
                            s = 0;
                            // w_i^T Q[i+1:end,j]
                            for (k = l;k < n;k++) {
                                s += reA[i * n + k] * reV[k * n + j];
                            }
                            // += - (\beta w_i) (w_i^T Q)
                            for (k = l;k < n;k++) {
                                reV[k * n + j] += s * reV[k * n + i];
                            }
                        }
                    }
                    // V[i+1:end,i] <- 0, V[i,i+1:end] <- 0
                    for (j = l;j < n;j++) {
                        reV[i * n + j] = 0;
                        reV[j * n + i] = 0;
                    }
                }
                // V[i,i] = 1
                reV[i * n + i] = 1.0;
                g = rv1[i];
                l = i;
            }
            // 3. Accumulate left side Householder transforms ... H_2^L H_1^L and
            //    store it in A.
            for (i = Math.min(m, n) - 1;i >= 0;i--) {
                l = i + 1;
                g = reS[i];
                // U[i,i+1:end] <- 0
                for (j = l;j < n;j++) {
                    reA[i * n + j] = 0;
                }
                if (g) {
                    // update U[i:end,i+1:end]
                    g = 1.0 / g;
                    for (j = l;j < n;j++) {
                        s = 0;
                        for (k = l;k < m;k++) {
                            s += reA[k * n + i] * reA[k * n + j];
                        }
                        f = (s / reA[i * n + i]) * g;
                        for (k = i;k < m;k++) {
                            reA[k * n + j] += f * reA[k * n + i];
                        }
                    }
                    // update U[i:end,i]
                    for (j = i;j < m;j++) {
                        reA[j * n + i] *= g;
                    }
                } else {
                    // U[i:end,i] <- 0
                    for (j = i;j < m;j++) {
                        reA[j * n + i] = 0;
                    }
                }
                reA[i * n + i]++;
            }
        }
        // 4. Diagonalize the bidiagonal matrix.
        for (k = n - 1;k >= 0;k--) {
            for (its = 0;its < 30;its++) {
                flag = 1;
                // test for splitting
                for (l = k;l >= 0;l--) {
                    nm = l - 1;
                    // Test if rv[l] is sufficiently small.
                    // Note: rv1[0] is always zero. So when l = 0, the if statement
                    // after this one will not be reached, preventing illegal access
                    // of reS[-1].
                    if (Math.abs(rv1[l]) + anorm === anorm) {
                        flag = 0;
                        break;
                    }
                    // Test if reS[nm] is sufficiently small.
                    if (Math.abs(reS[nm]) + anorm === anorm) {
                        break;
                    }
                }
                if (flag) {
                    // found a zero super diagonal element
                    c = 0;
                    s = 1;
                    for (i = l;i <= k;i++) {
                        f = s * rv1[i];
                        rv1[i] = c * rv1[i];
                        if (Math.abs(f) + anorm === anorm) {
                            break;
                        }
                        g = reS[i];
                        h = CMathHelper.length2(f, g);
                        reS[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = -f * h;
                        if (computeUV) {
                            for (j = 0;j < m;j++) {
                                y = reA[j * n + nm];
                                z = reA[j * n + i];
                                reA[j * n + nm] = y * c + z * s;
                                reA[j * n + i] = z * c - y * s;
                            }
                        }
                    }
                }
                z = reS[k];
                if (l === k) {
                    // convergence
                    if (z < 0) {
                        // make sure the singular value is nonnegative
                        reS[k] = -z;
                        if (computeUV) {
                            for (j = 0;j < n;j++) {
                                reV[j * n + k] = -reV[j * n + k];
                            }
                        }
                    }
                    break;
                }
                if (its === 29) {
                    throw new Error('Failed to converge.');
                }
                // QR iterations
                x = reS[l];
                nm = k - 1;
                y = reS[nm];
                g = rv1[nm];
                h = rv1[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = CMathHelper.length2(f, 1.0);
                f = ((x - z) * (x + z) + h * (
                        (y / (f + f >= 0 ? g : -g)) - h)) / x;
                c = 1.0;
                s = 1.0;
                for (j = l;j <= nm;j++) {
                    i = j + 1;
                    g = rv1[i];
                    y = reS[i];
                    h = s * g;
                    g = c * g;
                    z = CMathHelper.length2(f, h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;
                    if (computeUV) {
                        for (jj = 0;jj < n;jj++) {
                            x = reV[jj * n + j];
                            z = reV[jj * n + i];
                            reV[jj * n + j] = x * c + z * s;
                            reV[jj * n + i] = z * c - x * s;
                        }
                    }
                    z = CMathHelper.length2(f, h);
                    reS[j] = z;
                    if (z) {
                        z = 1.0 / z;
                        c = f * z;
                        s = h * z;
                    }
                    f = c * g + s * y;
                    x = c * y - s * g;
                    if (computeUV) {
                        for (jj = 0;jj < m;jj++) {
                            y = reA[jj * n + j];
                            z = reA[jj * n + i];
                            reA[jj * n + j] = y * c + z * s;
                            reA[jj * n + i] = z * c - y * s;
                        }
                    }
                }
                rv1[l] = 0.0;
                rv1[k] = f;
                reS[k] = x;
            }
        }
        // 5. Sort singular values in descending order.
        // We use simple selection sort here.
        for (i = 0;i < n - 1;i++) {
            x = reS[i];
            jj = i;
            // find the next maximum
            for (j = i + 1;j < n;j++) {
                if (reS[j] > x) {
                    x = reS[j];
                    jj = j;
                }
            }
            if (jj !== i) {
                // swap
                y = reS[i];
                reS[i] = reS[jj];
                reS[jj] = y;
                if (computeUV) {
                    for (k = 0;k < m;k++) {
                        y = reA[k * n + i];
                        reA[k * n + i] = reA[k * n + jj];
                        reA[k * n + jj] = y;
                    }
                    for (k = 0;k < n;k++) {
                        y = reV[k * n + i];
                        reV[k * n + i] = reV[k * n + jj];
                        reV[k * n + jj] = y;
                    }
                }
            }
        }
    }

    /**
     * Singular value decomposition for a complex matrix A such that A = USV^H.
     * Based on the above routine and Algorithm 358.
     * @param m Number of rows in A.
     * @param n Number of columns in A.
     * @param computeUV If set to false, only singular values will be computed.
     *                  In this case, A will still be overwritten, but V will be
     *                  untouched.
     * @param reA (Input/Output) Real part of the m x n Matrix A. Will be
     *            overwritten with the real part of U.
     * @param imA (Input/Output) Imaginary part of the m x n Matrix A. Will be
     *            overwritten with the imaginary part of U.
     * @param reS (Output) A n-element vector of singular values sorted in
     *            descending order.
     * @param reV (Output) Real part of the n x n matrix V. 
     * @param imV (Output) Imaginary part of the n x n matrix V. 
     */
    public static csvd(m: number, n: number, computeUV: boolean, reA: DataBlock,
                    imA: DataBlock, reS: DataBlock, reV: DataBlock, imV: DataBlock) {
        let flag: number, i: number, j: number, jj: number, k: number, l: number = 0;
        let its: number, nm: number = 0;
        let f: number, g: number = 0, h: number, c: number;
        let accRe: number, accIm: number;
        let x: number, y: number, z: number;
        let rv1 = new Array(n);
        let phaseRe = new Array(n);
        let phaseIm = new Array(n);
        let phase2Re = new Array(n);
        let phase2Im = new Array(n);
        for (i = 0;i < n;i++) {
            phaseRe[i] = 1;
            phaseIm[i] = 0;
            phase2Re[i] = 1;
            phase2Im[i] = 0;
        }
        phase2Re[0] = -1;
        phase2Im[0] = 0;
        // 1. Use Householder reflections to reduce A into bidiagonal form.
        // There is one difference from the real case: phase transform is required
        // to ensure the bidiagonal matrix is real. We have
        //  B = P_M L_M ... P_1 L_1 A R_1 T_1 R_2 T_2 ... R_N T_N
        // where L_i, R_i are Householder projection matrices, and P_i, R_i are
        // phase transform matrices.
        let scale = 0, anorm = 0;
        let s: number = 0; // s acts a an accumulator in various places
        for (i = 0;i < n;i++) {
            l = i + 1;
            // Left side Householder reflections.
            // Update the i-th super diagonal element to ||A[i,i+1:end]||.
            // Since we applied the phase transform, the elements in the bidiagonal
            // matrix will always be nonnegative.
            // Note that rv1[0] is always 0
            rv1[i] = scale * g;
            g = 0;
            s = 0;
            scale = 0;
            if (i < m) {
                // Compute scaling factor for the i-th column to be reduced.
                // Note: scaling a vector does not affect the resulting Householder
                // transform matrix.
                for (k = i;k < m;k++) {
                    scale += CMathHelper.length2(reA[k * n + i], imA[k * n + i]);
                }
                if (scale) {
                    // Scale is non-zero. If scale is zero, we do not need to do
                    // anything.
                    for (k = i;k < m;k++) {
                        // do scaling
                        reA[k * n + i] /= scale;
                        imA[k * n + i] /= scale;
                        s += reA[k * n + i] * reA[k * n + i] + imA[k * n + i] * imA[k * n + i];
                    }
                    // now s = ||A[i,i:end]||_2^2 / scale^2
                    // f <- |A[i,i]|
                    f = CMathHelper.length2(reA[i * n + i], imA[i * n + i]);
                    // g <- ||a|| (Here a = A[i:end,i])
                    g = Math.sqrt(s);
                    // \beta = 2/(w^H w) = 1/(s + |A[i,i]| * sqrt(s))
                    // h is actually -1/\beta
                    h = - f * g - s;
                    // Update A[i,i] according to Householder refection:
                    //  w <- a + exp(j\theta) ||a|| e_1, then
                    //  H a <- - exp(j\theta) ||a|| e_1
                    // The Householder vector is stored exactly in the i-th column.
                    if (f) {
                        phaseRe[i] = reA[i * n + i] / f;
                        phaseIm[i] = imA[i * n + i] / f;
                        reA[i * n + i] += phaseRe[i] * g;
                        imA[i * n + i] += phaseIm[i] * g;
                    } else {
                        // A[i,i] is zero, we set phase to zero
                        phaseRe[i] = 1;
                        phaseIm[i] = 0;
                        reA[i * n + i] = g;
                        imA[i * n + i] = 0;
                    }
                    // Apply Householder reflection to the remaining elements of A
                    //  (I - \beta ww^H) A = A - (\beta w)(w^H [a_1, a_2, ..., a_n])
                    for (j = l;j < n;j++) {
                        accRe = 0.0;
                        accIm = 0.0;
                        // w^H a_j
                        for (k = i;k < m;k++) {
                            accRe += reA[k * n + i] * reA[k * n + j] + imA[k * n + i] * imA[k * n + j];
                            accIm += reA[k * n + i] * imA[k * n + j] - imA[k * n + i] * reA[k * n + j];
                        }
                        // -\beta w^H a_j
                        accRe /= h;
                        accIm /= h;
                        //  a_j + (-\beta w^H a_j) w
                        for (k = i;k < m;k++) {
                            reA[k * n + j] += accRe * reA[k * n + i] - accIm * imA[k * n + i];
                            imA[k * n + j] += accRe * imA[k * n + i] + accIm * reA[k * n + i];
                        }
                    }
                    // scale back
                    for (k = i;k < m;k++) {
                        reA[k * n + i] *= scale;
                        imA[k * n + i] *= scale;
                    }
                    // apply phase transform
                    // Note that H a <- - exp(j\theta) ||a|| e_1
                    // The transform phase should be -exp(-j\theta)
                    for (k = l;k < n;k++) {
                        x = reA[i * n + k];
                        y = imA[i * n + k];
                        // note the conjugate here
                        reA[i * n + k] = - x * phaseRe[i] - y * phaseIm[i];
                        imA[i * n + k] = x * phaseIm[i] - y * phaseRe[i];
                    }
                } else {
                    // no need to transform here
                    phaseRe[i] = 1;
                    phaseIm[i] = 0;
                }
            }
            // Update the i-th diagonal element to ||A[i:end,i]||.
            // Since we applied the phase transform, the elements in the bidiagonal
            // matrix will always be nonnegative.
            reS[i] = scale * g;
            // Right side Householder reflections.
            g = 0;
            s = 0;
            scale = 0;
            // note that we do not need to work on the last column
            if (i < m && i !== n - 1) {
                for (k = l;k < n;k++) {
                    scale += CMathHelper.length2(reA[i * n + k], imA[i * n + k]);
                }
                if (scale) {
                    for (k = l;k < n;k++) {
                        reA[i * n + k] /= scale;
                        imA[i * n + k] /= scale;
                        s += reA[i * n + k] * reA[i * n + k] + imA[i * n + k] * imA[i * n + k];
                    }
                    // Almost same as above, except the projection vector is
                    // constructed from
                    //   w <- a^H + exp(-j\theta) ||a^H|| e_1, then
                    // where \theta is the angle of the original a(1)
                    // Then a (I - \beta w w^H) = -exp(j\theta) ||a|| e_1^T
                    // apply conjugate
                    for (k = l;k < n;k++) {
                        imA[i * n + k] = -imA[i * n + k];
                    }
                    f = CMathHelper.length2(reA[i * n + l], imA[i * n + l]);
                    g = Math.sqrt(s);
                    h = - f * g - s;
                    if (f) {
                        // stores the phase of the original a(1)
                        phase2Re[l] = reA[i * n + l] / f;
                        phase2Im[l] = - imA[i * n + l] / f;
                        reA[i * n + l] += phase2Re[l] * g;
                        imA[i * n + l] += - phase2Im[l] * g;
                    } else {
                        // A[i,i] is zero, we set phase to zero
                        phase2Re[l] = 1;
                        phase2Im[l] = 0;
                        reA[i * n + l] = g;
                        imA[i * n + l] = 0;
                    }
                    // Apply Householder reflection to the remaining elements
                    //  A (I - \beta ww^H) = A - (Aw) (\beta w^H)
                    for (j = l;j < m;j++) {
                        accRe = 0;
                        accIm = 0;
                        // acc <- - A[j,i+1:end] * w
                        for (k = l;k < n;k++) {
                            accRe += reA[j * n + k] * reA[i * n + k] - imA[j * n + k] * imA[i * n + k];
                            accIm += reA[j * n + k] * imA[i * n + k] + imA[j * n + k] * reA[i * n + k];
                        }
                        accRe /= h;
                        accIm /= h;
                        // A[j,k+1:end] += (A[j,k+1:end] * w) (- \beta w^H)
                        for (k = l;k < n;k++) {
                            reA[j * n + k] += accRe * reA[i * n + k] + accIm * imA[i * n + k];
                            imA[j * n + k] += - accRe * imA[i * n + k] + accIm * reA[i * n + k];
                        }
                    }
                    // scale back
                    for (k = l;k < n;k++) {
                        reA[i * n + k] *= scale;
                        imA[i * n + k] *= scale;
                    }
                    // apply phase transform
                    // Note that H a <- - exp(j\theta) ||a|| e_1^T
                    // The transform phase should be -exp(-j\theta)
                    for (k = l;k < m;k++) {
                        x = reA[k * n + l];
                        y = imA[k * n + l];
                        // note the conjugate here
                        reA[k * n + l] = - x * phase2Re[l] - y * phase2Im[l];
                        imA[k * n + l] = x * phase2Im[l] - y * phase2Re[l];
                    }
                } else {
                    phase2Re[l] = 1.0;
                    phase2Im[l] = 0.0;
                }
            }
            anorm = Math.max(anorm, Math.abs(reS[i]) + Math.abs(rv1[i]));
        }
        // Now reS and rv1 stores the elements bidiagonal matrix after removing
        // the phases.
        if (computeUV) {
            // 2. Accumulate right side transforms:
            //      ... A = B RP_N^H R_N ... R_1 RP_1^H
            //    Since the Householder projections R_i are Hermitian
            //    Initial V = RP_1 R_1 ... R_N RP_N
            //    We recursive accumulate the transforms starting from the right bottom
            //    corner using the following trick
            //      (I - \beta w_i w_i^H) Q = Q - (\beta w_i) (w_i^H Q)
            for (i = n - 1;i >= 0;i--) {
                if (i < n - 1) {
                    if (g) {
                        // Note that A[i,i+1] now stores w_i(1), which is
                        //  exp(j\theta_i) (r + sqrt(s))
                        // where r is the original |A[i,i+1]|.
                        // Because \beta = 1/(s + r * sqrt(s))
                        // -\beta can be recovered from -1/|A[i,i+1]|/g
                        h = - 1.0 / CMathHelper.length2(reA[i * n + l], imA[i * n + l]) / g;
                        for (j = l;j < n;j++) {
                            accRe = 0;
                            accIm = 0;
                            // w_i^H Q[i+1:end,j]
                            for (k = l;k < n;k++) {
                                //           w_i              Q_j
                                accRe += reA[i * n + k] * reV[k * n + j] + imA[i * n + k] * imV[k * n + j];
                                accIm += reA[i * n + k] * imV[k * n + j] - imA[i * n + k] * reV[k * n + j];
                            }
                            accRe *= h;
                            accIm *= h;
                            // += - (\beta w_i) (w_i^H Q)
                            for (k = l;k < n;k++) {
                                //                            w_i
                                reV[k * n + j] += accRe * reA[i * n + k] - accIm * imA[i * n + k];
                                imV[k * n + j] += accRe * imA[i * n + k] + accIm * reA[i * n + k];
                            }
                        }
                    }
                    // V[i+1:end,i] <- 0, V[i,i+1:end] <- 0
                    for (j = l;j < n;j++) {
                        reV[i * n + j] = 0;
                        reV[j * n + i] = 0;
                        imV[i * n + j] = 0;
                        imV[j * n + i] = 0;
                    }
                }
                // V[i,i] <- phase
                // -exp(-j\theta)
                reV[i * n + i] = -phase2Re[i];
                imV[i * n + i] = phase2Im[i];
                g = rv1[i];
                l = i;
            }
            // 3. Accumulate left side transforms and store it in A:
            //      LP_M L_M ... LP_1 L1 A ... = B V_0^H
            //    Initial U = LP_M^H L_M ... LP_1^H L1
            //    We use the following trick for in-place update:
            //      [a' b'] = [1 - \beta w_1^* w_1,   -\beta w_1 w_2^H D]
            //      [c' D']   [  - \beta w_1^* w_2, D -\beta w_2 w_2^H D]
            //    where a' is the new A[i,i], b' is the new A[i,i+1:end]
            //    c' is the new A[i+1:end,i], d' is the new A[i+1:end, i+1:end]
            //    and w = [w_1; w_2] where w_1 is the first element (original A[i,i]),
            //    and w_2 consists of the remaining elements (original A[i+1:end,i]).
            for (i = Math.min(m, n) - 1;i >= 0;i--) {
                l = i + 1;
                g = reS[i];
                // U[i,i+1:end] <- 0
                // This is safe as they store the right side Householder vectors
                for (j = l;j < n;j++) {
                    reA[i * n + j] = 0;
                    imA[i * n + j] = 0;
                }
                if (g) {
                    // update U[i:end,i+1:end]
                    // h is now -\beta
                    h = - 1.0 / CMathHelper.length2(reA[i * n + i], imA[i * n + i]) / g;
                    for (j = l;j < n;j++) {
                        // acc <- w_2^H * D
                        accRe = 0;
                        accIm = 0;
                        for (k = l;k < m;k++) {
                            //           w_2^H               D
                            accRe += reA[k * n + i] * reA[k * n + j] + imA[k * n + i] * imA[k * n + j];
                            accIm += reA[k * n + i] * imA[k * n + j] - imA[k * n + i] * reA[k * n + j];
                        }
                        // -\beta w_2^H D
                        accRe *= h;
                        accIm *= h;
                        // Obtain b' and D'.
                        for (k = i;k < m;k++) {
                            //                              w
                            reA[k * n + j] += accRe * reA[k * n + i] - accIm * imA[k * n + i];
                            imA[k * n + j] += accRe * imA[k * n + i] + accIm * reA[k * n + i];
                        }
                    }
                    // Update a' and c' without adding one to a' yet.
                    // This can be expressed as -\beta w_1^* w
                    accRe = reA[i * n + i] * h;
                    accIm = imA[i * n + i] * h;
                    for (k = i;k < m;k++) {
                        x = reA[k * n + i];
                        y = imA[k * n + i];
                        reA[k * n + i] = accRe * x + accIm * y; 
                        imA[k * n + i] = accRe * y - accIm * x; 
                    }
                    // add one
                    reA[i * n + i]++;
                    // apply phase transform
                    // conj(-exp(-j\theta)) = -exp(j\theta)
                    for (k = i;k < m;k++) {
                        x = reA[k * n + i];
                        y = imA[k * n + i];
                        reA[k * n + i] = - phaseRe[i] * x + phaseIm[i] * y; 
                        imA[k * n + i] = - phaseRe[i] * y - phaseIm[i] * x; 
                    }
                } else {
                    // U[i:end,i] <- 0
                    for (j = i;j < m;j++) {
                        reA[j * n + i] = 0;
                        imA[j * n + i] = 0;
                    }
                    reA[i * n + i] = 1;
                }
            }
        }

        // 4. Diagonalize the bidiagonal matrix.
        for (k = n - 1;k >= 0;k--) {
            for (its = 0;its < 30;its++) {
                flag = 1;
                // test for splitting
                for (l = k;l >= 0;l--) {
                    nm = l - 1;
                    // Test if rv[l] is sufficiently small.
                    // Note: rv1[0] is always zero. So when l = 0, the if statement
                    // after this one will not be reached, preventing illegal access
                    // of reS[-1].
                    if (Math.abs(rv1[l]) + anorm === anorm) {
                        flag = 0;
                        break;
                    }
                    // Test if reS[nm] is sufficiently small.
                    if (Math.abs(reS[nm]) + anorm === anorm) {
                        break;
                    }
                }
                if (flag) {
                    // found a zero super diagonal element
                    c = 0;
                    s = 1;
                    for (i = l;i <= k;i++) {
                        f = s * rv1[i];
                        rv1[i] = c * rv1[i];
                        if (Math.abs(f) + anorm === anorm) {
                            break;
                        }
                        g = reS[i];
                        h = CMathHelper.length2(f, g);
                        reS[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = -f * h;
                        if (computeUV) {
                            for (j = 0;j < m;j++) {
                                y = reA[j * n + nm];
                                z = reA[j * n + i];
                                reA[j * n + nm] = y * c + z * s;
                                reA[j * n + i] = z * c - y * s;
                                y = imA[j * n + nm];
                                z = imA[j * n + i];
                                imA[j * n + nm] = y * c + z * s;
                                imA[j * n + i] = z * c - y * s;
                            }
                        }
                    }
                }
                z = reS[k];
                if (l === k) {
                    // convergence
                    if (z < 0) {
                        // make sure the singular value is nonnegative
                        reS[k] = -z;
                        if (computeUV) {
                            for (j = 0;j < n;j++) {
                                reV[j * n + k] = -reV[j * n + k];
                                imV[j * n + k] = -imV[j * n + k];                        
                            }
                        }
                    }
                    break;
                }
                if (its === 29) {
                    throw new Error('Failed to converge.');
                }
                // QR iterations
                x = reS[l];
                nm = k - 1;
                y = reS[nm];
                g = rv1[nm];
                h = rv1[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = CMathHelper.length2(f, 1.0);
                f = ((x - z) * (x + z) + h * (
                        (y / (f + f >= 0 ? g : -g)) - h)) / x;
                c = 1.0;
                s = 1.0;
                for (j = l;j <= nm;j++) {
                    i = j + 1;
                    g = rv1[i];
                    y = reS[i];
                    h = s * g;
                    g = c * g;
                    z = CMathHelper.length2(f, h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;
                    if (computeUV) {
                        for (jj = 0;jj < n;jj++) {
                            x = reV[jj * n + j];
                            z = reV[jj * n + i];
                            reV[jj * n + j] = x * c + z * s;
                            reV[jj * n + i] = z * c - x * s;
                            x = imV[jj * n + j];
                            z = imV[jj * n + i];
                            imV[jj * n + j] = x * c + z * s;
                            imV[jj * n + i] = z * c - x * s;
                        }
                    }
                    z = CMathHelper.length2(f, h);
                    reS[j] = z;
                    if (z) {
                        z = 1.0 / z;
                        c = f * z;
                        s = h * z;
                    }
                    f = c * g + s * y;
                    x = c * y - s * g;
                    if (computeUV) {
                        for (jj = 0;jj < m;jj++) {
                            y = reA[jj * n + j];
                            z = reA[jj * n + i];
                            reA[jj * n + j] = y * c + z * s;
                            reA[jj * n + i] = z * c - y * s;
                            y = imA[jj * n + j];
                            z = imA[jj * n + i];
                            imA[jj * n + j] = y * c + z * s;
                            imA[jj * n + i] = z * c - y * s;
                        }
                    }
                }
                rv1[l] = 0.0;
                rv1[k] = f;
                reS[k] = x;
            }
        }
        // 5. Sort singular values in descending order.
        // We use simple selection sort here.
        for (i = 0;i < n - 1;i++) {
            x = reS[i];
            jj = i;
            // find the next maximum
            for (j = i + 1;j < n;j++) {
                if (reS[j] > x) {
                    x = reS[j];
                    jj = j;
                }
            }
            if (jj !== i) {
                // swap
                y = reS[i];
                reS[i] = reS[jj];
                reS[jj] = y;
                if (computeUV) {
                    for (k = 0;k < m;k++) {
                        y = reA[k * n + i];
                        reA[k * n + i] = reA[k * n + jj];
                        reA[k * n + jj] = y;
                        y = imA[k * n + i];
                        imA[k * n + i] = imA[k * n + jj];
                        imA[k * n + jj] = y;
                    }
                    for (k = 0;k < n;k++) {
                        y = reV[k * n + i];
                        reV[k * n + i] = reV[k * n + jj];
                        reV[k * n + jj] = y;
                        y = imV[k * n + i];
                        imV[k * n + i] = imV[k * n + jj];
                        imV[k * n + jj] = y;
                    }
                }
            }
        }
    }

}