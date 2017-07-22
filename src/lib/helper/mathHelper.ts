export class CMathHelper {

    /**
     * Safely computes sqrt(x*x+y*y) with minimal overflow.
     * For large x and y, evaluating x*x or y*y directly may cause overflow. To
     * avoid it, we use the following formulas instead:
     *  x*sqrt(1 + (y/x)*(y/x)) if y < x,
     *  y*sqrt(1 + (x/y)*(x/y)) otherwise.
     * @param {number} x
     * @param {number} y
     */
    public static length2(x: number, y: number): number {
        let absX = Math.abs(x),
            absY = Math.abs(y);
        if (absX > absY) {
            let ratio = absY / absX;
            return absX * Math.sqrt(1.0 + ratio * ratio);
        } else {
            if (absY === 0) {
                return absX;
            } else {
                let ratio = absX / absY;
                return absY * Math.sqrt(1.0 + ratio * ratio);
            }
        }
    }

    /**
     * Evaluates complex division x/y without overflow. This is based on
     * the cdiv function in ESIPACK (http://www.netlib.no/netlib/eispack/3090vf/double/cdiv.f)
     * with modification to handle NaN and Infinity.
     * @param reX Re(x)
     * @param imX Im(x)
     * @param reY Re(y)
     * @param imY Im(y)
     */
    public static cdivCC(reX: number, imX: number, reY: number, imY: number): [number, number] {
        let s = Math.abs(reY) + Math.abs(imY);
        if (s === 0.0) {
            // divide by zero
            if (reX === 0) {
                if (imX === 0) {
                    return [NaN, NaN];
                } else {
                    return [0, imX / 0];
                }
            } else {
                return [reX / 0, imX === 0 ? 0 : imX / 0];
            }
        } else {
            let reXs = reX / s;
            let imXs = imX / s;
            let reYs = reY / s;
            let imYs = imY / s;
            s = reYs * reYs + imYs * imYs;
            return [(reXs * reYs + imXs * imYs) / s, (imXs * reYs - reXs * imYs) / s];
        }
    }

    /**
     * Evaluates the division between a real number and a complex number using
     * Smith's formula.
     * @param reX Re(x)
     * @param reY Re(y)
     * @param imY Im(y)
     */
    public static cdivRC(reX: number, reY: number, imY: number): [number, number] {
        let r: number, t: number;
        if (Math.abs(reY) > Math.abs(imY)) {
            r = imY / reY;
            t = reY + imY * r;
            return [-r / t * reX, reX / t];
        } else {
            if (imY === 0) {
                return [Infinity, 0];
            } else {
                r = reY / imY;
                t = reY * r + imY;
                return [-reX / t, r / t * reX];
            }
        }
    }

    /**
     * Evaluates complex inversion 1/x using Smith's formula.
     * @param reX Re(x)
     * @param imX Im(x)
     */
    public static cReciprocal(reX: number, imX: number): [number, number] {
        let absReX = Math.abs(reX);
        let absImX = Math.abs(imX);
        let r: number, d: number;
        if (absImX < absReX) {
            r = imX / reX;
            d = reX + imX * r;
            return [1 / d, - r / d];
        } else {
            if (absImX === 0) {
                return [Infinity, 0];
            }
            r = reX / imX;
            d = imX + reX * r;
            return [r / d, - 1 / d];
        }
    }

    /**
     * Computes complex square root. This is based on the csroot function in
     * ESIPACK (http://www.netlib.no/netlib/eispack/3090vf/double/cdiv.f) with
     * modifications to handle NaN and Infinity.
     * @param reX Real part.
     * @param imX Imaginary part.
     */
    public static csqrt(reX: number, imX: number): [number, number] {
        if (isNaN(reX) || isNaN(imX)) {
            return [NaN, NaN];
        }
        if (!isFinite(imX)) {
            return [Infinity, imX > 0 ? Infinity : -Infinity];
        } else if (!isFinite(reX)) {
            return reX > 0 ? [Infinity, 0] : [0, imX >= 0 ? Infinity : -Infinity];
        }
        let reY: number = 0, imY: number = 0;
        let absReX = Math.abs(reX),
            absImX = Math.abs(imX);
        let l: number = 0, ratio: number;
        if (absReX > absImX) {
            ratio = absImX / absReX;
            l = absReX * Math.sqrt(1.0 + ratio * ratio);
        } else {
            if (absImX !== 0) {
                ratio = absReX / absImX;
                l = absImX * Math.sqrt(1.0 + ratio * ratio);
            }
        }
        let s = Math.sqrt(0.5 * (l + Math.abs(reX)));
        if (reX >= 0) reY = s;
        if (imX < 0) s = -s;
        if (reX <= 0) imY = s;
        if (reX < 0) reY = 0.5 * (imX / imY);
        if (reX > 0) imY = 0.5 * (imX / reY);
        return [reY, imY];
    }

}