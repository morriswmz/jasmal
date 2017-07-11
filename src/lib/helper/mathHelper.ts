export class MathHelper {

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
     * Evaluates complex division x/y using Smith's formula.
     * @param reX Re(x)
     * @param imX Im(x)
     * @param reY Re(y)
     * @param imY Im(y)
     */
    public static complexDiv(reX: number, imX: number, reY: number, imY: number): [number, number] {
        let absReY = Math.abs(reY);
        let absImY = Math.abs(imY);
        let r: number, d: number;
        if (absImY < absReY) {
            r = imY / reY;
            d = reY + imY * r;
            return [(reX + imX * r) / d, (imX - reX * r) / d];
        } else {
            if (absImY === 0) {
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
            }
            r = reY / imY;
            d = imY + reY * r;
            return [(imX + reX * r) / d, (- reX + imX * r) / d];
        }
    }

    /**
     * Evaluates complex inversion 1/x using Smith's formula.
     * @param reX 
     * @param imX 
     */
    public static complexInv(reX: number, imX: number): [number, number] {
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

}