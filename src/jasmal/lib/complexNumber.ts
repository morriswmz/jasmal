export class ComplexNumber {

    private _re: number;
    private _im: number;

    constructor();
    constructor(tuple: [number, number]);
    constructor(r: number, i: number);
    constructor(r: number | number[] = 0, i: number = 0) {
        if (Array.isArray(r)) {
            this._re = r[0];
            this._im = r[1];
        } else {
            this._re = r;
            this._im = i;
        }
    }

    public get re(): number {
        return this._re;
    }

    public get im(): number {
        return this._im;
    }

    public addc(y: ComplexNumber): ComplexNumber {
        return new ComplexNumber(this._re + y._re, this._im + y._im);
    }

    public addr(y: number): ComplexNumber {
        return new ComplexNumber(this._re + y, this._im);
    }

    public subc(y: ComplexNumber): ComplexNumber {
        return new ComplexNumber(this._re - y._re, this._im - y._im);
    }

    public subr(y: number): ComplexNumber {
        return new ComplexNumber(this._re - y, this._im);
    }

    public mulc(y: ComplexNumber): ComplexNumber {
        return new ComplexNumber(this._re * y._re - this._im * y._im, this._re * y._im + this._im * y._re);
    }

    public mulr(y: number): ComplexNumber {
        return new ComplexNumber(this._re * y, this._im * y);
    }

    public divc(y: ComplexNumber): ComplexNumber {
        if (y._im === 0) {
            return this.divr(y._re);
        }
        let [re, im] = CMath.cdivCC(this._re, this._im, y._re, y._im);
        return new ComplexNumber(re, im);
    }

    public divr(y: number): ComplexNumber {
        return new ComplexNumber(this._re / y, this._im / y);
    }

    public neg(): ComplexNumber {
        return new ComplexNumber(-this._re, -this._im);
    }

    public inv(): ComplexNumber {
        let [re, im] = CMath.cReciprocal(this._re, this._im);
        return new ComplexNumber(re, im);
    }

    public norm(): number {
        return ComplexNumber.norm2(this._re, this._im);
    }

    public angle(): number {
        return ComplexNumber.angle2(this._re, this._im);
    }

    public conjugate(): ComplexNumber {
        return new ComplexNumber(this._re, -this._im);
    }

    public equals(y: ComplexNumber): boolean {
        return this._re === y._re && this._im === y._im;
    }

    public toString(): string {
        return this._re + (this._im >= 0 ? '+' : '-') + this._im + 'j';
    }

    public static norm2(re: number, im: number) {
        return CMath.length2(re, im);
    }

    public static angle2(re: number, im: number) {
        return Math.atan2(im, re);
    }

}

export class CMath {

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
     * the cdiv function in EISPACK (http://www.netlib.no/netlib/eispack/3090vf/double/cdiv.f)
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
     * EISPACK (http://www.netlib.no/netlib/eispack/3090vf/double/cdiv.f) with
     * modifications to handle NaN and Infinity.
     * sqrt(z) => [sqrt(0.5*(Re(z) + |z|)), 0.5*Im(z) / sqrt(0.5*(Re(z) + |z|))]
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

    /**
     * Complex sine.
     * sin(z) = sin(Re(z)) cosh(Im(z)) + j cos(Re(z)) sinh(Im(z))
     * @param re 
     * @param im 
     */
    public static csin(re: number, im: number): [number, number] {
        let s = Math.sin(re), c = Math.cos(re);
        let ep = Math.exp(im), en = Math.exp(-im);
        return [0.5 * (ep + en) * s, 0.5 * (ep - en) * c];
    }

    /**
     * Complex cosine.
     * cos(z) = cos(Re(z)) cosh(Im(z)) - j sin(Re(z)) sinh(Im(z))
     * @param re 
     * @param im 
     */
    public static ccos(re: number, im: number): [number, number] {
        let s = Math.sin(re), c = Math.cos(re);
        let ep = Math.exp(im), en = Math.exp(-im);
        return [0.5 * (ep + en) * c, -0.5 * (ep - en) * s];
    }

    /**
     * Complex tangent.
     *           sin(2*Re(z)) + j sinh(2*Im(z))
     * tan(z) = --------------------------------
     *           cosh(2*Im(z)) + cos(2*Re(z))
     * @param re 
     * @param im 
     */
    public static ctan(re: number, im: number): [number, number] {
        let re2 = re + re;
        let im2 = im + im;
        let ep = Math.exp(im2), en = Math.exp(-im2);
        let d = 0.5 * (ep + en) + Math.cos(re2);
        return [Math.sin(re2) / d, 0.5 * (ep - en) / d];
    }

    /**
     * Complex cotangent.
     *           sin(2*Re(z)) - j sinh(2*Im(z))
     * cot(z) = --------------------------------
     *            cosh(2*Im(z)) - cos(2*Re(z))
     * @param re 
     * @param im 
     */
    public static ccot(re: number, im: number): [number, number] {
        let re2 = re + re;
        let im2 = im + im;
        let ep = Math.exp(im2), en = Math.exp(-im2);
        let d = 0.5 * (ep + en) - Math.cos(re2);
        return [Math.sin(re2) / d, -0.5 * (ep - en) / d];
    }

    /**
     * Complex hyperbolic sine.
     * sinh(z) = sinh(Re(z)) cos(Im(z)) + j cosh(Re(z)) sin(Im(z))
     * @param re 
     * @param im 
     */
    public static csinh(re: number, im: number): [number, number] {
        let s = Math.sin(im), c = Math.cos(im);
        let ep = Math.exp(re), en = Math.exp(-re);
        return [0.5 * (ep - en) * c, 0.5 * (ep + en) * s];
    }

    /**
     * Complex hyperbolic cosine.
     * cosh(z) = cosh(Re(z)) cos(Im(z)) + j sinh(Re(z)) sin(Im(z))
     * @param re 
     * @param im 
     */
    public static ccosh(re: number, im: number): [number, number] {
        let s = Math.sin(im), c = Math.cos(im);
        let ep = Math.exp(re), en = Math.exp(-re);
        return [0.5 * (ep + en) * c, 0.5 * (ep - en) * s];
    }

    /**
     * Complex hyperbolic tangent.
     * tanh(z) = (e^z - e^-z) / (e^z + e^-z)
     *            sinh(2*Re(z)) + j sin(2*Im(z))
     * tanh(z) = --------------------------------
     *             cosh(2*Re(z)) + cos(2*Im(z))
     * @param re 
     * @param im 
     */
    public static ctanh(re: number, im: number): [number, number] {
        let re2 = re + re;
        let im2 = im + im;
        let ep = Math.exp(re2), en = Math.exp(-re2);
        let d = 0.5 * (ep + en) + Math.cos(im2);
        return [0.5 * (ep - en) / d, Math.sin(im2) / d];
    }

    /**
     * Complex hyperbolic cotangent.
     * coth(z) = (e^z + e^-z) / (e^z - e^-z)
     * Let z = x + j y, then
     *            sinh(2*Re(z)) - j sin(2*Im(z))
     * coth(z) = --------------------------------
     *             cosh(2*Re(z)) - cos(2*Im(z))
     * @param re 
     * @param im 
     */
    public static ccoth(re: number, im: number): [number, number] {
        let re2 = re + re;
        let im2 = im + im;
        let ep = Math.exp(re2), en = Math.exp(-re2);
        let d = 0.5 * (ep + en) - Math.cos(im2);
        return [0.5 * (ep - en) / d, -Math.sin(im2) / d];
    }

    /**
     * Complex exponentiation.
     * @param re
     * @param im 
     */
    public static cexp(re: number, im: number): [number, number] {
        let ep = Math.exp(re);
        return [ep * Math.cos(im), ep * Math.sin(im)];
    }

    /**
     * Complex logarithm.
     * Log(z) = log(sqrt(Re(z)^2 + Im(z)^2)) + j atan2(Im(z), Re(z))
     * @param re 
     * @param im 
     */
    public static clog(re: number, im: number): [number, number] {
        if (isNaN(re) || isNaN(im)) {
            return [NaN, NaN];
        }
        if (im === 0) {
            if (re >= 0) {
                return [Math.log(re), 0];
            } else {
                return [Math.log(-re), Math.PI];
            }
        } else {
            let l = CMath.length2(re, im);
            return [Math.log(l), Math.atan2(im, re)];
        }
    }

    /**
     * Complex power for the special case when both inputs are real but the
     * output may be complex.
     * @param reX 
     * @param reY 
     */
    public static cpowRR(reX: number, reY: number): [number, number] {
        if (reX >= 0 || Math.floor(reY) === reY) {
            return [Math.pow(reX, reY), 0];
        } else {
            return CMath.cpow(reX, 0, reY, 0);
        }
    }

    /**
     * Complex power.
     * @param reX 
     * @param imX 
     * @param reY 
     * @param imY 
     */
    public static cpow(reX: number, imX: number, reY: number, imY: number): [number, number] {
        if (isNaN(reX) || isNaN(imX) || isNaN(reY) || isNaN(imY)) {
            return [NaN, NaN];
        }
        // compute Ln(x)
        let [lxr, lxi] = CMath.clog(reX, imX);
        let zr = lxr * reY - lxi * imY;
        let zi = lxr * imY + lxi * reY;
        return CMath.cexp(zr, zi);
    }

}