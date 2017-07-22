import { CMathHelper } from './helper/mathHelper';

export class ComplexNumber {

    private _re: number;
    private _im: number;

    constructor(r: number = 0, i: number = 0) {
        this._re = r;
        this._im = i;
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
        let [re, im] = CMathHelper.cdivCC(this._re, this._im, y._re, y._im);
        return new ComplexNumber(re, im);
    }

    public divr(y: number): ComplexNumber {
        return new ComplexNumber(this._re / y, this._im / y);
    }

    public neg(): ComplexNumber {
        return new ComplexNumber(-this._re, -this._im);
    }

    public inv(): ComplexNumber {
        let [re, im] = CMathHelper.cReciprocal(this._re, this._im);
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
        return CMathHelper.length2(re, im);
    }

    public static angle2(re: number, im: number) {
        return Math.atan2(im, re);
    }

}