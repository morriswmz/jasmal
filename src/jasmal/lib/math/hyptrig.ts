export class HyperbolicTrigonometry {

    /**
     * sinh(x) = (exp(x) - exp(-x)) / 2.
     * @param x 
     */
    public static sinh(x: number): number {
        return 0.5 * (Math.exp(x) - Math.exp(-x));
    }

    public static cosh(x: number): number {
        return 0.5 * (Math.exp(x) + Math.exp(-x));
    }

    public static tanh(x: number): number {
        let e2: number;
        if (x >= 0) {
            e2 = Math.exp(-x - x);
            return (1 - e2) / (1 + e2);
        } else {
            e2 = Math.exp(x + x);
            return (e2 - 1) / (e2 + 1);
        }
    }

    public static coth(x: number): number {
        let e2: number;
        if (x >= 0) {
            e2 = Math.exp(-x - x);
            return (1 + e2) / (1 - e2);
        } else {
            e2 = Math.exp(x + x);
            return (e2 + 1) / (e2 - 1);
        }
    }

    /**
     * asinh(x) = ln(x + sqrt(x^2 + 1))
     * @param x 
     */
    public static asinh(x: number): number {
        if (x === 0) {
            return 0;
        }
        let invX = 1 / x;
        return Math.log(x + Math.abs(x) * Math.sqrt(invX * invX + 1));
    }

    /**
     * acosh(x) = ln(x + sqrt(x^2 - 1))
     * @param x 
     */
    public static acosh(x: number): number {
        if (x < 1) {
            return NaN;
        }
        let invX = 1 / x;
        return Math.log(x + Math.abs(x) * Math.sqrt(1 - invX * invX));
    }

    /**
     * atanh(x) = 0.5 * ln((1 + x) / (1 - x))
     * @param x 
     */
    public static atanh(x: number): number {
        if (x < -1 || x > 1) {
            return NaN;
        }
        return 0.5 * Math.log((1 + x) / (1 - x));
    }

    /**
     * acoth(x) = 0.5 * ln((x + 1) / (x - 1))
     * @param x 
     */
    public static acoth(x: number): number {
        if (x >= -1 && x <= 1) {
            return NaN;
        }
        return 0.5 * Math.log((x + 1) / (x - 1));
    }

}
