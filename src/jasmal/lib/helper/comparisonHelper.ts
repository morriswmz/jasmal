/**
 * A collection of custom comparators.
 */
export class ComparisonHelper {
    
    /**
     * Returns 1 when a > b, 0 when a = b and -1 when a < b, where NaN is
     * treated as the largest number (larger than Infinity).
     * @param a 
     * @param b 
     */
    public static compareNumberAsc(a: number, b: number): number {
        // NaN is treated as the largest number
        if (isNaN(a)) {
            return isNaN(b) ? 0 : 1;
        } else {
            if (isNaN(b)) {
                return -1;
            }
            if (a > b) {
                return 1;
            } else if (a < b) {
                return -1;
            } else {
                return 0;
            }
        }
    }

    /**
     * Returns 1 when a < b, 0 when a = b and -1 when a > b, where NaN is
     * treated as the smallest number (smaller than -Infinity).
     * @param a 
     * @param b 
     */
    public static compareNumberDesc(a: number, b: number): number {
        return ComparisonHelper.compareNumberAsc(b, a);
    }

    /**
     * Compares a and b first. If a = b, then compares ia and ib. By definition,
     * ia and ib should never be equal to each other.
     * @param a 
     * @param b 
     * @param ia 
     * @param ib 
     */
    public static compareNumberWithIndexAsc(a: number, b: number, ia: number, ib: number): number {
        // NaN is treated as the largest number
        // ia and ib can never be equal
        if (isNaN(a)) {
            return isNaN(b) ? (ia > ib ? 1 : -1) : 1;
        } else {
            if (isNaN(b)) {
                return -1;
            }
            if (a > b) {
                return 1;
            } else if (a < b) {
                return -1;
            } else {
                return ia > ib ? 1 : -1;
            }
        }
    }

}
