/**
 * A collection of custom comparators.
 */
export class ComparisonHelper {
    
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

    public static compareNumberDesc(a: number, b: number): number {
        return ComparisonHelper.compareNumberAsc(b, a);
    }

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
