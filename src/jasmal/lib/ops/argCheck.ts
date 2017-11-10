import { DataHelper } from '../helper/dataHelper';
import { Tensor } from '../tensor';

export class ArgumentChecker {

    public static ensure(cond: boolean, msg: string, ...objs: any[]): void {
        if (cond) {
            return;
        }
        if (objs.length !== 0) {
            msg = msg.replace(/\{(\d+)\}/g, (_match, n) => {
                let idx = parseInt(n);
                return isNaN(idx) ? undefined : objs[idx];
            });
        }
        throw new Error(msg); 
    }

    public static ensureAllFinite(arr: ArrayLike<number>, subject?: string): void {
        if (DataHelper.isArrayAllFinite(arr)) {
            return;
        } 
        throw new Error(subject ? `${subject} cannot contain non-finite elements.` : 'All elements must be finite.');
    }

    public static ensureAllFiniteInTensor(x: Tensor): void {
        ArgumentChecker.ensureAllFinite(x.realData, 'Real part');
        if (x.hasComplexStorage()) {
            ArgumentChecker.ensureAllFinite(x.imagData, 'Imaginary part');
        }
    }

}
