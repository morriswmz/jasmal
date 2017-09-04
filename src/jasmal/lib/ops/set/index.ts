import { ISetOpProvider } from './definition';
import { OpInput } from '../../commonTypes';
import { Tensor } from '../../tensor';
import { DataHelper } from '../../helper/dataHelper';
import { ComparisonHelper } from '../../helper/comparisonHelper';

export class SetOpProviderFactory {

    public static create(): ISetOpProvider {

        function opUnique(x: OpInput, outputIndices: false): Tensor;  
        function opUnique(x: OpInput, outputIndices: true): [Tensor, number[], number[][]];        
        function opUnique(x: OpInput, outputIndices: boolean = false): Tensor | [Tensor, number[], number[][]] {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let n = X.size;
            let indices = DataHelper.naturalNumbers(n);
            let dataRe = X.realData;
            let dataIm: ArrayLike<number>;
            let ix: number[][] = [];
            let iy: number[] = [];
            let i: number, last: number, k: number;
            let curRe: number, curIm: number, newRe: number, newIm: number;
            let nUnique = 1;
            let uniqueRe: number[] = [];
            let uniqueIm: number[] = [];
            let Y: Tensor;
            if (X.hasNonZeroComplexStorage()) {
                dataIm = X.imagData;
                // sort complex numbers
                indices.sort((ia, ib) => {
                    // lexicographic order
                    let a = dataRe[ia], b = dataRe[ib];
                    let reOrder: number;
                    if (isNaN(a)) {
                        reOrder = isNaN(b) ? 0 : 1;
                    } else {
                        if (isNaN(b)) {
                            reOrder = -1;
                        } else {
                            reOrder = a > b ? 1 : (a === b ? 0 : -1);
                        }
                    }
                    return reOrder !== 0 ? reOrder : ComparisonHelper.compareNumberWithIndexAsc(dataIm[ia], dataIm[ib], ia, ib);
                });
                // deduplicate
                last = 0;
                k = indices[0];
                curRe = dataRe[k];
                curIm = dataIm[k];
                for (i = 1;i < n;i++) {
                    newRe = dataRe[indices[i]];
                    newIm = dataIm[indices[i]];
                    if (newRe !== curRe || newIm !== curIm) {
                        // found a different value
                        uniqueRe.push(curRe);
                        uniqueIm.push(curIm);
                        curRe = newRe;
                        curIm = newIm;
                        nUnique++;
                        if (outputIndices) {
                            iy.push(k);
                            k = indices[i];
                            ix.push(indices.slice(last, i));
                            last = i;
                        }
                    }
                }
                // process the last one
                uniqueRe.push(curRe);
                uniqueIm.push(curIm);
                if (outputIndices) {
                    iy.push(k);
                    ix.push(indices.slice(last, i));
                }
                Y = Tensor.fromArray(uniqueRe, uniqueIm, X.dtype);
            } else {
                // sort real numbers
                indices.sort((ia, ib) => ComparisonHelper.compareNumberWithIndexAsc(dataRe[ia], dataRe[ib], ia, ib));
                // deduplicate
                last = 0;
                k = indices[0];
                curRe = dataRe[k];
                for (i = 1;i < n;i++) {
                    newRe = dataRe[indices[i]];
                    if (newRe !== curRe) {
                        // found a different value
                        uniqueRe.push(curRe);
                        curRe = newRe;
                        nUnique++;
                        if (outputIndices) {
                            iy.push(k);
                            k = indices[i];
                            ix.push(indices.slice(last, i));
                            last = i;
                        }
                    }
                }
                // process the last one
                uniqueRe.push(curRe);
                if (outputIndices) {
                    iy.push(k);
                    ix.push(indices.slice(last, i));
                }
                Y = Tensor.fromArray(uniqueRe, [], X.dtype);
            }
            return outputIndices ? [Y, iy, ix] : Y;
        };

        return {
            unique: opUnique
        };
    }

}
