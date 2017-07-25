import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { OpInput, OpOutput } from '../../commonTypes';
import { OutputDTypeResolver } from "../../dtype";
import { Tensor } from "../../tensor";
import { DataHelper } from "../../helper/dataHelper";

export interface ILogExpMathOpSet {

    log(x: OpInput, inPlace?: boolean): OpOutput;

    exp(x: OpInput, inPlace?: boolean): OpOutput;

}

export class LogExpMathOpSetFactory {

    public static create(compiler: TensorElementWiseOpCompiler): ILogExpMathOpSet {

        const opExp = compiler.makeUnaryOp({
            opR: '$reY = Math.exp($reX);',
            opC: '$tmp1 = Math.exp($reX); $reY = $tmp1 * Math.cos($imX); $imY = $tmp1 * Math.sin($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opLog = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isComplex || infoX.re < 0 || DataHelper.anyNegative(infoX.reArr)) {
                return opLogA(infoX, inPlace);
            } else {
                return opLogP(infoX, inPlace);
            }
        };

        const opLogP = compiler.makeUnaryOp({
            opR: '$reY = Math.log($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opLogA = compiler.makeUnaryOp({
            opR: '$tmp1 = clog($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = clog($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'clog': (re: number, im: number): [number, number] => {
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
                        // Ln(x + j y) = ln(sqrt(x^2 + y^2)) + j atan2(y, x)
                        // inline length2
                        let absRe = Math.abs(re),
                            absIm = Math.abs(im);
                        let l: number, ratio: number;
                        if (absRe > absIm) {
                            ratio = absIm / absRe;
                            l = absRe * Math.sqrt(1.0 + ratio * ratio);
                        } else {
                            if (absIm === 0) {
                                l =  absRe;
                            } else {
                                ratio = absRe / absIm;
                                l =  absIm * Math.sqrt(1.0 + ratio * ratio);
                            }
                        }
                        return [Math.log(l), Math.atan2(im, re)];
                    }
                }
            }
        });

        return {
            log: opLog,
            exp: opExp,
        };

    }

}