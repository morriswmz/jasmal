import { ElementWiseOpGenerator } from '../generator';
import { OpInput, OpOutput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { Tensor } from '../../tensor';
import { DataHelper } from '../../helper/dataHelper';

export interface ILogExpMathOpSet {

    /**
     * Computes the natural logarithm for each element in the input.
     * @param x
     * @param inPlace
     */
    log(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Computes the exponential for each element in the input.
     * @param x
     * @param inPlace
     */
    exp(x: OpInput, inPlace?: boolean): OpOutput;

}

export class LogExpMathOpSetFactory {

    public static create(generator: ElementWiseOpGenerator): ILogExpMathOpSet {

        const opExp = generator.makeUnaryOp({
            opR: '$reY = Math.exp($reX);',
            opC: '$tmp1 = CMath.cexp($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
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

        const opLogP = generator.makeUnaryOp({
            opR: '$reY = Math.log($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opLogA = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.clog($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.clog($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        return {
            log: opLog,
            exp: opExp,
        };

    }

}
