import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { OpInput, OpOutput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { Tensor } from '../../tensor';
import { DataHelper } from '../../helper/dataHelper';
import { CMath } from '../../complexNumber';

export interface ILogExpMathOpSet {

    /**
     * Computes element-wise natural logarithm.
     * @param x
     * @param inPlace
     */
    log(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Computes element-wise exponentiation.
     * @param x
     * @param inPlace
     */
    exp(x: OpInput, inPlace?: boolean): OpOutput;

}

export class LogExpMathOpSetFactory {

    public static create(compiler: TensorElementWiseOpCompiler): ILogExpMathOpSet {

        const opExp = compiler.makeUnaryOp({
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

        const opLogP = compiler.makeUnaryOp({
            opR: '$reY = Math.log($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opLogA = compiler.makeUnaryOp({
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