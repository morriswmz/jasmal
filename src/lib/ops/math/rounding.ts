import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { OutputDTypeResolver } from '../../dtype';
import { OpInput, OpOutput } from '../../commonTypes';

export interface IRoundingMathOpSet {

    floor(x: OpInput, inPlace?: boolean): OpOutput;

    ceil(x: OpInput, inPlace?: boolean): OpOutput;

    round(x: OpInput, inPlace?: boolean): OpOutput;

}

export class RoundingMathOpSetFactory {

    public static create(compiler: TensorElementWiseOpCompiler): IRoundingMathOpSet {

        const opFloor = compiler.makeUnaryOp({
            opR: '$reY = Math.floor($reX);',
            opC: '$reY = Math.floor($reX); $imY = Math.floor($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
        });

        const opCeil = compiler.makeUnaryOp({
            opR: '$reY = Math.ceil($reX);',
            opC: '$reY = Math.ceil($reX); $imY = Math.ceil($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
        });

        const opRound = compiler.makeUnaryOp({
            opR: '$reY = Math.round($reX);',
            opC: '$reY = Math.round($reX); $imY = Math.round($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
        });

        return {
            floor: opFloor,
            ceil: opCeil,
            round: opRound
        };
    }

}