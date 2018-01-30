import { ElementWiseOpGenerator } from '../generator';
import { OutputDTypeResolver } from '../../core/dtype';
import { OpInput, OpOutput } from '../../commonTypes';

export interface IRoundingMathOpSet {

    /**
     * Applies `Math.floor()` for every element in the input.
     * For complex numbers, `Math.floor()` is applied to both the real part and
     * the imaginary part.
     */
    floor(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Applies `Math.ceil()` for every element in the input.
     * For complex numbers, `Math.ceil()` is applied to both the real part and
     * the imaginary part.
     */
    ceil(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Applies `Math.round()` for every element in the input.
     * For complex numbers, `Math.round()` is applied to both the real part and
     * the imaginary part.
     * Note: in JavaScript, `Math.round(-0.5)` gives -0, `Math.round(0.5)` give 1.
     */
    round(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Rounds every element in the input towards zero.
     * For complex numbers, the operation is applied to both the real part and
     * the imaginary part.
     */
    fix(x: OpInput, inPlace?: boolean): OpOutput;

}

export class RoundingMathOpSetFactory {

    public static create(generator: ElementWiseOpGenerator): IRoundingMathOpSet {

        const opFloor = generator.makeUnaryOp({
            opR: '$reY = Math.floor($reX);',
            opC: '$reY = Math.floor($reX); $imY = Math.floor($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat64
        });

        const opCeil = generator.makeUnaryOp({
            opR: '$reY = Math.ceil($reX);',
            opC: '$reY = Math.ceil($reX); $imY = Math.ceil($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat64
        });

        const opRound = generator.makeUnaryOp({
            opR: '$reY = Math.round($reX);',
            opC: '$reY = Math.round($reX); $imY = Math.round($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat64
        });

        const opFix = generator.makeUnaryOp({
            opR: '$reY = ($reX >= 0) ? Math.floor($reX) : Math.ceil($reX);',
            opC: '$reY = ($reX >= 0) ? Math.floor($reX) : Math.ceil($reX);\n' +
                 '$reX = ($imX >= 0) ? Math.floor($imX) : Math.ceil($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat64
        });

        return {
            floor: opFloor,
            ceil: opCeil,
            round: opRound,
            fix: opFix
        };
    }

}
