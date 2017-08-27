import { ElementWiseOpGenerator } from '../generator';
import { DType, OutputDTypeResolver } from '../../dtype';
import { OpInput, OpOutput } from '../../commonTypes';

export interface IBasicMathOpSet {

    /**
     * Computes the absolute value for real numbers, and magnitude for complex
     * numbers.
     */
    abs(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Sign function. For real numbers, sign(x) returns 1 if x > 0, 0 if x = 0,
     * and -1 if x < 0. For non-zero complex numbers, sign(x) = x / abs(x).
     */
    sign(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Computes element-wise minimum between two compatible inputs.
     */
    min2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;
    
    /**
     * Computes element-wise maximum between two compatible inputs.
     */
    max2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Evaluates complex conjugate for each element in the input.
     */
    conj(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Computes the phase angle for each element in the input.
     */
    angle(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Converts angles from radians to degrees.
     */
    rad2deg(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Converts angles from degrees to radians.
     */
    deg2rad(x: OpInput, inPlace?: boolean): OpOutput;
    
}

export class BasicMathOpSetFactory {

    public static create(generator: ElementWiseOpGenerator): IBasicMathOpSet {

        const opAbs = generator.makeUnaryOp({
            opR: '$reY = Math.abs($reX);',
            opC: '$reY = CMath.length2($reX, $imX);',
        }, {
            // custom rule here, only convert to float when the input is complex
            outputDTypeResolver: (t, isComplex) => isComplex ? DType.FLOAT64 : t
        });

        const opSign = generator.makeUnaryOp({
            opR: '$reY = $reX > 0 ? 1 : ($reX < 0 ? -1 : ($reX === 0 ? 0 : NaN));',
            opC: '$tmp1 = CMath.length2($reX, $imX);' +
                 'if ($tmp1 === 0) { $reY = 0; $imY = 0; } else { $reY = $reX / $tmp1; $imY = $imX / $tmp1; }'
        }, {
            outputDTypeResolver: (t, isComplex) => isComplex ? DType.FLOAT64 : t
        });

        const opMin2 = generator.makeBinaryOp({
            opRR: '$reZ = Math.min($reX, $reY);'
        });

        const opMax2 = generator.makeBinaryOp({
            opRR: '$reZ = Math.max($reX, $reY);'
        });

        const opConj = generator.makeUnaryOp({
            opR: '$reY = $reX;',
            opC: '$reY = $reX; $imY = -$imX;'
        });

        const opAngle = generator.makeUnaryOp({
            opR: '$reY = 0;',
            opC: '$reY = Math.atan2($imX, $reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opRad2Deg = generator.makeUnaryOp({
            opR: '$reY = 180 / Math.PI * $reX;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opDeg2Rad = generator.makeUnaryOp({
            opR: '$reY = Math.PI / 180 * $reX;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        return {
            abs: opAbs,
            sign: opSign,
            min2: opMin2,
            max2: opMax2,
            conj: opConj,
            angle: opAngle,
            rad2deg: opRad2Deg,
            deg2rad: opDeg2Rad
        };

    }

}
