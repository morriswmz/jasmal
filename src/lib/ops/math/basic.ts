import { CMath } from '../../complexNumber';
import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { DType, OutputDTypeResolver } from '../../dtype';
import { OpInput, OpOutput } from '../../commonTypes';

export interface IBasicMathOpSet {

    abs(x: OpInput, inPlace?: boolean): OpOutput;

    sign(x: OpInput, inPlace?: boolean): OpOutput;

    min2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;
    
    max2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    conj(x: OpInput, inPlace?: boolean): OpOutput;

    angle(x: OpInput, inPlace?: boolean): OpOutput;

    rad2deg(x: OpInput, inPlace?: boolean): OpOutput;

    deg2rad(x: OpInput, inPlace?: boolean): OpOutput;
    
}

export class BasicMathOpSetFactory {

    public static create(compiler: TensorElementWiseOpCompiler): IBasicMathOpSet {

        const opAbs = compiler.makeUnaryOp({
            opR: '$reY = Math.abs($reX);',
            opC: '$reY = CMath.length2($reX, $imX);',
        }, {
            // custom rule here, only convert to float when the input is complex
            outputDTypeResolver: (t, isComplex) => isComplex ? DType.FLOAT64 : t
        });

        const opSign = compiler.makeUnaryOp({
            opR: '$reY = $reX > 0 ? 1 : ($reX < 0 ? -1 : ($reX === 0 ? 0 : NaN));',
            opC: '$tmp1 = CMath.length2($reX, $imX);' +
                 'if ($tmp1 === 0) { $reY = 0; $imY = 0; } else { $reY = $reX / $tmp1; $imY = $imX / $tmp1; }'
        }, {
            outputDTypeResolver: (t, isComplex) => isComplex ? DType.FLOAT64 : t
        });

        const opMin2 = compiler.makeBinaryOp({
            opRR: '$reZ = Math.min($reX, $reY);'
        });

        const opMax2 = compiler.makeBinaryOp({
            opRR: '$reZ = Math.max($reX, $reY);'
        });

        const opConj = compiler.makeUnaryOp({
            opR: '$reY = $reX;',
            opC: '$reY = $reX; $imY = -$imX;'
        });

        const opAngle = compiler.makeUnaryOp({
            opR: '$reY = 0;',
            opC: '$reY = Math.atan2($imX, $reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opRad2Deg = compiler.makeUnaryOp({
            opR: '$reY = 180 / Math.PI * $reX;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opDeg2Rad = compiler.makeUnaryOp({
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