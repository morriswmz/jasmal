import { IArithmeticOpProvider } from './definition';
import { ElementWiseOpGenerator } from '../generator';
import { DType, OutputDTypeResolver, DTypeHelper } from '../../dtype';
import { Tensor } from "../../tensor";
import { ComplexNumber, CMath } from "../../complexNumber";
import { ShapeHelper } from "../../helper/shapeHelper";

export class ArithmeticOpProviderFactory {
    public static create(generator: ElementWiseOpGenerator): IArithmeticOpProvider {
        return {
            add: generator.makeBinaryOp({
                opRR: '$reZ = $reX + $reY;',
                opRC: '$reZ = $reX + $reY; $imZ = $imY;',
                opCR: '$reZ = $reX + $reY; $imZ = $imX;',
                opCC: '$reZ = $reX + $reY; $imZ = $imX + $imY;'
            }, {
                outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
            }),
            sub: generator.makeBinaryOp({
                opRR: '$reZ = $reX - $reY;',
                opRC: '$reZ = $reX - $reY; $imZ = -$imY;',
                opCR: '$reZ = $reX - $reY; $imZ = $imX;',
                opCC: '$reZ = $reX - $reY; $imZ = $imX - $imY;'
            }, {
                outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
            }),
            neg: generator.makeUnaryOp({
                opR: '$reY = -$reX;',
                opC: '$reY = -$reX; $imY = -$imX;'
            }, {
                outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
            }),
            mul: generator.makeBinaryOp({
                opRR: '$reZ = $reX * $reY;',
                // calculate the imaginary part first so no temporary variable
                // is needed.
                opRC: '$imZ = $reX * $imY; $reZ = $reX * $reY;',
                opCR: '$reZ = $reX * $reY; $imZ = $imX * $reY;',
                opCC: '$tmp1 = $reX; $reZ = $tmp1 * $reY - $imX * $imY; $imZ = $tmp1 * $imY + $imX * $reY;'
            }, {
                outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
            }),
            div: generator.makeBinaryOp({
                opRR: '$reZ = $reX / $reY;',
                opRC: '$tmp1 = CMath.cdivRC($reX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
                opCR: '$reZ = $reX / $reY; $imZ = $imX / $reY;',
                opCC: '$tmp1 = CMath.cdivCC($reX, $imX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];'
            }, {
                outputDTypeResolver: OutputDTypeResolver.bToFloat
            }),
            reciprocal: generator.makeUnaryOp({
                opR: '$reY = 1 / $reX;',
                opC: '$tmp1 = CMath.cReciprocal($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
            }, {
                outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
            }),
            rem: generator.makeBinaryOp({
                opRR: '$reZ = $reX % $reY;'
            }, {
                outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
            })
        }
    }
}