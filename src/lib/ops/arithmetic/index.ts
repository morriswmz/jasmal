import { IArithmeticOpProvider } from './definition';
import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { DType, OutputDTypeResolver, DTypeHelper } from '../../dtype';
import { Tensor } from "../../tensor";
import { ComplexNumber } from "../../complexNumber";
import { ShapeHelper } from "../../helper/shapeHelper";
import { CMathHelper } from "../../helper/mathHelper";

export class ArithmeticOpProviderFactory {
    public static create(): IArithmeticOpProvider {
        let compiler = TensorElementWiseOpCompiler.getInstance();
        return {
            add: compiler.makeBinaryOp({
                opRR: '$reZ = $reX + $reY;',
                opRC: '$reZ = $reX + $reY; $imZ = $imY;',
                opCR: '$reZ = $reX + $reY; $imZ = $imX;',
                opCC: '$reZ = $reX + $reY; $imZ = $imX + $imY;'
            }),
            sub: compiler.makeBinaryOp({
                opRR: '$reZ = $reX - $reY;',
                opRC: '$reZ = $reX - $reY; $imZ = -$imY;',
                opCR: '$reZ = $reX - $reY; $imZ = $imX;',
                opCC: '$reZ = $reX - $reY; $imZ = $imX - $imY;'
            }),
            neg: compiler.makeUnaryOp({
                opR: '$reY = -$reX;',
                opC: '$reY = -$reX; $imY = -$imX;'
            }),
            mul: compiler.makeBinaryOp({
                opRR: '$reZ = $reX * $reY;',
                // calculate the imaginary part first so no temporary variable
                // is needed.
                opRC: '$imZ = $reX * $imY; $reZ = $reX * $reY;',
                opCR: '$reZ = $reX * $reY; $imZ = $imX * $reY;',
                opCC: '$tmp1 = $reX; $reZ = $tmp1 * $reY - $imX * $imY; $imZ = $tmp1 * $imY + $imX * $reY;'
            }),
            div: compiler.makeBinaryOp({
                opRR: '$reZ = $reX / $reY;',
                opRC: '$tmp1 = cdivRC($reX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
                opCR: '$reZ = $reX / $reY; $imZ = $imX / $reY;',
                opCC: '$tmp1 = cdivCC($reX, $imX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];'
            }, {
                outputDTypeResolver: OutputDTypeResolver.bToFloat,
                inlineFunctions: {
                    'cdivRC': CMathHelper.cdivRC,
                    'cdivCC': CMathHelper.cdivCC
                }
            }),
            reciprocal: compiler.makeUnaryOp({
                opR: '$reY = 1 / $reX;',
                opC: '$tmp1 = cReciprocal($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
            }, {
                outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat,
                inlineFunctions: {
                    'cReciprocal': CMathHelper.cReciprocal
                }
            }),
            rem: compiler.makeBinaryOp({
                opRR: '$reZ = $reX % $reY;'
            })
        }
    }
}