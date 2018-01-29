import { IArithmeticOpProvider } from './definition';
import { ElementWiseOpGenerator } from '../generator';
import { OutputDTypeResolver } from '../../dtype';
import { IJasmalModuleFactory, JasmalOptions } from '../../jasmal';

export class ArithmeticOpProviderFactory implements IJasmalModuleFactory<IArithmeticOpProvider> {

    constructor(private _generator: ElementWiseOpGenerator) {
    }

    public create(_options: JasmalOptions): IArithmeticOpProvider {

        const opAdd = this._generator.makeBinaryOp({
            opRR: '$reZ = $reX + $reY;',
            opRC: '$reZ = $reX + $reY; $imZ = $imY;',
            opCR: '$reZ = $reX + $reY; $imZ = $imX;',
            opCC: '$reZ = $reX + $reY; $imZ = $imX + $imY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opSub = this._generator.makeBinaryOp({
            opRR: '$reZ = $reX - $reY;',
            opRC: '$reZ = $reX - $reY; $imZ = -$imY;',
            opCR: '$reZ = $reX - $reY; $imZ = $imX;',
            opCC: '$reZ = $reX - $reY; $imZ = $imX - $imY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opNeg = this._generator.makeUnaryOp({
            opR: '$reY = -$reX;',
            opC: '$reY = -$reX; $imY = -$imX;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat64
        });

        const opMul = this._generator.makeBinaryOp({
            opRR: '$reZ = $reX * $reY;',
            // For the RC and CR case, x and y cannot be the same.
            // We calculate the imaginary part first so no temporary
            // variable is needed.
            opRC: '$imZ = $reX * $imY; $reZ = $reX * $reY;',
            opCR: '$reZ = $reX * $reY; $imZ = $imX * $reY;',
            opCC: '$tmp1 = $reX; $tmp2 = $reY; $reZ = $tmp1 * $tmp2 - $imX * $imY; $imZ = $tmp1 * $imY + $imX * $tmp2;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opDiv = this._generator.makeBinaryOp({
            opRR: '$reZ = $reX / $reY;',
            opRC: '$tmp1 = CMath.cdivRC($reX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
            opCR: '$reZ = $reX / $reY; $imZ = $imX / $reY;',
            opCC: '$tmp1 = CMath.cdivCC($reX, $imX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToFloat64
        });

        const opReciprocal = this._generator.makeUnaryOp({
            opR: '$reY = 1 / $reX;',
            opC: '$tmp1 = CMath.cReciprocal($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat64
        });
        
        const opRem =  this._generator.makeRealOutputBinaryOp({
            opRR: '$reZ = $reX % $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });
        
        return {
            add: opAdd,
            sub: opSub,
            neg: opNeg,
            mul: opMul,
            div: opDiv,
            reciprocal: opReciprocal,
            rem: opRem
        };
    }
    
}
