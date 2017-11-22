import { IBinaryOpProvider } from './definition';
import { ElementWiseOpGenerator } from '../generator/index';
import { OutputDTypeResolver } from '../../dtype';
import { IJasmalModuleFactory, JasmalOptions } from '../../jasmal';

export class BinaryOpProviderFactory implements IJasmalModuleFactory<IBinaryOpProvider> {

    public constructor(private _generator: ElementWiseOpGenerator) {
    }

    public create(_options: JasmalOptions): IBinaryOpProvider {
        const opBitwiseAnd = this._generator.makeRealOutputBinaryOp({
            opRR: '$reZ = $reX & $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseOr = this._generator.makeRealOutputBinaryOp({
            opRR: '$reZ = $reX | $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseXor = this._generator.makeRealOutputBinaryOp({
            opRR: '$reZ = $reX ^ $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseNot = this._generator.makeRealOutputUnaryOp({
            opR: '$reY = ~($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uNoChangeExceptLogicToInt,
        });

        const opLeftShift = this._generator.makeRealOutputBinaryOp({
            opRR: '$reZ = ($reX) << ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opRightShiftSP = this._generator.makeRealOutputBinaryOp({
            opRR: '$reZ = ($reX) >> ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opRightShiftZF = this._generator.makeRealOutputBinaryOp({
            opRR: '$reZ = ($reX) >>> ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        return {
            bitwiseAnd: opBitwiseAnd,
            bitwiseOr: opBitwiseOr,
            bitwiseXor: opBitwiseXor,
            bitwiseNot: opBitwiseNot,
            leftShift: opLeftShift,
            rightShiftSP: opRightShiftSP,
            rightShiftZF: opRightShiftZF,
        };
    }

}

