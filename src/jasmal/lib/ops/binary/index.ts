import { IBinaryOpProvider } from './definition';
import { ElementWiseOpGenerator } from '../generator/index';
import { OutputDTypeResolver } from '../../dtype';

export class BinaryOpProviderFactory {

    public static create(generator: ElementWiseOpGenerator): IBinaryOpProvider {

        const opBitwiseAnd = generator.makeRealOutputBinaryOp({
            opRR: '$reZ = $reX & $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseOr = generator.makeRealOutputBinaryOp({
            opRR: '$reZ = $reX | $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseXor = generator.makeRealOutputBinaryOp({
            opRR: '$reZ = $reX ^ $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseNot = generator.makeRealOutputUnaryOp({
            opR: '$reY = ~($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uNoChangeExceptLogicToInt,
        });

        const opLeftShift = generator.makeRealOutputBinaryOp({
            opRR: '$reZ = ($reX) << ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opRightShiftSP= generator.makeRealOutputBinaryOp({
            opRR: '$reZ = ($reX) >> ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opRightShiftZF = generator.makeRealOutputBinaryOp({
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

