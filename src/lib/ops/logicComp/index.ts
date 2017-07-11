import { ILogicComparisonOpProvider } from '../definition';
import { NOT_IMPLEMENTED } from '../../constant';
import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { DType, OutputDTypeResolver } from '../../dtype';
import { OpInput } from '../../commonTypes';
import { Tensor } from '../../tensor';
import { DataHelper } from '../../helper/dataHelper';

export class LogicComparisonOpProviderFactory {
    
    public static create(): ILogicComparisonOpProvider {
        
        const compiler = TensorElementWiseOpCompiler.GetInstance();

        const opEq = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX === $reY) ? 1 : 0;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogic,
            noInPlaceOperation: true
        });

        const opNeq = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX !== $reY) ? 1 : 0;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogic,
            noInPlaceOperation: true
        });

        const opGt = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX > $reY) ? 1 : 0;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogicRealOnly,
            noInPlaceOperation: true
        });

        const opGe = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX >= $reY) ? 1 : 0;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogicRealOnly,
            noInPlaceOperation: true
        });

        const opLt = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX < $reY) ? 1 : 0;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogicRealOnly,
            noInPlaceOperation: true
        });

        const opLe = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX <= $reY) ? 1 : 0;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogicRealOnly,
            noInPlaceOperation: true
        });

        const opAnd = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX !== 0) & ($reY !== 0);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogicRealOnly,
        });

        const opOr = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX !== 0) | ($reY !== 0);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogicRealOnly,
        });

        const opXor = compiler.makeBinaryOp({
            opRR: '$reZ = ($reX !== 0) ^ ($reY !== 0);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToLogicRealOnly,
        });

        const opNot = compiler.makeUnaryOp({
            opR: '$reY = ($reX !== 0) ? 0 : 1;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToLogicRealOnly,
        });

        const opAll = (x: OpInput): boolean => {
            let v = Tensor.analyzeOpInput(x);
            if (v.hasOnlyOneElement) {
                return v.re !== 0 || v.im !== 0;
            } else {
                let re = v.reArr;
                if (v.isComplex) {
                    let im = v.imArr;
                    for (let i = 0;i < re.length;i++) {
                        if (re[i] === 0 && im[i] === 0) {
                            return false;
                        }
                    }
                    return true;
                } else {
                    return DataHelper.isArrayAllNonZeros(re);
                }
            }
        };

        const opAny = (x: OpInput): boolean => {
            let v = Tensor.analyzeOpInput(x);
            if (v.hasOnlyOneElement) {
                return v.re !== 0 || v.im !== 0;
            } else {
                let re = v.reArr;
                if (v.isComplex) {
                    let im = v.imArr;
                    for (let i = 0;i < re.length;i++) {
                        if (re[i] !== 0 || im[i] !== 0) {
                            return true;
                        }
                    }
                    return false;
                } else {
                    return !DataHelper.isArrayAllZeros(re);
                }  
            }
        };

        return {
            eq: opEq,
            neq: opNeq,
            gt: opGt,
            ge: opGe,
            lt:opLt,
            le: opLe,
            and: opAnd,
            or: opOr,
            xor: opXor,
            not: opNot,
            all: opAll,
            any: opAny
        }

    }
}