import { Tensor } from '../../tensor';
import { OpInput, OpOutput, OpInputInternal } from '../../commonTypes'; 
import { DataBlock } from '../../storage';
import { ComplexNumber, CMath } from '../../complexNumber';
import { ShapeHelper, BroadcastingCheckResult } from '../../helper/shapeHelper';
import { TemplateEngine } from './templateEngine';
import { T_BLOCK_TEMPLATE, S_BLOCK_TEMPLATE, UNARY_OP_TEMPLATE,
         BIN_EL_OP_TEMPLATE, SS_BLOCK_TEMPLATE, ST_BLOCK_TEMPLATE,
         TS_BLOCK_TEMPLATE, TT_BLOCK_TEMPLATE, TT_NORMAL_BLOCK_TEMPLATE,
         TT_BROADCAST_SUB_BLOCK_TEMPLATE, TT_BROADCAST_BLOCK_TEMPLATE } from './templates';
import { OutputDTypeResolver, DType, DTypeHelper } from '../../dtype';

/**
 * General rules:
 * 1. If all the inputs are scalars, the output is scalar.
 * 2. If any of the input is an array/tensor, the output will be a tensor.
 */

/**
 * Represents a binary operation.
 */
export type GenericBinaryOp = (x: OpInputInternal, y: OpInputInternal, inPlace?: boolean) => OpOutput;
/**
 * Represents a unary operation.
 */
export type GenericUnaryOp = (x: OpInputInternal, inPlace?: boolean) => OpOutput;
/**
 * Represents a unary operation with a parameter.
 */
export type OneParamUnaryOp = (x: OpInputInternal, p: number, inPlace?: boolean) => OpOutput;

/**
 * Defines the core operations.
 */
export interface BinaryEWOpTemplate {
    /**
     * A template that defines the operation between two real operands.
     * Available symbols: $reX, $reY, $reZ, $tmp1, $tmp2, $tmp3
     * Example:
     *  $reZ = $reX + $reY;
     */
    opRR: string;
    /**
     * A template that defines the operation between a real operand and a
     * complex operand.
     * Available symbols: $reX, $reY, $imY, $reZ, $imZ, $tmp1, $tmp2, $tmp3
     * Example:
     *  $reZ = $reX + $reY;
     *  $imZ = $imY;
     */
    opRC?: string;
    /**
     * A template that defines the operation between a complex operand and a
     * real operand.
     * Available symbols: $reX, $imX, $reY, $reZ, $imZ, $tmp1, $tmp2, $tmp3
     * Example:
     *  $reZ = $reX + $reY;
     *  $reZ = $imX;
     */
    opCR?: string;
    /**
     * A template that defines the operation between two complex operands.
     * Available symbols: $reX, $imX, $reY, $imY, $reZ, $imZ, $tmp1, $tmp2,
     *                    $tmp3
     * Example:
     *  $reZ = $reX + $reY;
     *  $imZ = $imX + $imY;
     */
    opCC?: string;
}

export interface UnaryEWOpConfig {
    outputDTypeResolver?: (t: DType, isComplex: boolean) => DType | undefined;
    noInPlaceOperation?: boolean;
    inlineFunctions?: {[key: string]: Function};
}

export interface BinaryEWOpConfig {
    /**
     * Default value is OutputDTypeResolver.bWider.
     */
    outputDTypeResolver?: (t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean) => DType | undefined;
    noInPlaceOperation?: boolean;
    inlineFunctions?: {[key: string]: Function};
}

export interface UnaryEWOpTemplate {
    /**
     * A template that defines the operation on a real operand.
     * Available symbols: $reX, $reY, $imY, $tmp1, $tmp2, $tmp3
     * If $imY$ shows up in the template, the compiler is smart enough to
     * generate code that produce a complex tensor.
     * Example:
     *  $reY = -$reX;
     */
    opR: string;
    /**
     * A template that defines the operation on a complex operand.
     * Available symbols: $reX, $imX, $reY, $imY, $tmp1, $tmp2, $tmp3
     * If $imY$ does not show up in the template, the compiler is smart enough
     * to generate code that produce a real tensor.
     * If this template is omitted, the compiled function will only accept
     * real inputs.
     * Example:
     *  $reY = Math.abs($reX);
     *  $imY = Math.abs($imX);
     */
    opC?: string;
}

/**
 * Classes and helpers required by the compiled function. An object implementing
 * this interface will be passed as an argument so these classes and helpers 
 * can be accessed in the function body.
 */
interface OpCommonDependencies {
    Tensor: Function;
    ComplexNumber: Function;
    CMath: Function;
    computeStrides: (shape: number[]) => number[];
    isWiderType(original: DType, newType: DType): boolean;
    dTypeToString: (dtype: DType) => string;
}

interface BinaryOpDependencies extends OpCommonDependencies {
    compareShape: (shape1: number[], shape2: number[]) => boolean;
    checkBroadcastingCompatibility: (shapeX: number[], shapeY: number[]) => BroadcastingCheckResult;
    determineOutputType: (t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean) => DType | undefined;
}

interface UnaryOpDependencies extends OpCommonDependencies {
    determineOutputType: (t: DType, isComplex: boolean) => DType | undefined;
}

export class TensorElementWiseOpCompiler {
    
    private _engine: TemplateEngine;

    protected constructor() {
        this._engine = new TemplateEngine();
    }

    private static _instance: TensorElementWiseOpCompiler;

    public static getInstance(): TensorElementWiseOpCompiler {
        if (!TensorElementWiseOpCompiler._instance) {
            TensorElementWiseOpCompiler._instance = new TensorElementWiseOpCompiler();
        }
        return TensorElementWiseOpCompiler._instance;
    }

    public makeUnaryOp(opTemplate: UnaryEWOpTemplate,
                       opConfig?: UnaryEWOpConfig): GenericUnaryOp {
        let funcBody = this.generateUnaryOpFuncBody(opTemplate, opConfig);
        let deps = this._getUnaryOpDependencies(opConfig);
        let fn = (new Function('__dep__', funcBody))(deps);
        return fn;
    }

    public makeOneParamUnaryOp(opTemplate: UnaryEWOpTemplate,
                               opConfig?: UnaryEWOpConfig): OneParamUnaryOp {
        let funcBody = this.generateUnaryOpFuncBody(opTemplate, opConfig, true);
        let deps = this._getUnaryOpDependencies(opConfig);
        let fn = (new Function( '__dep__', funcBody))(deps);
        return fn;
    }

    private _getUnaryOpDependencies(opConfig?: UnaryEWOpConfig): UnaryOpDependencies {
        return {
            Tensor: Tensor,
            ComplexNumber: ComplexNumber,
            CMath: CMath,
            computeStrides: ShapeHelper.computeStrides,
            determineOutputType: opConfig && opConfig.outputDTypeResolver
                ? opConfig.outputDTypeResolver : OutputDTypeResolver.uNoChange,
            isWiderType: DTypeHelper.isWiderType,
            dTypeToString: DTypeHelper.dTypeToString
        };
    }

    public generateUnaryOpFuncBody(opTemplate: UnaryEWOpTemplate,
                                   opConfig?: UnaryEWOpConfig,
                                   hasParam?: boolean): string {
        // check templates
        // we allow complex input by default
        let realInputOnly = opTemplate.opC == undefined;
        let templateConfig = {
            NO_COMPLEX_INPUT: realInputOnly,
            NO_IN_PLACE: opConfig ? !!opConfig.noInPlaceOperation : false,
            HAS_PARAM: hasParam || false
        };
        const opRSymbolSet = hasParam
            ? ['$reX', '$reY', '$imY', '$tmp1', '$tmp2', '$tmp3', '$tmp4', '$param']
            : ['$reX', '$reY', '$imY', '$tmp1', '$tmp2', '$tmp3', '$tmp4'];
        const opCSymbolSet = hasParam
            ? ['$reX', '$imX', '$reY', '$imY', '$tmp1', '$tmp2', '$tmp3', '$tmp4', '$param']
            : ['$reX', '$imX', '$reY', '$imY', '$tmp1', '$tmp2', '$tmp3', '$tmp4'];
        let opRSymbols = this._checkUsedSymbols(opTemplate.opR, opRSymbolSet);
        let opCSymbols = realInputOnly ? [] : this._checkUsedSymbols(<string>opTemplate.opC, opCSymbolSet);
        let tBlockTemplate: string;
        if (opRSymbols.indexOf('$imY') >= 0) {
            templateConfig['OUTPUT_R_COMPLEX'] = true;
        }
        if (opCSymbols.indexOf('$imY') >= 0) {
            templateConfig['OUTPUT_C_COMPLEX'] = true;
        }
        const symbolMapScalar = {
            '$reX': 'reXScalar',
            '$imX': 'imXScalar',
            '$reY': 'reYScalar',
            '$imY': 'imYScalar',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4',
            '$param': 'param'
        };
        const symbolMapTensor = {
            '$reX': 'reX[i]',
            '$imX': 'imX[i]',
            '$reY': 'reY[i]',
            '$imY': 'imY[i]',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4',
            '$param': 'param'
        };
        const blockMapTensor = {
            '$RBlock': this._engine.generate(opTemplate.opR, symbolMapTensor, templateConfig),
            '$CBlock': realInputOnly
                ? undefined
                : this._engine.generate(<string>opTemplate.opC, symbolMapTensor, templateConfig)
        }
        const blockMapScalar = {
            '$RBlock': this._engine.generate(opTemplate.opR, symbolMapScalar, templateConfig),
            '$CBlock': realInputOnly
                ? undefined
                : this._engine.generate(<string>opTemplate.opC, symbolMapScalar, templateConfig)
        }
        const blockMap = {
            '$InlineFunctions': this._flattenInlineFunctions(opConfig && opConfig.inlineFunctions ? opConfig.inlineFunctions : {}),
            '$TBlock': this._engine.generate(T_BLOCK_TEMPLATE, blockMapTensor, templateConfig),
            '$SBlock': this._engine.generate(S_BLOCK_TEMPLATE, blockMapScalar, templateConfig)
        }
        return this._engine.generate(UNARY_OP_TEMPLATE, blockMap, templateConfig);
    }

    public makeBinaryOp(opTemplate: BinaryEWOpTemplate,
                        config?: BinaryEWOpConfig): GenericBinaryOp {
        let funcBody = this.generateBinaryOpFuncBody(opTemplate, config);
        let outputDTypeResolver = (config && config.outputDTypeResolver)
            ? config.outputDTypeResolver
            : OutputDTypeResolver.bWider;
        let deps: BinaryOpDependencies = {
            Tensor: Tensor,
            ComplexNumber: ComplexNumber,
            CMath: CMath,
            computeStrides: ShapeHelper.computeStrides,
            compareShape: ShapeHelper.compareShape,
            determineOutputType: outputDTypeResolver,
            checkBroadcastingCompatibility: ShapeHelper.checkBroadcastingCompatibility,
            isWiderType: DTypeHelper.isWiderType,
            dTypeToString: DTypeHelper.dTypeToString
        };
        let fn = (new Function('__dep__', funcBody))(deps);
        return fn;
    }

    public generateBinaryOpFuncBody(opTemplate: BinaryEWOpTemplate, opConfig?: BinaryEWOpConfig): string {
        let realInputOnly = false;
        if (opTemplate.opCR == undefined && opTemplate.opRC == undefined && opTemplate.opCC == undefined) {
            realInputOnly = true;
        } else if (!(opTemplate.opCR != undefined && opTemplate.opRC != undefined && opTemplate.opCC != undefined)) {
            throw new Error('Partially defined opCR, opRC, and opCC.');
        }
        // check templates
        let templateConfig: {[key: string]: boolean} = {
            NO_COMPLEX_INPUT: realInputOnly,
            NO_IN_PLACE: opConfig ? !!opConfig.noInPlaceOperation : false
        };
        const symbolSetRR = ['$reX', '$reY', '$reZ', '$imZ', '$tmp1', '$tmp2', '$tmp3', '$tmp4'];
        const symbolSetRC = ['$reX', '$reY', '$imY', '$reZ', '$imZ', '$tmp1', '$tmp2', '$tmp3', '$tmp4'];
        const symbolSetCR = ['$reX', '$imX', '$reY', '$reZ', '$imZ', '$tmp1', '$tmp2', '$tmp3', '$tmp4'];
        const symbolSetCC = ['$reX', '$imX', '$reY', '$imY', '$reZ', '$imZ', '$tmp1', '$tmp2', '$tmp3', '$tmp4'];
        if (this._checkUsedSymbols(opTemplate.opRR, symbolSetRR).indexOf('$imZ') >= 0) {
            templateConfig['OUTPUT_RR_COMPLEX'] = true;
        }
        if (!realInputOnly) {
            if (this._checkUsedSymbols(<string>opTemplate.opRC, symbolSetRC).indexOf('$imZ') >= 0) {
                templateConfig['OUTPUT_RC_COMPLEX'] = true;
            }
            if (this._checkUsedSymbols(<string>opTemplate.opCR, symbolSetCR).indexOf('$imZ') >= 0) {
                templateConfig['OUTPUT_CR_COMPLEX'] = true;
            }
            if (this._checkUsedSymbols(<string>opTemplate.opCC, symbolSetCC).indexOf('$imZ') >= 0) {
                templateConfig['OUTPUT_CC_COMPLEX'] = true;
            }
        }
        const blockMap = {
            '$InlineFunctions': this._flattenInlineFunctions(opConfig && opConfig.inlineFunctions ? opConfig.inlineFunctions : {}),
            '$SSBlock': this._compileSSBlock(opTemplate, templateConfig),
            '$STBlock': this._compileSTBlock(opTemplate, templateConfig),
            '$TSBlock': this._compileTSBlock(opTemplate, templateConfig),
            '$TTBlock': this._compileTTBlock(opTemplate, templateConfig)
        };
        return this._engine.generate(BIN_EL_OP_TEMPLATE, blockMap, templateConfig);
    }

    private _compileSSBlock(opTemplate: BinaryEWOpTemplate, templateConfig: {[key: string]: boolean}): string {
        const symbolMap = {
            '$reX': 'reXScalar',
            '$imX': 'imXScalar',
            '$reY': 'reYScalar',
            '$imY': 'imYScalar',
            '$reZ': 'reZScalar',
            '$imZ': 'imZScalar',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4'
        };
        const blockMap = {
            '$RRBlock': this._engine.generate(opTemplate.opRR, symbolMap, templateConfig),
            '$RCBlock': opTemplate.opRC != undefined
                ? this._engine.generate(opTemplate.opRC, symbolMap, templateConfig)
                : undefined,
            '$CRBlock': opTemplate.opCR != undefined
                ? this._engine.generate(opTemplate.opCR, symbolMap, templateConfig)
                : undefined,
            '$CCBlock': opTemplate.opCC != undefined
                ? this._engine.generate(opTemplate.opCC, symbolMap, templateConfig)
                : undefined
        };
        return this._engine.generate(SS_BLOCK_TEMPLATE, blockMap, templateConfig);
    }

    private _compileSTBlock(opTemplate: BinaryEWOpTemplate, templateConfig: {[key: string]: boolean}): string {
        const symbolMap = {
            '$reX': 'reXScalar',
            '$imX': 'imXScalar',
            '$reY': 'reY[i]',
            '$imY': 'imY[i]',
            '$reZ': 'reZ[i]',
            '$imZ': 'imZ[i]',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4'
        };
        const blockMap = {
            '$RRBlock': this._engine.generate(opTemplate.opRR, symbolMap, templateConfig),
            '$RCBlock': opTemplate.opRC != undefined
                ? this._engine.generate(opTemplate.opRC, symbolMap, templateConfig)
                : undefined,
            '$CRBlock': opTemplate.opCR != undefined
                ? this._engine.generate(opTemplate.opCR, symbolMap, templateConfig)
                : undefined,
            '$CCBlock': opTemplate.opCC != undefined
                ? this._engine.generate(opTemplate.opCC, symbolMap, templateConfig)
                : undefined
        };
        return this._engine.generate(ST_BLOCK_TEMPLATE, blockMap, templateConfig);
    }

    private _compileTSBlock(opTemplate: BinaryEWOpTemplate, templateConfig: {[key: string]: boolean}): string {
        const symbolMap = {
            '$reX': 'reX[i]',
            '$imX': 'imX[i]',
            '$reY': 'reYScalar',
            '$imY': 'imYScalar',
            '$reZ': 'reZ[i]',
            '$imZ': 'imZ[i]',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4'
        };
        const blockMap = {
            '$RRBlock': this._engine.generate(opTemplate.opRR, symbolMap, templateConfig),
            '$RCBlock': opTemplate.opRC != undefined
                ? this._engine.generate(opTemplate.opRC, symbolMap, templateConfig)
                : undefined,
            '$CRBlock': opTemplate.opCR != undefined
                ? this._engine.generate(opTemplate.opCR, symbolMap, templateConfig)
                : undefined,
            '$CCBlock': opTemplate.opCC != undefined
                ? this._engine.generate(opTemplate.opCC, symbolMap, templateConfig)
                : undefined
        };
        return this._engine.generate(TS_BLOCK_TEMPLATE, blockMap, templateConfig);
    }

    private _compileTTBlock(opTemplate: BinaryEWOpTemplate, templateConfig: {[key: string]: boolean}): string {
        const blockMap = {
            '$TTNormalBlock': this._compileTTNormalBlock(opTemplate, templateConfig),
            '$TTBroadcastBlock': this._compileTTBroadcastBlock(opTemplate, templateConfig)
        }
        return this._engine.generate(TT_BLOCK_TEMPLATE, blockMap, templateConfig);
    }

    private _compileTTNormalBlock(opTemplate: BinaryEWOpTemplate, templateConfig: {[key: string]: boolean}): string {
        const symbolMap = {
            '$reX': 'reX[i]',
            '$imX': 'imX[i]',
            '$reY': 'reY[i]',
            '$imY': 'imY[i]',
            '$reZ': 'reZ[i]',
            '$imZ': 'imZ[i]',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4'
        };
        const blockMap = {
            '$RRBlock': this._engine.generate(opTemplate.opRR, symbolMap, templateConfig),
            '$RCBlock': opTemplate.opRC != undefined
                ? this._engine.generate(opTemplate.opRC, symbolMap, templateConfig)
                : undefined,
            '$CRBlock': opTemplate.opCR != undefined
                ? this._engine.generate(opTemplate.opCR, symbolMap, templateConfig)
                : undefined,
            '$CCBlock': opTemplate.opCC != undefined
                ? this._engine.generate(opTemplate.opCC, symbolMap, templateConfig)
                : undefined
        }
        return this._engine.generate(TT_NORMAL_BLOCK_TEMPLATE, blockMap, templateConfig);
    }

    private _compileTTBroadcastBlock(opTemplate: BinaryEWOpTemplate, templateConfig: {[key: string]: boolean}): string {
        const symbolMapNormal = {
            '$reX': 'reX[offsetX + i]',
            '$imX': 'imX[offsetX + i]',
            '$reY': 'reY[offsetY + i]',
            '$imY': 'imY[offsetY + i]',
            '$reZ': 'reZ[offsetZ + i]',
            '$imZ': 'imZ[offsetZ + i]',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4'
        };
        const symbolMapFixX = {
            '$reX': 'reX[offsetX]',
            '$imX': 'imX[offsetX]',
            '$reY': 'reY[offsetY + i]',
            '$imY': 'imY[offsetY + i]',
            '$reZ': 'reZ[offsetZ + i]',
            '$imZ': 'imZ[offsetZ + i]',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4'
        };
        const symbolMapFixY = {
            '$reX': 'reX[offsetX + i]',
            '$imX': 'imX[offsetX + i]',
            '$reY': 'reY[offsetY]',
            '$imY': 'imY[offsetY]',
            '$reZ': 'reZ[offsetZ + i]',
            '$imZ': 'imZ[offsetZ + i]',
            '$tmp1': 'tmp1',
            '$tmp2': 'tmp2',
            '$tmp3': 'tmp3',
            '$tmp4': 'tmp4'
        };
        let blockMap: {[key: string]: string} = {};
        ['RR', 'RC', 'CR', 'CC'].forEach(s => {
            let opTemplateName = 'op' + s;
            if (opTemplate[opTemplateName] == undefined) {
                return;
            }
            let subBlockMap = {
                '$OpFixX': this._engine.generate(opTemplate[opTemplateName], symbolMapFixX, templateConfig),
                '$OpFixY': this._engine.generate(opTemplate[opTemplateName], symbolMapFixY, templateConfig),
                '$OpNormal': this._engine.generate(opTemplate[opTemplateName], symbolMapNormal, templateConfig)
            };
            blockMap['$' + s + 'Block'] = this._engine.generate(TT_BROADCAST_SUB_BLOCK_TEMPLATE, subBlockMap, templateConfig);
        });
        return this._engine.generate(TT_BROADCAST_BLOCK_TEMPLATE, blockMap, templateConfig);
    }

    /**
     * Checks the symbols used in the given code. Throws when encounters any
     * symbol that is not allowed. Returns a list of used symbols.
     * @param code Block of code to be checked.
     * @param allowed A list of allowed symbols. Case sensitive.
     */
    private _checkUsedSymbols(code: string, allowed: string[]): string[] {
        let reSymbol = /\$\w+/g;
        let m: RegExpExecArray | null;
        let used: {[key: string]: boolean} = {};
        while (m = reSymbol.exec(code)) {
            if (allowed.indexOf(m[0]) < 0) {
                throw new Error(`Symbol ${m[0]} is not permitted in the following code:\n${code}`);
            }
            used[m[0]] = true;
        }
        let result: string[] = [];
        for (let prop in used) {
            if (used.hasOwnProperty(prop)) {
                result.push(prop);
            }
        }
        return result;
    }

    private _flattenInlineFunctions(fs: {[key: string]: Function}): string {
        let result = '';
        for (let key in fs) {
            if (fs.hasOwnProperty(key)) {
                let fStr = fs[key].toString();
                if (fStr.indexOf('[native code]') > 0) {
                    throw new Error('Cannot inline native functions.');
                }
                // replace function name
                let idxFirstP = fStr.indexOf('(');
                if (idxFirstP < 0) {
                    throw new Error('Cannot find the first pair of parenthesis.');
                }
                // note that the specified function name is NOT checked
                fStr = 'function ' + key + fStr.substr(idxFirstP);
                result += fStr + '\n';
            }
        }
        return result;
    }

}