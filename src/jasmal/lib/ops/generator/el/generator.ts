import { OpGeneratorBase } from '../generatorBase';
import { OpInputInternal, OpOutput, RealOpOutput } from '../../../commonTypes';
import { DType, OutputDTypeResolver, DTypeHelper } from '../../../core/dtype';
import { ShapeHelper } from '../../../helper/shapeHelper';
import { Tensor } from '../../../core/tensor';
import { ComplexNumber } from '../../../core/complexNumber';
import { CMath } from '../../../math/cmath';
import { UNARY_OP_TEMPLATE, S_BLOCK_TEMPLATE, T_BLOCK_TEMPLATE } from './unaryOpTemplates';
import { BIN_EL_OP_TEMPLATE, SS_BLOCK_TEMPLATE, ST_BLOCK_TEMPLATE,
         TS_BLOCK_TEMPLATE, TT_BLOCK_TEMPLATE, TT_NORMAL_BLOCK_TEMPLATE,
         TT_BROADCAST_SUB_BLOCK_TEMPLATE, TT_BROADCAST_BLOCK_TEMPLATE } from './binaryOpTemplates';
import { ObjectHelper } from '../../../helper/objHelper';

/**
 * General rules:
 * 1. If all the inputs are scalars, the output is scalar.
 * 2. If any of the input is an array/tensor, the output will be a tensor.
 */

/**
 * Represents a binary operation.
 */
export type GenericBinaryOp<TOut> = (x: OpInputInternal, y: OpInputInternal, inPlace?: boolean) => TOut;
/**
 * Represents a unary operation.
 */
export type GenericUnaryOp<TOut> = (x: OpInputInternal, inPlace?: boolean) => TOut;
/**
 * Represents a unary operation with a parameter.
 */
export type OneParamUnaryOp<TOut> = (x: OpInputInternal, p: number, inPlace?: boolean) => TOut;

/**
 * Defines the core operations.
 */
export interface BinaryOpTemplate {
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

export interface CommonOpConfig {
    noInPlaceOperation?: boolean;
    inlineFunctions?: {[key: string]: Function};
    extraDependencies?: {[key: string]: any};
}

export interface UnaryOpConfig extends CommonOpConfig {
    outputDTypeResolver?: (t: DType, isComplex: boolean) => DType | undefined;
}

export interface BinaryOpConfig extends CommonOpConfig {
    /**
     * Default value is OutputDTypeResolver.bWider.
     */
    outputDTypeResolver?: (t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean) => DType | undefined;
}

export interface UnaryOpTemplate {
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
    ShapeHelper: Function;
    DTypeHelper: Function;
    [key: string]: any;
}

interface BinaryOpDependencies extends OpCommonDependencies {
    outputDTypeResolver: (t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean) => DType | undefined;
}

interface UnaryOpDependencies extends OpCommonDependencies {
    outputDTypeResolver: (t: DType, isComplex: boolean) => DType | undefined;
}

export class ElementWiseOpGenerator extends OpGeneratorBase {
    
    private static _instance: ElementWiseOpGenerator;

    public static getInstance(): ElementWiseOpGenerator {
        if (!ElementWiseOpGenerator._instance) {
            ElementWiseOpGenerator._instance = new ElementWiseOpGenerator();
        }
        return ElementWiseOpGenerator._instance;
    }

    public makeUnaryOp(opTemplate: UnaryOpTemplate,
                       opConfig?: UnaryOpConfig): GenericUnaryOp<OpOutput> {
        let deps = this._getUnaryOpDependencies(opConfig);
        let funcBody = this.generateUnaryOpFuncBody(opTemplate, ObjectHelper.properties(deps), opConfig);
        let fn = (new Function(this.DEP_OBJ_NAME, funcBody))(deps);
        return fn;
    }

    public makeRealOutputUnaryOp(opTemplate: UnaryOpTemplate,
                                 opConfig?: UnaryOpConfig): GenericUnaryOp<RealOpOutput> {
        let deps = this._getUnaryOpDependencies(opConfig);
        let funcBody = this.generateUnaryOpFuncBody(opTemplate, ObjectHelper.properties(deps), opConfig, false, true);
        let fn = (new Function(this.DEP_OBJ_NAME, funcBody))(deps);
        return fn;
    }

    public makeOneParamUnaryOp(opTemplate: UnaryOpTemplate,
                               opConfig?: UnaryOpConfig): OneParamUnaryOp<OpOutput> {
        let deps = this._getUnaryOpDependencies(opConfig);
        let funcBody = this.generateUnaryOpFuncBody(opTemplate, ObjectHelper.properties(deps), opConfig, true);
        let fn = (new Function(this.DEP_OBJ_NAME, funcBody))(deps);
        return fn;
    }

    private _getUnaryOpDependencies(opConfig?: UnaryOpConfig): UnaryOpDependencies {
        let base = {
            Tensor: Tensor,
            ComplexNumber: ComplexNumber,
            CMath: CMath,
            ShapeHelper: ShapeHelper,
            DTypeHelper: DTypeHelper,
            outputDTypeResolver: opConfig && opConfig.outputDTypeResolver
                ? opConfig.outputDTypeResolver : OutputDTypeResolver.uNoChange,
        };
        return opConfig && opConfig.extraDependencies
            ? ObjectHelper.extend(base, opConfig.extraDependencies)
            : base;
    }

    public generateUnaryOpFuncBody(opTemplate: UnaryOpTemplate,
                                   depNames: string[],
                                   opConfig?: UnaryOpConfig,
                                   hasParam?: boolean,
                                   forceRealOutput?: boolean): string {
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
        if (opRSymbols.indexOf('$imY') >= 0) {
            templateConfig['OUTPUT_R_COMPLEX'] = true;
        }
        if (opCSymbols.indexOf('$imY') >= 0) {
            templateConfig['OUTPUT_C_COMPLEX'] = true;
        }
        if (forceRealOutput) {
            if (templateConfig['OUTPUT_R_COMPLEX'] || templateConfig['OUTPUT_C_COMPLEX']) {
                throw new Error('Specified templates generates complex outputs when only real outputs are allowed.');
            }
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
            '$Dependencies': this._generateDependencyBlock(depNames),
            '$TBlock': this._engine.generate(T_BLOCK_TEMPLATE, blockMapTensor, templateConfig),
            '$SBlock': this._engine.generate(S_BLOCK_TEMPLATE, blockMapScalar, templateConfig)
        }
        return this._engine.generate(UNARY_OP_TEMPLATE, blockMap, templateConfig);
    }

    public makeBinaryOp(opTemplate: BinaryOpTemplate,
                        config?: BinaryOpConfig): GenericBinaryOp<OpOutput> {
        let deps = this._getBinaryOpDependencies(config);
        let funcBody = this.generateBinaryOpFuncBody(opTemplate, ObjectHelper.properties(deps), config);
        let fn = (new Function(this.DEP_OBJ_NAME, funcBody))(deps);
        return fn;
    }

    public makeRealOutputBinaryOp(opTemplate: BinaryOpTemplate,
                                  config?: BinaryOpConfig): GenericBinaryOp<RealOpOutput> {
        let deps = this._getBinaryOpDependencies(config);
        let funcBody = this.generateBinaryOpFuncBody(opTemplate, ObjectHelper.properties(deps), config, true);
        let fn = (new Function(this.DEP_OBJ_NAME, funcBody))(deps);
        return fn;
    }

    private _getBinaryOpDependencies(config?: BinaryOpConfig): BinaryOpDependencies {
        let outputDTypeResolver = (config && config.outputDTypeResolver)
            ? config.outputDTypeResolver
            : OutputDTypeResolver.bWider;
        let base: BinaryOpDependencies = {
            Tensor: Tensor,
            ComplexNumber: ComplexNumber,
            CMath: CMath,
            ShapeHelper: ShapeHelper,
            DTypeHelper: DTypeHelper,
            outputDTypeResolver: outputDTypeResolver,
        };
        return config && config.extraDependencies
            ? ObjectHelper.extend(base, config.extraDependencies)
            : base;
    }

    public generateBinaryOpFuncBody(opTemplate: BinaryOpTemplate, depNames: string[],
                                    opConfig?: BinaryOpConfig, forceRealOutput?: boolean): string {
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
        if (forceRealOutput) {
            if (templateConfig['OUTPUT_RR_COMPLEX'] || templateConfig['OUTPUT_RC_COMPLEX'] ||
                templateConfig['OUTPUT_CR_COMPLEX'] || templateConfig['OUTPUT_CC_COMPLEX'])
            {
                throw new Error('Specified templates generates complex outputs when only real outputs are allowed.');
            }
        }
        const blockMap = {
            '$InlineFunctions': this._flattenInlineFunctions(opConfig && opConfig.inlineFunctions ? opConfig.inlineFunctions : {}),
            '$Dependencies': this._generateDependencyBlock(depNames),
            '$SSBlock': this._compileSSBlock(opTemplate, templateConfig),
            '$STBlock': this._compileSTBlock(opTemplate, templateConfig),
            '$TSBlock': this._compileTSBlock(opTemplate, templateConfig),
            '$TTBlock': this._compileTTBlock(opTemplate, templateConfig)
        };
        return this._engine.generate(BIN_EL_OP_TEMPLATE, blockMap, templateConfig);
    }

    private _compileSSBlock(opTemplate: BinaryOpTemplate, templateConfig: {[key: string]: boolean}): string {
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

    private _compileSTBlock(opTemplate: BinaryOpTemplate, templateConfig: {[key: string]: boolean}): string {
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

    private _compileTSBlock(opTemplate: BinaryOpTemplate, templateConfig: {[key: string]: boolean}): string {
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

    private _compileTTBlock(opTemplate: BinaryOpTemplate, templateConfig: {[key: string]: boolean}): string {
        const blockMap = {
            '$TTNormalBlock': this._compileTTNormalBlock(opTemplate, templateConfig),
            '$TTBroadcastBlock': this._compileTTBroadcastBlock(opTemplate, templateConfig)
        }
        return this._engine.generate(TT_BLOCK_TEMPLATE, blockMap, templateConfig);
    }

    private _compileTTNormalBlock(opTemplate: BinaryOpTemplate, templateConfig: {[key: string]: boolean}): string {
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

    private _compileTTBroadcastBlock(opTemplate: BinaryOpTemplate, templateConfig: {[key: string]: boolean}): string {
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

}
