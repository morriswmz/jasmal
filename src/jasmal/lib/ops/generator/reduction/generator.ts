import { OpGeneratorBase } from '../generatorBase';
import { OpInput, OpOutput, RealOpOutput, RealOpOutputWithIndex } from '../../../commonTypes';
import { DType, OutputDTypeResolver, DTypeHelper } from '../../../core/dtype';
import { Tensor } from '../../../core/tensor';
import { ComplexNumber } from '../../../core/complexNumber';
import { ShapeHelper } from '../../../helper/shapeHelper';
import { DataHelper } from '../../../helper/dataHelper';
import { T_R_BLOCK_TEMPLATE, T_C_BLOCK_TEMPLATE, REDUCTION_OP_TEMPLATE,
         S_BLOCK_TEMPLATE, T_BLOCK_TEMPLATE } from './templates';
import { ObjectHelper } from '../../../helper/objHelper';

export type ReductionOp<TOut> = (x: OpInput, axis?: number, keepDims?: boolean) => TOut;

export type ReductionOpWithIndexOutput<TOut> = (x: OpInput, axis?: number, keepDims?: boolean) => TOut;

export interface ReductionOpDependencies {
    Tensor: Function;
    ComplexNumber: Function;
    DataHelper: Function;
    ShapeHelper: Function;
    DTypeHelper: Function;
    outputDTypeResolver: (t: DType, isComplex: boolean) => DType | undefined;
}

export interface ReductionOpConfig {
    outputDTypeResolver?: (t: DType, isComplex: boolean) => DType | undefined;
}

export type RIROReducer = (reX: ArrayLike<number>, offset: number, stride: number, n: number) => number;
export type CIROReducer = (reX: ArrayLike<number>, imX: ArrayLike<number>, offset: number, stride: number, n: number) => number;
export type CICOReducer = (reX: ArrayLike<number>, imX: ArrayLike<number>, offset: number, stride: number, n: number) => [number, number];

export type RIRIOReducer = (reX: ArrayLike<number>, offset: number, stride: number, n: number) => [number, number];
export type CIRIOReducer = (reX: ArrayLike<number>, imX: ArrayLike<number>, offset: number, stride: number, n: number) => [number, number];
export type CICIOReducer = (reX: ArrayLike<number>, imX: ArrayLike<number>,
                            offset: number, stride: number, n: number) => [number, number, number];

export class ReductionOpGenerator extends OpGeneratorBase {

    private static _instance: ReductionOpGenerator;

    public static getInstance(): ReductionOpGenerator {
        if (!ReductionOpGenerator._instance) {
            ReductionOpGenerator._instance = new ReductionOpGenerator();
        }
        return ReductionOpGenerator._instance;
    }

    public makeRealOnlyOp(fReal: RIROReducer, config: ReductionOpConfig = {}): ReductionOp<RealOpOutput> {
        let deps = this._getDependencies(config);
        let funcBody = this.generateOpFuncBody({
            NO_COMPLEX_INPUT: true,
            OUTPUT_INDICES: false,
            OUTPUT_R_COMPLEX: false,
            OUTPUT_C_COMPLEX: false
        }, ObjectHelper.properties(deps));
        let fn = new Function(this.DEP_OBJ_NAME, 'fReal', funcBody);
        return fn(deps, fReal);
    }

    public makeRealOnlyOpWithIndexOutput(fReal: RIRIOReducer,
                                         config?: ReductionOpConfig): ReductionOpWithIndexOutput<RealOpOutputWithIndex>
    {
        let deps = this._getDependencies(config);
        let funcBody = this.generateOpFuncBody({
            NO_COMPLEX_INPUT: true,
            OUTPUT_INDICES: true,
            OUTPUT_R_COMPLEX: false,
            OUTPUT_C_COMPLEX: false
        }, ObjectHelper.properties(deps));
        let fn = new Function(this.DEP_OBJ_NAME, 'fReal', funcBody);
        return fn(deps, fReal);
    }

    public makeOp(fReal: RIROReducer, fComplex: CICOReducer,
                  outputComplexWhenInputIsComplex: true,
                  config?: ReductionOpConfig): ReductionOp<OpOutput>;
    public makeOp(fReal: RIROReducer, fComplex: CIROReducer,
                  outputComplexWhenInputIsComplex: false,
                  config?: ReductionOpConfig): ReductionOp<RealOpOutput>
    public makeOp(fReal: RIROReducer, fComplex: CICOReducer | CIROReducer,
                  outputComplexWhenInputIsComplex: boolean,
                  config?: ReductionOpConfig): ReductionOp<OpOutput> {
        let deps = this._getDependencies(config);
        let funcBody = this.generateOpFuncBody({
            NO_COMPLEX_INPUT: false,
            OUTPUT_INDICES: false,
            OUTPUT_R_COMPLEX: false,
            OUTPUT_C_COMPLEX: outputComplexWhenInputIsComplex
        }, ObjectHelper.properties(deps));
        let fn = new Function(this.DEP_OBJ_NAME, 'fReal', 'fComplex', funcBody);
        return fn(deps, fReal, fComplex);
    }

    public generateOpFuncBody(config: {[key: string]: boolean}, depNames: string[]): string {
        let tBlockMap = {
            '$T_R_BLOCK': this._engine.generate(T_R_BLOCK_TEMPLATE, {}, config),
            '$T_C_BLOCK': config.NO_COMPLEX_INPUT ? '' : this._engine.generate(T_C_BLOCK_TEMPLATE, {}, config)
        }
        let mainBlockMap = {
            '$Dependencies': this._generateDependencyBlock(depNames),
            '$S_BLOCK': this._engine.generate(S_BLOCK_TEMPLATE, {}, config),
            '$T_BLOCK': this._engine.generate(T_BLOCK_TEMPLATE, tBlockMap, config)
        };
        return this._engine.generate(REDUCTION_OP_TEMPLATE, mainBlockMap, config);
    }

    private _getDependencies(config?: ReductionOpConfig): ReductionOpDependencies {
        return {
            Tensor: Tensor,
            ComplexNumber: ComplexNumber,
            ShapeHelper: ShapeHelper,
            DataHelper: DataHelper,
            DTypeHelper: DTypeHelper,
            outputDTypeResolver: config && config.outputDTypeResolver
                ? config.outputDTypeResolver
                : OutputDTypeResolver.uNoChange 
        };
    }
        
}
