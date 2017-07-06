import { DType } from './dtype';
import { Tensor } from './tensor';
import {
    IRandomOpProvider, IArithmeticOpProvider, IMathOpProvider,
    ICoreOpProvider, ILogicComparisonOpProvider, IMatrixOpProvider,
    MatrixModifier, IDataOpProvider
} from './ops/definition';
import { RandomOpProviderFactory } from './ops/random';
import { ArithmeticOpProviderFactory } from './ops/arithmetic';
import { MathOpProviderFactory } from './ops/math';
import { ComplexNumber } from './complexNumber';
import { CoreOpProviderFactory } from './ops/core';
import { LogicComparisonOpProviderFactory } from './ops/logicComp';
import { MatrixOpProviderFactory } from './ops/matrix';
import { DataOpProviderFactory } from './ops/data';
import { TypedArray } from './commonTypes';

export interface CustomProviders {
    
}

export interface JasmalOptions {
    backend?: any;
    customProviders?: CustomProviders;
}


export interface Jasmal extends ICoreOpProvider, IMatrixOpProvider,
    IRandomOpProvider, IArithmeticOpProvider, IMathOpProvider,
    ILogicComparisonOpProvider, IDataOpProvider {

    /**
     * Logic data type.
     */
    LOGIC: DType;
    INT32: DType;
    FLOAT64: DType;

    MM_NONE: MatrixModifier;
    MM_TRANSPOSED: MatrixModifier;
    MM_HERMITIAN: MatrixModifier;

    PI: number;
    J: ComplexNumber;

    complexNumber(re: number, im?: number): ComplexNumber;
    isComplexNumber(x: any): boolean;

    zeros(shape: number[], dtype?: DType): Tensor;
    ones(shape: number[], dtype?: DType): Tensor;
    fromArray(re: any[] | TypedArray, im?: any[] | TypedArray, dtype?: DType): Tensor;
    complex(x: Tensor, y: Tensor): Tensor;

}

export class JasmalEngine {
    
    public static createInstance(options?: JasmalOptions): Jasmal {

        let coreOpProvider = CoreOpProviderFactory.create();
        let randomOpProvider = RandomOpProviderFactory.create();
        let arithmeticOpProvider = ArithmeticOpProviderFactory.create();
        let matrixOpProvider = MatrixOpProviderFactory.create(arithmeticOpProvider);
        let mathOpProvider = MathOpProviderFactory.create();
        let logicCompOpProvider = LogicComparisonOpProviderFactory.create();
        let dataOpProvider = DataOpProviderFactory.create();
        

        return {
            LOGIC: DType.LOGIC,
            INT32: DType.INT32,
            FLOAT64: DType.FLOAT64,

            MM_NONE: MatrixModifier.None,
            MM_TRANSPOSED: MatrixModifier.Transposed,
            MM_HERMITIAN: MatrixModifier.Hermitian,

            J: new ComplexNumber(0, 1),
            PI: Math.PI,

            complexNumber: (re, im?) => new ComplexNumber(re, im),
            isComplexNumber: x => x instanceof ComplexNumber,
            zeros: Tensor.zeros,
            ones: Tensor.ones,
            fromArray: (re, im, dtype) => Tensor.fromArray(re, im, dtype),
            complex: (x, y) => Tensor.complex(x, y),

            reshape: coreOpProvider.reshape,
            flatten: coreOpProvider.flatten,
            squeeze: coreOpProvider.squeeze,
            vec: coreOpProvider.vec,
            concat: coreOpProvider.concat,
            tile: coreOpProvider.tile,
            prependAxis: coreOpProvider.prependAxis,
            appendAxis: coreOpProvider.appendAxis,
            linspace: coreOpProvider.linspace,
            logspace: coreOpProvider.logspace,
            real: coreOpProvider.real,
            imag: coreOpProvider.imag,
            isreal: coreOpProvider.isreal,
            isnan: coreOpProvider.isnan,
            isinf: coreOpProvider.isinf,
            find: coreOpProvider.find,

            eye: matrixOpProvider.eye,
            hilb: matrixOpProvider.hilb,
            diag: matrixOpProvider.diag,
            matmul: matrixOpProvider.matmul,
            kron: matrixOpProvider.kron,
            transpose: matrixOpProvider.transpose,
            hermitian: matrixOpProvider.hermitian,
            trace: matrixOpProvider.trace,
            inv: matrixOpProvider.inv,
            det: matrixOpProvider.det,
            norm: matrixOpProvider.norm,
            lu: matrixOpProvider.lu,
            svd: matrixOpProvider.svd,
            rank: matrixOpProvider.rank,

            eq: logicCompOpProvider.eq,
            neq: logicCompOpProvider.neq,
            gt: logicCompOpProvider.gt,
            ge: logicCompOpProvider.ge,
            lt: logicCompOpProvider.lt,
            le: logicCompOpProvider.le,
            and: logicCompOpProvider.and,
            or: logicCompOpProvider.or,
            xor: logicCompOpProvider.xor,
            not: logicCompOpProvider.not,
            all: logicCompOpProvider.all,
            any: logicCompOpProvider.any,

            add: arithmeticOpProvider.add,
            sub: arithmeticOpProvider.sub,
            neg: arithmeticOpProvider.neg,
            mul: arithmeticOpProvider.mul,
            div: arithmeticOpProvider.div,
            reciprocal: arithmeticOpProvider.reciprocal,

            abs: mathOpProvider.abs,
            sign: mathOpProvider.sign,
            min2: mathOpProvider.min2,
            max2: mathOpProvider.max2,
            conj: mathOpProvider.conj,
            angle: mathOpProvider.angle,
            sin: mathOpProvider.sin,
            cos: mathOpProvider.cos,
            tan: mathOpProvider.tan,
            cot: mathOpProvider.cot,
            asin: mathOpProvider.asin,
            acos: mathOpProvider.acos,
            atan: mathOpProvider.atan,
            sinh: mathOpProvider.sinh,
            cosh: mathOpProvider.cosh,
            tanh: mathOpProvider.tanh,
            sqrt: mathOpProvider.sqrt,
            exp: mathOpProvider.exp,
            pow2: mathOpProvider.pow2,
            log: mathOpProvider.log,
            floor: mathOpProvider.floor,
            ceil: mathOpProvider.ceil,
            round: mathOpProvider.round,
            rad2deg: mathOpProvider.rad2deg,
            deg2rad: mathOpProvider.deg2rad,

            seed: randomOpProvider.seed,
            rand: randomOpProvider.rand,
            randn: randomOpProvider.randn,
            randi: randomOpProvider.randi,

            min: dataOpProvider.min,
            max: dataOpProvider.max,
            sum: dataOpProvider.sum,
            prod: dataOpProvider.prod,
            cumsum: dataOpProvider.cumsum,
            mean: dataOpProvider.mean,
            median: dataOpProvider.median,
            std: dataOpProvider.std,
            var: dataOpProvider.var
        }
        
    }
}