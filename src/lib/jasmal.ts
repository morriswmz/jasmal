import { DType } from './dtype';
import { Tensor } from './tensor';
import { TypedArray } from './commonTypes';
import { IRandomEngine } from './ops/random/engine';
import { ICoreOpProvider } from './ops/core/definition';
import { IArithmeticOpProvider } from './ops/arithmetic/definition';
import { IRandomOpProvider } from './ops/random/definition';
import { MatrixModifier, IMatrixOpProvider } from './ops/matrix/definition';
import { IMathOpProvider } from './ops/math/definition';
import { ILogicComparisonOpProvider } from './ops/logicComp/definition';
import { IDataOpProvider } from './ops/data/definition';
import { RandomOpProviderFactory } from './ops/random';
import { ArithmeticOpProviderFactory } from './ops/arithmetic';
import { MathOpProviderFactory } from './ops/math';
import { ComplexNumber } from './complexNumber';
import { CoreOpProviderFactory } from './ops/core';
import { LogicComparisonOpProviderFactory } from './ops/logicComp';
import { MatrixOpProviderFactory } from './ops/matrix';
import { DataOpProviderFactory } from './ops/data';
import { ObjectHelper } from './helper/objHelper';
import { ElementWiseOpGenerator, ReductionOpGenerator } from './ops/generator';

export interface JasmalOptions {
    rngEngine?: string | IRandomEngine;
}

export interface Jasmal extends ICoreOpProvider, IMatrixOpProvider,
    IRandomOpProvider, IArithmeticOpProvider, IMathOpProvider,
    ILogicComparisonOpProvider, IDataOpProvider {

    /**
     * Logic data type.
     */
    readonly LOGIC: DType;
    readonly INT32: DType;
    readonly FLOAT64: DType;

    readonly MM_NONE: MatrixModifier;
    readonly MM_TRANSPOSED: MatrixModifier;
    readonly MM_HERMITIAN: MatrixModifier;

    readonly PI: number;
    readonly J: ComplexNumber;

    complexNumber(re: number, im?: number): ComplexNumber;
    isComplexNumber(x: any): boolean;
    isTensor(x: any): boolean;

    zeros(shape: number[], dtype?: DType): Tensor;
    ones(shape: number[], dtype?: DType): Tensor;
    fromArray(re: any[] | TypedArray, im?: any[] | TypedArray, dtype?: DType): Tensor;
    complex(x: Tensor, y: Tensor): Tensor;

}

export class JasmalEngine {

    public static getDefaultOptions(): JasmalOptions {
        return {
            rngEngine: 'twister'
        };
    }
    
    public static createInstance(options?: JasmalOptions): Jasmal {

        let defaultOptions = JasmalEngine.getDefaultOptions();
        if (options) {
            for (let prop in options) {
                if (options.hasOwnProperty(prop)) {
                    defaultOptions[prop] = options[prop];
                }
            }
            options = defaultOptions;
        } else {
            options = defaultOptions;
        }

        let elementWiseOpGen = ElementWiseOpGenerator.getInstance();
        let reductionOpGen = ReductionOpGenerator.getInstance();

        let coreOpProvider = CoreOpProviderFactory.create(elementWiseOpGen);
        let randomOpProvider = RandomOpProviderFactory.create(options.rngEngine);
        let arithmeticOpProvider = ArithmeticOpProviderFactory.create(elementWiseOpGen);
        let matrixOpProvider = MatrixOpProviderFactory.create(arithmeticOpProvider);
        let mathOpProvider = MathOpProviderFactory.create(elementWiseOpGen);
        let logicCompOpProvider = LogicComparisonOpProviderFactory.create(elementWiseOpGen);
        let dataOpProvider = DataOpProviderFactory.create(coreOpProvider, reductionOpGen);
        
        let jasmalCore =  {
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
            isTensor: x => x instanceof Tensor,
            zeros: Tensor.zeros,
            ones: Tensor.ones,
            fromArray: (re, im, dtype) => Tensor.fromArray(re, im, dtype),
            complex: (x, y) => Tensor.complex(x, y)
        };

        return ObjectHelper.createExtendChain(jasmalCore)
            .extend(coreOpProvider)
            .extend(randomOpProvider)
            .extend(arithmeticOpProvider)
            .extend(matrixOpProvider)
            .extend(mathOpProvider)
            .extend(logicCompOpProvider)
            .extend(dataOpProvider)
            .end();
        
    }
}