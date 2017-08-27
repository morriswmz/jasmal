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
import { IPolynomialOpProvider } from './ops/poly/definition';
import { RandomOpProviderFactory } from './ops/random';
import { ArithmeticOpProviderFactory } from './ops/arithmetic';
import { MathOpProviderFactory } from './ops/math';
import { ComplexNumber } from './complexNumber';
import { CoreOpProviderFactory } from './ops/core';
import { LogicComparisonOpProviderFactory } from './ops/logicComp';
import { MatrixOpProviderFactory } from './ops/matrix';
import { DataOpProviderFactory } from './ops/data';
import { PolynomialOpProviderFactory } from './ops/poly';
import { ObjectHelper } from './helper/objHelper';
import { ElementWiseOpGenerator, ReductionOpGenerator } from './ops/generator';

export interface JasmalOptions {
    rngEngine?: string | IRandomEngine;
}

/**
 * JASMAL is exported as an interface.
 * This means you can replace any of its functions I you want to.
 */
export interface Jasmal extends ICoreOpProvider, IMatrixOpProvider,
    IRandomOpProvider, IArithmeticOpProvider, IMathOpProvider,
    ILogicComparisonOpProvider, IDataOpProvider, IPolynomialOpProvider {

    /**
     * Logic data type.
     */
    readonly LOGIC: DType;
    /**
     * 32-bit signed integer type.
     */
    readonly INT32: DType;
    /**
     * Double type.
     */
    readonly FLOAT64: DType;

    /**
     * No modifier will be applied.
     */
    readonly MM_NONE: MatrixModifier;
    /**
     * Matrix should be transposed before performing the operation.
     */
    readonly MM_TRANSPOSED: MatrixModifier;
    /**
     * Matrix should be Hermitian transposed before performing the operation.
     */
    readonly MM_HERMITIAN: MatrixModifier;

    /**
     * Same as `Math.PI`.
     */
    readonly PI: number;
    /**
     * Imaginary unit.
     */
    readonly J: ComplexNumber;

    /**
     * Creates a complex number.
     */
    complexNumber(re: number, im?: number): ComplexNumber;
    /**
     * Checks if the input is an instance of ComplexNumber.
     */
    isComplexNumber(x: any): boolean;
    /**
     * Checks if the input is an instance of Tensor.
     */
    isTensor(x: any): boolean;

    /**
     * Creates a tensor of the specified shape filled with zeros.
     * @param shape The shape of the tensor.
     * @param dtype (Optional) Data type. Default value is FLOAT64.
     */
    zeros(shape: ArrayLike<number>, dtype?: DType): Tensor;
    /**
     * Creates a tensor of the specified shape filled with ones.
     * @param shape The shape of the tensor.
     * @param dtype (Optional) Data type. Default value is FLOAT64.
     */
    ones(shape: ArrayLike<number>, dtype?: DType): Tensor;
    /**
     * Creates a tensor from JavaScript arrays (can be nested arrays or typed
     * arrays).
     * @param re Real part.
     * @param im (Optional) Imaginary part. Set this to [] if there is no
     *           imaginary part. Otherwise its structure must match that of the
     *           real part. Default value is [].
     * @param dtype (Optional) Data type. Default value is FLOAT64. If the data
     *              type is set to LOGIC, `im` must be set to [].
     */
    fromArray(re: any[] | TypedArray, im?: any[] | TypedArray, dtype?: DType): Tensor;
    /**
     * Combines two real tensors of the same shape into a complex tensor.
     * @param re Real part.
     * @param im Imaginary part.
     */
    complex(re: Tensor, im: Tensor): Tensor;

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
        let polyOpProvider = PolynomialOpProviderFactory.create(coreOpProvider, matrixOpProvider);
        
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
            .extend(polyOpProvider)
            .end();
        
    }
}
