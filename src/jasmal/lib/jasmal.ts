import { DType } from './core/dtype';
import { Tensor } from './core/tensor';
import { TypedArray } from './commonTypes';
import { IRandomEngine } from './ops/random/engine';
import { ICoreOpProvider } from './ops/core/definition';
import { IArithmeticOpProvider } from './ops/arithmetic/definition';
import { IRandomOpProvider } from './ops/random/definition';
import { IMatrixOpProvider } from './ops/matrix/definition';
import { IMathOpProvider } from './ops/math/definition';
import { ILogicComparisonOpProvider } from './ops/logicComp/definition';
import { IBinaryOpProvider } from './ops/binary/definition';
import { IDataOpProvider } from './ops/data/definition';
import { IPolynomialOpProvider } from './ops/poly/definition';
import { ISetOpProvider } from './ops/set/definition';
import { RandomOpProviderFactory } from './ops/random';
import { ArithmeticOpProviderFactory } from './ops/arithmetic';
import { MathOpProviderFactory } from './ops/math';
import { ComplexNumber } from './core/complexNumber';
import { CoreOpProviderFactory } from './ops/core';
import { LogicComparisonOpProviderFactory } from './ops/logicComp';
import { BinaryOpProviderFactory } from './ops/binary/index';
import { MatrixOpProviderFactory } from './ops/matrix';
import { DataOpProviderFactory } from './ops/data';
import { PolynomialOpProviderFactory } from './ops/poly';
import { SetOpProviderFactory } from './ops/set/index';
import { ObjectHelper } from './helper/objHelper';
import { ElementWiseOpGenerator, ReductionOpGenerator } from './ops/generator';
import { EPSILON } from './constant';
import { IBlaoBackend, ISpecialLinearSystemSolverBackend, ILUBackend, IQRBackend,
         ICholeskyBackend, ISvdBackend, IEigenBackend } from './linalg/backend';
import { MatrixModifier } from './linalg/modifiers';

export interface JasmalOptions {
    rngEngine?: string | IRandomEngine;
    linalg?: LinalgOptions;
    providers?: ProviderOptions;
}

export interface LinalgOptions {
    blao?: IBlaoBackend;
    lu?: ILUBackend;
    qr?: IQRBackend;
    chol?: ICholeskyBackend;
    svd?: ISvdBackend;
    eigen?: IEigenBackend;
    linsolve?: ISpecialLinearSystemSolverBackend;
}

export interface ProviderOptions {
    core?: ICoreOpProvider;
    random?: IRandomOpProvider;
    arithmetic?: IArithmeticOpProvider;
    math?: IMathOpProvider;
    matrix?: IMatrixOpProvider;
    logic?: ILogicComparisonOpProvider;
    binary?: IBinaryOpProvider;
    data?: IDataOpProvider;
    polynomial?: IPolynomialOpProvider;
    set?: ISetOpProvider;
}

export interface IJasmalModuleFactory<M> {

    /**
     * Creates a module object whose members will be copied to the final Jasmal
     * instance. These members should not depend on `this`.
     */
    create(options: JasmalOptions): M;

}

export interface JasmalBase {
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
     * Machine precision for double.
     */
    readonly EPSILON: number;

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

/**
 * JASMAL is exported as an interface.
 * This means you can replace any of its functions you want to.
 */
export interface Jasmal extends JasmalBase, ICoreOpProvider, IMatrixOpProvider,
    IRandomOpProvider, IArithmeticOpProvider, IMathOpProvider,
    ILogicComparisonOpProvider, IBinaryOpProvider, IDataOpProvider,
    IPolynomialOpProvider, ISetOpProvider {}

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

        const elementWiseOpGen = ElementWiseOpGenerator.getInstance();
        const reductionOpGen = ReductionOpGenerator.getInstance();

        const customProviders = options.providers;

        const coreOpProvider = customProviders && customProviders.core
            ? customProviders.core
            : (new CoreOpProviderFactory(elementWiseOpGen)).create(options);
        const randomOpProvider = customProviders && customProviders.random
            ? customProviders.random
            : (new RandomOpProviderFactory()).create(options);
        const arithmeticOpProvider = customProviders && customProviders.arithmetic
            ? customProviders.arithmetic
            : (new ArithmeticOpProviderFactory(elementWiseOpGen)).create(options);
        const mathOpProvider = customProviders && customProviders.math
            ? customProviders.math
            : (new MathOpProviderFactory(elementWiseOpGen)).create(options);
        const matrixOpProvider = customProviders && customProviders.matrix
            ? customProviders.matrix
            : (new MatrixOpProviderFactory(arithmeticOpProvider, mathOpProvider)).create(options);
        const logicCompOpProvider = customProviders && customProviders.logic
            ? customProviders.logic
            : (new LogicComparisonOpProviderFactory(elementWiseOpGen)).create(options);
        const binaryOpProvider = customProviders && customProviders.binary
            ? customProviders.binary
            : (new BinaryOpProviderFactory(elementWiseOpGen)).create(options);
        const dataOpProvider = customProviders && customProviders.data
            ? customProviders.data
            : (new DataOpProviderFactory(coreOpProvider, arithmeticOpProvider,
                mathOpProvider, matrixOpProvider, reductionOpGen)).create(options);
        const polyOpProvider = customProviders && customProviders.polynomial
            ? customProviders.polynomial
            : (new PolynomialOpProviderFactory(coreOpProvider, matrixOpProvider)).create(options);
        const setOpProvider = customProviders && customProviders.set
            ? customProviders.set
            : (new SetOpProviderFactory(coreOpProvider, logicCompOpProvider)).create(options);
        
        let jasmalCore: JasmalBase =  {
            LOGIC: DType.LOGIC,
            INT32: DType.INT32,
            FLOAT64: DType.FLOAT64,

            MM_NONE: MatrixModifier.None,
            MM_TRANSPOSED: MatrixModifier.Transposed,
            MM_HERMITIAN: MatrixModifier.Hermitian,

            J: new ComplexNumber(0, 1),
            PI: Math.PI,
            EPSILON: EPSILON,

            complexNumber: (re, im?) => new ComplexNumber(re, im || 0),
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
            .extend(binaryOpProvider)
            .extend(dataOpProvider)
            .extend(polyOpProvider)
            .extend(setOpProvider)
            .end();
        
    }
}
