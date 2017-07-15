import { IMatrixOpProvider, IArithmeticOpProvider, IMathOpProvider, MatrixModifier } from '../definition';
import { Tensor } from '../../tensor';
import { DataBlock } from '../../storage';
import { DType } from '../../dtype';
import { OpInput, OpOutput, Scalar } from '../../commonTypes';
import { IMatrixBasicSubroutines, BuiltInMBS } from './mbs';
import { ComplexNumber } from '../../complexNumber';
import { LU } from './decomp/lu';
import { SVD } from './decomp/svd';
import { DataHelper } from '../../helper/dataHelper';
import { EPSILON } from '../../constant';
import { NormFunction } from './norm';
import { Eigen } from "./decomp/eigen";

export class MatrixOpProviderFactory {

    public static create(arithmOp: IArithmeticOpProvider, mbs: IMatrixBasicSubroutines = new BuiltInMBS()): IMatrixOpProvider {

        const opEye = (m: number, n?: number, dtype: DType = DType.FLOAT64) => {
            if (n == undefined) n = m;
            let X = Tensor.zeros([m, n], dtype);
            let l = Math.min(n, m);
            for (let i = 0;i < l;i++) {
                X.realData[i * n + i] = 1;
            }
            return X;
        };

        const opHilb = (n: number) => {
            let m = Tensor.zeros([n, n]);
            for (let i = 0;i < n;i++) {
                for (let j = 0;j < n;j++) {
                    m.realData[i * n + j] = 1 / (i + j + 1);
                }
            }
            return m;
        };

        const opDiag = (x: OpInput): Tensor => {
            let X = Tensor.analyzeOpInput(x);
            if (X.hasOnlyOneElement) {
                return Tensor.scalar(X.re, X.im);
            } else {
                let Y: Tensor;
                let reX = X.reArr;
                let imX = X.imArr;
                let reY: DataBlock, imY: DataBlock;
                let n: number;
                if (X.originalShape.length === 1) {
                    // vector -> diagonal matrix
                    n = reX.length;
                    Y = Tensor.zeros([reX.length, reX.length], X.originalDType);
                    reY = Y.realData;
                    if (X.isComplex) {
                        Y.ensureComplexStorage();
                        imY = Y.imagData;
                        for (let i = 0;i < n;i++) {
                            reY[i * n + i] = reX[i];
                            imY[i * n + i] = imX[i];
                        }
                    } else {
                        for (let i = 0;i < n;i++) {
                            reY[i * n + i] = reX[i];
                        }
                    }
                } else if (X.originalShape.length === 2) {
                    // extra matrix diagonals
                    n = Math.min(X.originalShape[0], X.originalShape[1]);
                    Y = Tensor.zeros([n], X.originalDType);
                    reY = Y.realData;
                    if (X.isComplex) {
                        Y.ensureComplexStorage();
                        imY = Y.imagData;
                        for (let i = 0;i < n;i++) {
                            reY[i] = reX[i * X.originalShape[1] + i];
                            imY[i] = imX[i * X.originalShape[1] + i];
                        }
                    } else {
                        for (let i = 0;i < n;i++) {
                            reY[i] = reX[i * X.originalShape[1] + i];
                        }
                    }
                } else {
                    throw new Error('Matrix or vector expected.')
                }
                return Y;
            }
        };

        const isSymmetric = (m: number, n: number, x: ArrayLike<number>, skew: boolean): boolean => {
            if (skew) {
                for (let i = 0;i < m;i++) {
                    for (let j = 0;j < n;j++) {
                        if (x[i * n + j] !== -x[j * m + i]) {
                            return false;
                        }
                    }
                }
            } else {
                for (let i = 0;i < m;i++) {
                    for (let j = 0;j < n;j++) {
                        if (x[i * n + j] !== x[j * m + i]) {
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        const opIsSymmetric = (x: OpInput, skew: boolean = false): boolean => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.ndim !== 2) {
                throw new Error('Matrix expected.');
            }
            let [m, n] = X.shape;
            if (X.hasComplexStorage()) {
                return isSymmetric(m, n, X.realData, skew) && isSymmetric(m, n, X.imagData, skew);
            } else {
                return isSymmetric(m, n, X.realData, skew);
            }
        };

        const opIsHermitian = (x: OpInput, skew: boolean = false): boolean => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.ndim !== 2) {
                throw new Error('Matrix expected.');
            }
            let [m, n] = X.shape;
            if (X.hasComplexStorage()) {
                return isSymmetric(m, n, X.realData, skew) && isSymmetric(m, n, X.imagData, !skew);
            } else {
                return isSymmetric(m, n, X.realData, skew);
            }
        };

        const matMulOutputTypeResolver = (t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean): DType | undefined => {
            // integer stays integer, otherwise promote to floats
            switch (t1) {
                case DType.LOGIC:
                case DType.INT32:
                    switch (t2) {
                        case DType.LOGIC:
                        case DType.INT32:
                            return DType.INT32;
                        case DType.FLOAT64:
                            return DType.FLOAT64;
                    }
                case DType.FLOAT64:
                    return DType.FLOAT64;
            }
            return undefined;
        };

        const opMatMul = (x: OpInput, y: OpInput, yModifier: MatrixModifier = MatrixModifier.None): OpOutput => {
            if (x instanceof ComplexNumber || typeof x === 'number' ||
                y instanceof ComplexNumber || typeof y === 'number') {
                return arithmOp.mul(x, y);
            }
            let vx = Tensor.analyzeOpInput(x);
            let vy = Tensor.analyzeOpInput(y);
            if (vx.originalShape.length > 2 || vy.originalShape.length > 2) {
                throw new Error('Matrix or vector expected.');
            }
            let m = vx.originalShape.length === 1 ? 1 : vx.originalShape[0];
            let n1 = vx.originalShape.length === 1 ? vx.originalShape[0] : vx.originalShape[1];
            let n2 = vy.originalShape.length === 1 ? 1 : vy.originalShape[0];
            let p = vy.originalShape.length === 1 ? vy.originalShape[0] : vy.originalShape[1];
            if (yModifier > 0) {
                let tmp = n2;
                n2 = p;
                p = tmp;
            }
            if (n1 !== n2) {
                throw new Error(`Matrix dimensions (${m}, ${n1}) and (${n2}, ${p}) are not compatible.`);
            }
            let dims: [number, number, number] = [m, n1, p];
            let Z = Tensor.zeros([m, p],
                matMulOutputTypeResolver(vx.originalDType, vx.isComplex, vy.originalDType, vy.isComplex));
            if (vx.isComplex) {
                Z.ensureComplexStorage();
                if (vy.isComplex) {
                    mbs.cmmul(dims, yModifier, vx.reArr, vx.imArr,
                        vy.reArr, vy.imArr, Z.realData, Z.imagData);
                } else {
                    mbs.mmul(dims, yModifier, vx.reArr, vy.reArr, Z.realData);
                    mbs.mmul(dims, yModifier, vx.imArr, vy.reArr, Z.imagData);
                }
            } else {
                if (vy.isComplex) {
                    Z.ensureComplexStorage();
                    if (yModifier === MatrixModifier.Hermitian) {
                        mbs.mmul(dims, MatrixModifier.Transposed, vx.reArr, vy.reArr, Z.realData);
                        mbs.mmul(dims, MatrixModifier.Transposed, vx.reArr, vy.imArr, Z.imagData);
                        let reZ = Z.imagData;
                        for (let i = 0;i < reZ.length;i++) {
                            reZ[i] = -reZ[i];
                        }
                    } else {
                        mbs.mmul(dims, yModifier, vx.reArr, vy.reArr, Z.realData);
                        mbs.mmul(dims, yModifier, vx.reArr, vy.imArr, Z.imagData);  
                    }  
                } else {
                    mbs.mmul(dims, yModifier, vx.reArr, vy.reArr, Z.realData);   
                }
            }
            return Z;
        };

        const opKron = (x: OpInput, y: OpInput): Tensor => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let Y = y instanceof Tensor ? y : Tensor.toTensor(y);
            if (X.ndim > 2 || Y.ndim > 2) {
                throw new Error('Kronecker produce for ndim > 2 is not supported.');
            }
            if (X.ndim === 1 && Y.ndim === 1) {
                return (<Tensor>arithmOp.mul(X.getReshapedCopy([-1, 1]), Y)).reshape([-1]);
            }
            // see https://www.mathworks.com/matlabcentral/fileexchange/24499-kronecker
            let shapeX = X.shape;
            let shapeY = Y.shape;
            let A = X.getReshapedCopy([shapeX[0], 1, shapeX[1], 1]);
            let B = Y.getReshapedCopy([1, shapeY[0], 1, shapeY[1]]);
            return (<Tensor>arithmOp.mul(A, B)).reshape([shapeX[0] * shapeY[0], shapeX[1] * shapeY[1]]);
        };

        const opTranspose = (x: OpInput): Tensor => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let shapeX = X.shape;
            if (shapeX.length === 1) {
                // we treat 1D vector as a row vector
                return X.getReshapedCopy([-1, 1]);
            } else if (shapeX.length === 2) {
                let Y = Tensor.zeros([shapeX[1], shapeX[0]], X.dtype);
                mbs.transpose(<[number, number]>shapeX, X.realData, Y.realData);
                if (X.hasComplexStorage()) {
                    Y.ensureComplexStorage();
                    mbs.transpose(<[number, number]>shapeX, X.imagData, Y.imagData);
                }
                return Y;
            } else {
                throw new Error('Matrix expected.');
            }
        };

        const opHermitian = (x: OpInput): Tensor => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let shapeX = X.shape;
            let Y: Tensor;
            if (shapeX.length === 1) {
                // we treat 1D vector as a row vector
                Y = X.getReshapedCopy([-1, 1]);
                if (Y.hasComplexStorage()) {
                    Y.ensureUnsharedLocalStorage();
                    let im = Y.imagData;
                    for (let i = 0;i < im.length;i++) {
                        im[i] = -im[i];
                    }
                }
                return Y;
            } else if (shapeX.length === 2) {
                let Y = Tensor.zeros([shapeX[1], shapeX[0]], X.dtype);
                if (X.hasComplexStorage()) {
                    Y.ensureComplexStorage();
                    mbs.hermitian(<[number, number]>shapeX, X.realData, X.imagData, Y.realData, Y.imagData);
                } else {
                    mbs.transpose(<[number, number]>shapeX, X.realData, Y.realData);
                }
                return Y;
            } else {
                throw new Error('Matrix expected.');
            }
        };

        const opTrace = (x: OpInput): Scalar => {
            let v = Tensor.analyzeOpInput(x);
            if (v.hasOnlyOneElement) {
                return v.isComplex ? new ComplexNumber(v.re, v.im) : v.re;
            } else {
                if (v.originalShape.length !== 2 || v.originalShape[0] !== v.originalShape[1]) {
                    throw new Error('Square matrix expected.');
                }
                let re = v.reArr;
                let im = v.reArr;
                let accRe = 0, accIm = 0;
                for (let i = 0;i < re.length;i += v.originalShape[0]) {
                    accRe += re[i];
                }
                if (v.isComplex) {
                    for (let i = 0;i < im.length;i += v.originalShape[0]) {
                        accIm += im[i];
                    }
                }
                return accIm === 0 ? accRe : new ComplexNumber(accRe, accIm);
            }
        };

        const opNorm = (x: OpInput, p: number | string): number => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.ndim > 2) {
                throw new Error('Norm only works for vectors and matrices.');
            }
            if (X.ndim === 1 || (X.ndim === 2 && X.shape[1] === 1)) {
                // vector norm
                if (typeof(p) !== 'number') {
                    throw new Error('For vectors, p must be nonnegative.');
                }
                switch (p) {
                    case 0:
                        return X.hasComplexStorage()
                            ? NormFunction.cvec0Norm(X.realData, X.imagData)
                            : NormFunction.vec0Norm(X.realData);
                    case 2:
                        return X.hasComplexStorage()
                            ? NormFunction.cvec2Norm(X.realData, X.imagData)
                            : NormFunction.vec2Norm(X.realData);
                    case Infinity:
                        return X.hasComplexStorage()
                            ? NormFunction.cvecInfNorm(X.realData, X.imagData)
                            : NormFunction.vecInfNorm(X.realData);
                    default:
                        return X.hasComplexStorage()
                            ? NormFunction.cvecPNorm(X.realData, X.imagData, p)
                            : NormFunction.vecPNorm(X.realData, p);
                }
            } else {
                // matrix norm
                let shape = X.shape;
                if (typeof(p) === 'string') {
                    if (p.toLowerCase() !== 'fro') {
                        throw new Error('Expecting "fro".');
                    }
                    return X.hasComplexStorage()
                        ? NormFunction.cvec2Norm(X.realData, X.imagData)
                        : NormFunction.vec2Norm(X.realData);
                } else {
                    switch (p) {
                        case 1:
                            return X.hasComplexStorage()
                                ? NormFunction.cmat1Norm(shape[0], shape[1], X.realData, X.imagData)
                                : NormFunction.mat1Norm(shape[0], shape[1], X.realData);
                        case 2:
                            return X.hasComplexStorage()
                                ? NormFunction.cmat2Norm(shape[0], shape[1], X.realData, X.imagData)
                                : NormFunction.mat2Norm(shape[0], shape[1], X.realData);
                        case Infinity:
                            return X.hasComplexStorage()
                                ? NormFunction.cmatInfNorm(shape[0], shape[1], X.realData, X.imagData)
                                : NormFunction.matInfNorm(shape[0], shape[1], X.realData);
                        default:
                            throw new Error('Only 1, 2, and infinity norm are supported for matrices.')
                    }
                }
            }
        };

        function doCompactLU(x: OpInput): [Tensor, number[], number] {
            let X = x instanceof Tensor ? x.asType(DType.FLOAT64) : Tensor.toTensor(x);            
            let shapeX = X.shape;
            if (shapeX.length !== 2 || shapeX[0] !== shapeX[1]) {
                throw new Error('Square matrix expected.');
            }
            X.ensureUnsharedLocalStorage();
            let m = shapeX[0];
            let p: number[] = new Array(m);
            let sign: number;
            if (X.hasComplexStorage() && !DataHelper.isArrayAllZeros(X.imagData)) {
                sign = LU.clu(m, X.realData, X.imagData, p);
            } else {
                X.trimImaginaryPart();
                sign = LU.lu(m, X.realData, p);
            }
            return [X, p, sign];
        }

        function opLu(x: OpInput, compact: true): [Tensor, number[]];
        function opLu(x: OpInput, compact: false): [Tensor, Tensor, Tensor];
        function opLu(x: OpInput, compact: boolean = false): [Tensor, number[]] | [Tensor, Tensor, Tensor] {
            let [X, p, sign] = doCompactLU(x);            
            if (compact) {
                return [X, p];
            } else {
                let shapeX = X.shape;
                let L = Tensor.zeros(shapeX, DType.FLOAT64);
                let U = Tensor.zeros(shapeX, DType.FLOAT64);
                let P = Tensor.zeros(shapeX, DType.FLOAT64);
                LU.compactToFull(shapeX[0], false, X.realData, L.realData, U.realData);
                if (X.hasComplexStorage()) {
                    L.ensureComplexStorage();
                    U.ensureComplexStorage();
                    P.ensureComplexStorage();
                    LU.compactToFull(shapeX[0], true, X.imagData, L.imagData, U.imagData);
                }
                LU.permutationToFull(p, P.realData);
                return [L, U, P];
            }
        }

        function opInv(x: OpInput): Tensor {
            let [X, p, sign] = doCompactLU(x);
            let m = X.shape[0];
            let B = opEye(m);
            if (X.hasComplexStorage()) {
                B.ensureComplexStorage();
                LU.cluSolve(m, m, X.realData, X.imagData, p, B.realData, B.imagData);
            } else {
                LU.luSolve(m, m, X.realData, p, B.realData);
            }
            return B;
        }

        function opDet(x: OpInput): Scalar {
            let [X, p, sign] = doCompactLU(x);
            let m = X.shape[0];
            if (X.hasComplexStorage()) {
                let c = <ComplexNumber>X.getEl(0);
                for (let i = 1;i < m;i++) {
                    c = c.mulc(<ComplexNumber>X.getEl(i, i));
                }
                return c.mulr(sign);
            } else {
                let c = <number>X.getEl(0);
                for (let i = 1;i < m;i++) {
                    c = c * <number>X.getEl(i, i);
                }
                return c * sign;
            }
        }

        const opSvd = (x: OpInput): [Tensor, Tensor, Tensor] => {
            let X = x instanceof Tensor ? x.asType(DType.FLOAT64) : Tensor.toTensor(x);
            let shapeX = X.shape;
            if (shapeX.length !== 2) {
                throw new Error('Matrix expected.');
            }
            X.ensureUnsharedLocalStorage();
            let s = new Array(shapeX[1]);
            let ns = Math.min(shapeX[0], shapeX[1]);
            let V = Tensor.zeros([shapeX[1], shapeX[1]]);
            if (X.hasComplexStorage() && !DataHelper.isArrayAllZeros(X.imagData)) {
                V.ensureComplexStorage();
                SVD.csvd(shapeX[0], shapeX[1], true, X.realData, X.imagData, s, V.realData, V.imagData);
            } else {
                SVD.svd(shapeX[0], shapeX[1], true, X.realData, s, V.realData);
                X.trimImaginaryPart();
            }
            // m <  n : U - m x m, S - m x n, V - n x n
            // m >= n : U - m x n, S - n x n, V - n x n
            if (ns < shapeX[1]) {
                let S = Tensor.zeros(shapeX);
                let reS = S.realData;
                for (let i = 0;i < ns;i++) {
                    reS[i * shapeX[1] + i] = s[i];
                }
                return [<Tensor>X.get(':', ':' + ns, true), S, V];
            } else {
                return [X, opDiag(s), V];
            }
        };

        const opRank = (x: OpInput): number => {
            let X = x instanceof Tensor ? x.copy(true) : Tensor.toTensor(x);
            let shape = X.shape;
            let s = new Array(shape[1]);
            if (X.hasComplexStorage()) {
                SVD.csvd(shape[0], shape[1], false, X.realData, X.imagData, s, [], []);
            } else {
                SVD.svd(shape[0], shape[1], false, X.realData, s, []);
            }
            // threshold over singular values
            let sMax = s[0];
            if (sMax === 0) {
                return 0;
            }
            let threshold = EPSILON * sMax;
            let r = 0;
            for (;r < s.length;r++) {
                if (s[r] < threshold) {
                    break;
                }
            }
            return r;
        }

        const opEig = (x: OpInput): [Tensor, Tensor] => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let shapeX = X.shape;
            if (X.ndim !== 2 || shapeX[0] !== shapeX[1]) {
                throw new Error('Square matrix expected.');
            }
            if (X.hasComplexStorage()) {
                // Hermitian check
                if (opIsHermitian(X)) {    
                    X = X.copy(true);
                    let E = Tensor.zeros(shapeX);
                    let v = new Array(shapeX[0]);
                    E.ensureComplexStorage();
                    Eigen.eigHermitian(shapeX[0], X.realData, X.imagData, v,
                        E.realData, E.imagData);
                    return [E, opDiag(v)];
                } else {
                    throw new Error('Only Hermitian matrices are supported.');
                }
            } else {
                // symmetry check
                if (opIsSymmetric(X)) {
                    let E = Tensor.zeros(shapeX);
                    let v = new Array(shapeX[0]);
                    Eigen.eigSym(shapeX[0], X.realData, v, E.realData);
                    return [E, opDiag(v)];
                } else {
                    throw new Error('Only symmetric matrices are supported.')
                }
            }
        }

        return {
            isSymmetric: opIsSymmetric,
            isHermitian: opIsHermitian,
            eye: opEye,
            hilb: opHilb,
            diag: opDiag,
            matmul: opMatMul,
            kron: opKron,
            transpose: opTranspose,
            hermitian: opHermitian,
            trace: opTrace,
            inv: opInv,
            det: opDet,
            norm: opNorm,
            lu: opLu,
            svd: opSvd,
            rank: opRank,
            eig: opEig
        }
    }
}