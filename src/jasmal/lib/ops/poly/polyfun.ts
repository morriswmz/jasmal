export class PolynomialEvaluator {

    public static evalPolyRR(p: ArrayLike<number>, x: number): number {
        if (p.length === 0) {
            return 0;
        }
        let acc = p[0];
        for (let i = 1;i < p.length;i++) {
            acc = acc * x + p[i];
        }
        return acc;
    }

    public static evalPolyRC(p: ArrayLike<number>, reX: number, imX: number): [number, number] {
        if (p.length === 0) {
            return [0, 0];
        }
        let accRe = p[0];
        let accIm = 0;
        let tmp: number;
        for (let i = 1;i < p.length;i++) {
            tmp = accRe;
            accRe = accRe * reX - accIm * imX;
            accIm = tmp * imX + accIm * reX;
            accRe += p[i];
        }
        return [accRe, accIm];
    }

    public static evalPolyCR(reP: ArrayLike<number>, imP: ArrayLike<number>, x: number): [number, number] {
        return [PolynomialEvaluator.evalPolyRR(reP, x), PolynomialEvaluator.evalPolyRR(imP, x)];
    }

    public static evalPolyCC(reP: ArrayLike<number>, imP: ArrayLike<number>, reX: number, imX: number): [number, number] {
        if (reP.length === 0) {
            return [0, 0];
        }
        let accRe = reP[0];
        let accIm = imP[0];
        let tmp: number;
        for (let i = 1;i < reP.length;i++) {
            tmp = accRe;
            accRe = accRe * reX - accIm * imX;
            accIm = tmp * imX + accIm * reX;
            accRe += reP[i];
            accIm += imP[i];
        }
        return [accRe, accIm];
    }

}