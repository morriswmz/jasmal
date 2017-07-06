export class ShapeHelper {

    public static validateShape(shape: number[]): void {
        if (shape.length === 0 || shape.length == undefined) {
            throw new Error('Shape must be a non-empty array.');
        }
        for (let i = 0;i < shape.length;i++) {
            if ((shape[i] | 0) !== shape[i]) {
                throw new Error('Shape can only consists of integers.');
            }
            if (shape[i] <= 0) {
                throw new Error(`The length of dimension ${i} must be positive.`);
            }
        }
    }

    public static getSizeFromShape(shape: number[]): number {
        let s = shape[0];
        for (let i = 1;i < shape.length;i++) {
            s *= shape[i];
        }
        return s;
    }

    public static computeStrides(shape: number[]): number[] {
        let strides = [1];
        let d = 1;
        for (let i = shape.length - 1;i > 0;i--) {
            d *= shape[i];
            strides.unshift(d);
        }
        return strides;
    }

    public static shapeToString(shape: number[]): string {
        return `[${shape.join('x')}]`;
    }

    public static isScalarShape(shape: number[]): boolean {
        for (let i = 0;i < shape.length;i++) {
            if (shape[i] !== 1) {
                return false;
            }
        }
        return true;
    }

    public static getSqueezedShape(shape: number[]): number[] {
        let newShape: number[] = [];
        for (let i = 0;i < shape.length;i++) {
            if (shape[i] !== 1) {
                newShape.push(shape[i]);
            }
        }
        if (newShape.length === 0) {
            newShape.push(1);
        }
        return newShape;
    }

    public static compareShape(shape1: number[], shape2: number[]): boolean {
        if (shape1.length !== shape2.length) {
            return false;
        }
        for (let i = 0;i < shape1.length;i++) {
            if (shape1[i] !== shape2[i]) {
                return false;
            }
        }
        return true;
    }

    public static compareSqueezedShape(shape1: number[], shape2: number[]): boolean {
        return ShapeHelper.compareShape(ShapeHelper.getSqueezedShape(shape1), ShapeHelper.getSqueezedShape(shape2));
    }

}