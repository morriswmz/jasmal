import { IArithmeticOpProvider } from '../definition';
import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { DType, OutputDTypeCalculator, DTypeHelper } from '../../dtype';
import { Tensor } from "../../tensor";
import { ComplexNumber } from "../../complexNumber";
import { ShapeHelper } from "../../helper/shapeHelper";

const PROG_DIV_RC =
`if (Math.abs($reY) > Math.abs($imY)) {
    $tmp1 = $imY / $reY;
    $tmp2 = $reY + $imY * $tmp1;
    $imZ = -$tmp1 / $tmp2 * $reX
    $reZ = $reX / $tmp2;
} else {
    if ($imY === 0) {
        $reZ = Infinity;  $imZ = 0;
    } else {
        $tmp1 = $reY / $imY;
        $tmp2 = $reY * $tmp1 + $imY;
        $imZ = -$reX / $tmp2;
        $reZ = $tmp1 / $tmp2 * $reX;
    }
}`;

const PROG_DIV_CC =
`if (Math.abs($reY) > Math.abs($imY)) {
    $tmp1 = $imY / $reY;
    $tmp2 = $reY + $imY * $tmp1;
    $tmp3 = $reX;
    $reZ = ($tmp3 + $imX * $tmp1) / $tmp2;
    $imZ = ($imX - $tmp3 * $tmp1) / $tmp2;
} else {
    if ($imY === 0) {
        if ($reX === 0) {
            if ($imX === 0) {
                $reZ = NaN; $imZ = NaN;
            } else {
                $reZ = 0; $imZ = $imX / 0;
            }
        } else {
            $reZ = $reX / 0;
            $imZ = $imX === 0 ? 0 : $imX / 0;
        }
    } else {
        $tmp1 = $reY / $imY;
        $tmp2 = $imY + $reY * $tmp1;
        $tmp3 = $reX;
        $reZ = ($imX + $tmp3 * $tmp1) / $tmp2;
        $imZ = (- $tmp3 + $imX * $tmp1) / $tmp2;
    }
}`;

const PROG_RECIPROCAL_C =
`if (Math.abs($imX) < Math.abs($reX)) {
    $tmp1 = $imX / $reX; $tmp2 = $reX + $imX * $tmp1;
    $reY = 1 / $tmp2; $imY = - $tmp1 / $tmp2;
} else {
    if ($imX === 0) {
        $reY = Infinity; $imY = 0;
    } else {
        $tmp1 = $reX / $imX; $tmp2 = $imX + $reX * $tmp1;
        $reY = $tmp1 / $tmp2; $imY = - 1 / $tmp2;
    }
}`;

export class ArithmeticOpProviderFactory {
    public static create(): IArithmeticOpProvider {
        let compiler = TensorElementWiseOpCompiler.GetInstance();
        return {
            add: compiler.makeBinaryOp({
                opRR: '$reZ = $reX + $reY;',
                opRC: '$reZ = $reX + $reY; $imZ = $imY;',
                opCR: '$reZ = $reX + $reY; $imZ = $imX;',
                opCC: '$reZ = $reX + $reY; $imZ = $imX + $imY;'
            }),
            sub: compiler.makeBinaryOp({
                opRR: '$reZ = $reX - $reY;',
                opRC: '$reZ = $reX - $reY; $imZ = -$imY;',
                opCR: '$reZ = $reX - $reY; $imZ = $imX;',
                opCC: '$reZ = $reX - $reY; $imZ = $imX - $imY;'
            }),
            mul: compiler.makeBinaryOp({
                opRR: '$reZ = $reX * $reY;',
                // calculate the imaginary part first so no temporary variable
                // is needed.
                opRC: '$imZ = $reX * $imY; $reZ = $reX * $reY;',
                opCR: '$reZ = $reX * $reY; $imZ = $imX * $reY;',
                opCC: '$tmp1 = $reX; $reZ = $tmp1 * $reY - $imX * $imY; $imZ = $tmp1 * $imY + $imX * $reY;'
            }),
            div: compiler.makeBinaryOp({
                opRR: '$reZ = $reX / $reY;',
                opRC: PROG_DIV_RC,
                opCR: '$reZ = $reX / $reY; $imZ = $imX / $reY;',
                opCC: PROG_DIV_CC
            }, {outputDTypeCalculator: OutputDTypeCalculator.bToFloat}),
            neg: compiler.makeUnaryOp({
                opR: '$reY = -$reX;',
                opC: '$reY = -$reX; $imY = -$imX;'
            }),
            reciprocal: compiler.makeUnaryOp({
                opR: '$reY = 1 / $reX;',
                opC: PROG_RECIPROCAL_C
            }, {outputDTypeCalculator: OutputDTypeCalculator.uToFloat})
        }
    }
}