import { IRandomOpProvider } from './definition';
import { Tensor } from '../../tensor';
import { IRandomEngine, MT19937Engine, NativeEngine } from './engine';
import { MAX_SAFE_INTEGER } from '../../constant';

export class RandomOpProviderFactory {
    public static create(rngEngine: IRandomEngine | string = 'twister'): IRandomOpProvider {
        // init engine
        let engine: IRandomEngine;
        if (Object.prototype.toString.call(rngEngine) === '[object String]') {
            switch (rngEngine) {
                case 'native':
                    engine = new NativeEngine();
                    break;
                case 'twister':
                case 'MT19937Engine':
                    engine = new MT19937Engine();
                    break;
                default:
                    throw new Error(`Unknown random engine name "${rngEngine}".`);
            }
        } else {
            engine = <IRandomEngine>rngEngine;
        }

        // variables for the normal random number generator
        let randnNeedNewPair = true;
        let randnNumber2 = 0;

        function opSeed(s: number): void;
        function opSeed(): number;
        function opSeed(s?: number): number | void {
            if (s == undefined) {
                return engine.getSeed();
            } else {
                engine.setSeed(s);
                return;
            }
        }

        function opRand(): number;
        function opRand(shape: ArrayLike<number>): Tensor;
        function opRand(shape?: ArrayLike<number>): number | Tensor {
            if (shape) {
                let t = Tensor.zeros(shape),
                    re = t.realData,
                    i: number,
                    n = t.size;
                for (i = 0;i < n;i++) {
                    re[i] = engine.nextDouble();
                }
                return t;
            } else {
                return engine.nextDouble();
            }
        };

        function opRandn(): number;
        function opRandn(shape: ArrayLike<number>): Tensor;
        function opRandn(shape?: ArrayLike<number>): number | Tensor {
            if (shape) {
                let t = Tensor.zeros(shape),
                    re = t.realData,
                    i: number,
                    n = t.size;
                for (i = 0;i < n;i++) {
                    re[i] = _nextRandn();
                }
                return t;
            } else {
                return _nextRandn();
            }
        }

        function _nextRandn(): number {
            if (randnNeedNewPair) {
                let v1: number, v2: number, rsq: number, fac: number;
                do {
                    v1 = 2.0 * engine.nextDouble() - 1.0;
                    v2 = 2.0 * engine.nextDouble() - 1.0;
                    rsq = v1 * v1 + v2 * v2;
                } while (rsq >= 1.0 || rsq === 0.0);
                fac = Math.sqrt(-2.0 * Math.log(rsq) / rsq);
                randnNumber2 = v1 * fac;
                randnNeedNewPair = false;
                return v2 * fac;
            } else {
                randnNeedNewPair = true;
                return randnNumber2;
            }
        }

        function opRandi(high: number): number;
        function opRandi(low: number, high: number): number;
        function opRandi(low: number, high: number, shape: ArrayLike<number>): Tensor;
        function opRandi(low: number, high?: number, shape?: ArrayLike<number> | undefined): number | Tensor {
            if (high == undefined) {
                high = low;
                low = 0;
            }
            if ((low < 0) || (low > MAX_SAFE_INTEGER) || (Math.floor(low) !== low)) {
                throw new Error('The value low must be a nonnegative integer that is less than 2^53.')
            } 
            if ((high < 0) || (high > MAX_SAFE_INTEGER) || (Math.floor(high) !== high)) {
                throw new Error('The value high must be a nonnegative integer that is less than 2^53.')
            }
            if (low > high) {
                throw new Error('The value low cannot be higher than high.')
            }
            let range = high - low;
            if (shape) {
                let t = Tensor.zeros(shape),
                    re = t.realData;
                let i: number, n = t.size;
                if (range === 0) {
                    for (i = 0;i < n;i++) {
                        re[i] = low;
                    }
                } else {
                    for (i = 0;i < n;i++) {
                        re[i] = low + _nextRandi(range);
                    }
                }
                return t;
            } else {
                return range === 0 ? low : low + _nextRandi(range);
            }
        }

        function _nextRandi(max: number): number {
            let threshold = (MAX_SAFE_INTEGER + 1) % (max + 1);
            let a: number, b: number, x: number;
            for (;;) {
                a = engine.nextUint32() >>> 5;
                b = engine.nextUint32() >>> 6;
                x = a * 67108864 + b; // 53-bit
                if (x >= threshold) {
                    return x % (max + 1);
                }
            }
        }

        return {
            seed: opSeed,
            rand: opRand,
            randi: opRandi,
            randn: opRandn
        };
    }
}
