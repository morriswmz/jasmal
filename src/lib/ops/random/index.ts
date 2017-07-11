import { IRandomOpProvider } from '../definition';
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

        function rand(): number;
        function rand(shape: number[]): Tensor;
        function rand(shape?: number[]): number | Tensor {
            if (shape) {
                let t = Tensor.zeros(shape),
                    re = t.realData;
                for (let i = 0;i < t.size;i++) {
                    re[i] = engine.nextDouble();
                }
                return t;
            } else {
                return engine.nextDouble();
            }
        };

        function randn(): number;
        function randn(shape: number[]): Tensor;
        function randn(shape?: number[]): number | Tensor {
            if (shape) {
                let t = Tensor.zeros(shape),
                    re = t.realData;
                for (let i = 0;i < t.size;i++) {
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

        function randi(high: number): number;
        function randi(low: number, high: number): number;
        function randi(low: number, high: number, shape: number[]): Tensor;
        function randi(low: number, high?: number, shape?: number[] | undefined): number | Tensor {
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
                if (range === 0) {
                    for (let i = 0;i < t.size;i++) {
                        re[i] = low;
                    }
                } else {
                    for (let i = 0;i < t.size;i++) {
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
                x = a * 67108864 + b;
                if (x >= threshold) {
                    return x % (max + 1);
                }
            }
        }

        return {
            seed: s => engine.seed(s),
            rand: rand,
            randi: randi,
            randn: randn
        };
    }
}
