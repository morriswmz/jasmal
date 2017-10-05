import { MAX_SAFE_INTEGER } from "../constant";


const PRIME_TABLE = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
    307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383,
    389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569,
    571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647,
    653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
    751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839,
    853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
    947, 953, 967, 971, 977, 983, 991, 997];

export class Factorization {

    /**
     * Checks if a nonnegative integer is a prime number.
     * @param x 
     */
    public static isPrime(x: number): boolean {
        if (x < 2 || !isFinite(x) || Math.floor(x) !== x) {
            return false;
        }
        if (x % 2 === 0) {
            return x === 2;
        } else {
            let ub = Math.floor(Math.sqrt(x));
            let cur = 3;
            while (cur <= ub) {
                if (x % cur === 0) {
                    return false;
                }
                cur += 2;
            }
            return true;
        }
    }

    /**
     * Finds the next prime starting from x.
     * @param x 
     */
    public static nextPrime(x: number): number {
        x = Math.ceil(x);
        if (x <= 2) {
            return 2;
        }
        if (x % 2 === 0) {
            x++;
        }
        while (!Factorization.isPrime(x)) {
            x += 2;
        }
        return x;
    }

    /**
     * Factorizes a composite number.
     * @param x
     */
    public static factorize(x: number): number[] {
        if (x < 0 || !isFinite(x) || Math.floor(x) !== x) {
            throw new Error('Input must be a nonnegative integer.');
        }
        if (x > MAX_SAFE_INTEGER) {
            throw new Error(`Maximum allowed value is ${MAX_SAFE_INTEGER}.`);
        }
        if (x < 4) {
            return [x];
        }
        let factors: number[] = [];
        let curIndex = 0;
        let curFactor = PRIME_TABLE[curIndex];
        let ub = Math.floor(Math.sqrt(x));
        let lastX = x;
        while (true) {
            while (x % curFactor === 0) {
                factors.push(curFactor);
                x /= curFactor;
            }
            if (x === 1) {
                break;
            }
            if (x !== lastX) {
                ub = Math.floor(Math.sqrt(x));
                lastX = x;
            }
            if (curIndex < PRIME_TABLE.length) {
                curFactor = PRIME_TABLE[curIndex];
                curIndex++;
            } else {
                curFactor = Factorization.nextPrime(curFactor + 2);
            }
            if (curFactor > ub) {
                factors.push(x);
                break;
            }
        }
        return factors;
    }

}
