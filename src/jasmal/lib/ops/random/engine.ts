// Uniform random number generators
// ================================

export interface IRandomEngine {
    /**
     * Sets the seed of the RNG engine.
     * @param {number} x
     * @returns
     */
    setSeed(x: number): IRandomEngine;

    /**
     * Retrieves the seed of the RNG engine.
     */
    getSeed(): number;

    /**
     * Retrieves an unsigned 32-bit integer.
     * @returns
     */
    nextUint32(): number;

    /**
     * Retrieves an double within (0,1) with 53-bit precision.
     * @returns
     */
    nextDouble(): number;
}

const mulUint32: (x: number, y: number) => number = (<any>Math).imul instanceof Function
    ? (x, y) => (<any>Math).imul(x, y) >>> 0
    : (x, y) => {
        let ah = (x >>> 16) & 0xffff,
            al = x & 0xffff,
            bh = (y >>> 16) & 0xffff,
            bl = y & 0xffff;
        let high = (ah * bl + al * bh) & 0xffff;
        return (((high << 16) >>> 0) + (al * bl)) >>> 0;
    };

/**
 * MT19937 random number generator.
 * Adapted from http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
 * Original license attached immediately after this class.
 */
export class MT19937Engine implements IRandomEngine {

    private _mt = new Array<number>(624);
    private _mti = 625;
    private _seed;

    public setSeed(x: number): IRandomEngine {
        this._seed = x >>> 0;
        this._mt[0] = this._seed;
        for (this._mti = 1; this._mti < 624; this._mti++) {
            let d = this._mt[this._mti - 1] ^ (this._mt[this._mti - 1] >>> 30);
            this._mt[this._mti] = (mulUint32(d, 1812433253) + this._mti) >>> 0;
        }
        return this;
    }

    public getSeed(): number {
        return this._seed;
    }

    public nextUint32(): number {
        let mag01 = [0, 0x9908b0df];
        let y: number;
        if (this._mti >= 624) {
            if (this._mti === 625) {
                // not initialized
                this.setSeed(5489);
            }
            let kk = 0;
            for (; kk < 624 - 397; kk++) {
                y = (this._mt[kk] & 0x80000000) | (this._mt[kk + 1] & 0x7fffffff);
                this._mt[kk] = this._mt[kk + 397] ^ (y >>> 1) ^ mag01[y & 0x01];
            }
            for (; kk < 623; kk++) {
                y = (this._mt[kk] & 0x80000000) | (this._mt[kk + 1] & 0x7fffffff);
                this._mt[kk] = this._mt[kk + 397 - 624] ^ (y >>> 1) ^ mag01[y & 0x01]; 
            }
            y = (this._mt[623] & 0x80000000) | (this._mt[0] & 0x7fffffff);
            this._mt[623] = this._mt[396] ^ (y >>> 1) ^ mag01[y & 0x01];
            this._mti = 0;
        }
        y = this._mt[this._mti++];
        // tempering
        y ^= (y >>> 11);
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= (y >>> 18);
        return y >>> 0;
    }

    public nextDouble(): number {
        let a: number, b: number;
        do {
            a = this.nextUint32() >>> 5;
            b = this.nextUint32() >>> 6;
        } while (a === 0 && b === 0);
        // 2^26 = 67108864, 2^53 = 9007199254740992
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992);
    }

}

/* 
   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

/**
 * Wrapper for Math.random().
 */
export class NativeEngine implements IRandomEngine {

    public setSeed(_x: number): IRandomEngine {
        // we cannot specify the seed for Math.random().
        throw new Error('Seeding is not supported with the native random engine.');
    }

    public getSeed(): number {
        throw new Error('Seeding is not supported with the native random engine.');
    }

    public nextDouble(): number {
        let x: number;
        do {
            x = Math.random();
        } while (x === 0);
        return x;
    }

    public nextUint32(): number {
        return Math.floor(this.nextDouble() * 4294967296);
    }

}
