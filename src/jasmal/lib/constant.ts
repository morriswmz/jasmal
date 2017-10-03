
/**
 * Maximum safe integer in JavaScript (2^53 - 1).
 */
export const MAX_SAFE_INTEGER = 9007199254740991;
/**
 * Minimum safe integer in JavaScript -(2^53 - 1).
 */
export const MIN_SAFE_INTEGER = -9007199254740991;

/**
 * Machine precision.
 */
export const EPSILON = 7/3 - 4/3 - 1;

export const M_PI_2 = Math.PI / 2;

export const LOGE2 = Math.log(2);

export const NOT_IMPLEMENTED = () => {
    throw new Error('Not implemented.');
};
