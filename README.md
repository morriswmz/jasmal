# JASMAL

**JASMAL** stands for **J**ust **A**nother Java**S**cript **MA**trix **L**ibrary,
or **JA**va**S**cript **MA**trix **L**ibrary. This is a **work-in-progress**
library I used to create interactive simulations on my
[blog](http://research.wmz.ninja/articles/2017/07/music-in-the-browser.html).
Initially, I thought it should be a small project. Later, I realized that I
need to implement much more functionalities to make it possible to run the
even the classical DOA estimation algorithms. It was a good learning progress,
especially for numerical algorithms in matrix computations. Because this project
is open source, I hope you find it useful when looking for implementations of
algorithms in matrix computations.

Despite its name, JASMAL can actually handle multi-dimensional arrays. It also has

* built-in complex number support
* flexible indexing schemes (e.g., `get('::-1', ':')` returns a new matrix
  with the rows reversed)
* broadcasting support for various binary operations
* subroutines for common matrix operations such as `trace()`, `inv()`, `det()`,
  `linsolve()`, `rank()`, `kron()`.
* subroutines for LU decomposition, QR decomposition, singular value
  decomposition, and eigendecomposition for both real and complex matrices 
* set functions such as `union()`, `intersect()`, and `setdiff()`

**Important:** Although many of the operations have been [tested](src/jasmal/spec/)
for common inputs, there may be still many bugs.

# Basic Usage

To access the JASMAL engine, simply use `const T = require('jasmal').JasmalEngine.createInstance()`.

For browsers, you can build a standalone version with `tsconfig.client.json` and
import it using [SystemJS](https://github.com/systemjs/systemjs):

``` HTML
<script type="text/javascript" src="system-production.js"></script>
<script type="text/javascript" src="jasmal.js"></script>
<script type="text/javascript">
    System.import('jasmal/index').then(function (jasmal) {
        var T = jasmal.JasmalEngine.createInstance();
        console.log(T.eye(3).toString());
    });
</script>
```

## Tensor objects and complex numbers

JASMAL is built around [tensor](src/jasmal/lib/tensor.ts) objects, which use typed
arrays for data storage. In plain JavaScript, jagged arrays is usually used to
store multi-dimensional arrays. The following code show how to convert between
JavaScript arrays and tensor objects:

``` JavaScript
// Creates a tensor from JavaScript arrays.
let A = T.fromArray([[1, 2], [3, 4]]); // real 2 x 2
// Specifying the data type for the underlying storage.
// Note that [] means no imaginary part.
let B = T.fromArray([[1, 2], [3, 4]], [], T.INT32);
let C = T.fromArray([[1, 2], [3, 4]], [[-1, -2], [-3, -4]]); // complex 2 x 2
// Convert a tensor to a JavaScript array.
let a = A.toArray(true); // real part only, arr = [[1, 2], [3, 4]]
let [reC, imC] = C.toArray(false); // convert both real and imaginary parts
                                   // reC = [[1, 2], [3, 4]];
                                   // imC = [[-1, -2], [-3, -4]];
// Check if the given object is an instance of Tensor.
T.isTensor(A); // true
T.isTensor([1, 2]); // false
```

Note that during the conversions the data are **always copied** because JASMAL
cannot be sure whether you will modify the array elements in the future.

JASMAL uses `Float64Array` as the underlying storage by default, which is marked
with `T.FLOAT64`. Another two data types supported by JASMAL are `T.INT32` and
`T.LOGIC`, which are backed by `Int32Array` and `Uint8Array`, respectively.
You can convert between different data types via `asType()`:

``` JavaScript
let a = T.fromArray([1, 2, 3], [], T.INT32);
// INT32 -> FLOAT64
let b = A.asType(T.FLOAT64);

let z = T.fromArray([1, 2, 3], [3, 4, 5], T.FLOAT64);
// FLOAT64 -> INT32, down casting is handled by typed arrays
// Here m is a complex vector whose real part and imaginary parts are integers.
let m = z.asType(T.INT32);
// You CANNOT convert complex numbers to logic values.
let n = z.asType(T.LOGIC); // Will throw an error!
```

JASMAL also includes a built-in [ComplexNumber](src/jasmal/lib/complexNumber.ts)
type to support complex scalars. Complex numbers can be created with the
following code:

``` JavaScript
// Creates a complex number.
let c = T.complexNumber(1, -1);
// Retrieves the real part.
let re = c.re;
// Test if c is a ComplexNumber instance.
console.log(T.isComplexNumber(c)); // true
```

Instead of only allowing tensor objects as inputs, most of the JASMAL functions
also allows JavaScript arrays (including typed arrays), ComplexNumber instances,
or numbers as inputs. Conversion to tensor objects is automatically performed,
and converted tensor objects will **always** have FLOAT64 as the data type.
If you wish to control the data type, you will need to manually do the
conversion using `fromArray()`.
For instance, `T.add([[1], [2]], [[3, 4]])` produces the same result as the 
following code:

``` JavaScript
let x = T.fromArray([[1], [2]]);
let y = T.fromArray([[3, 4]]);
let z = T.add(x, y);
```

This is very convenient, but it leads to one problem: how is the output type
determined? For instance, should `T.add(1, 2)` output a number or a tensor
object? In JASMAL, unless otherwise specified, the following rules applies:

* If all the inputs are scalars (`number` or `ComplexNumber`), the output will
  be a scalar. In this case, if the output's imaginary part is zero, a
  JavaScript number will be returned. Otherwise a `ComplexNumber` instance will
  be returned.
* If any of the inputs is a tensor object or a JavaScript array, the output will
  be a tensor object.
* Some functions have a parameter named `keepDims`. If `keepDims` is set to
  `true`, the output will always be a tensor object.

## Creation

``` JavaScript
// Creates a 3 x 3 zero matrix.
let Z = T.zeros([3, 3]);
// Creates a 3 x 3 identity matrix.
let I = T.eye([3, 3]);
// Creates a 3 x 4 x 3 array whose elements are all ones with data type INT32.
let X = T.ones([3, 4, 3], T.INT32);

```

## Indexing/Slicing

JASMAL support a flexible indexing/slicing scheme similar to that in MATLAB/Python.

``` JavaScript
let A = T.eye([3, 3]);
A.set(0, 10); // sets the first element to 10
A.set(':', 1); // sets all the elements to 1
A.set(-1, 10); // sets the last element to 10
A.set(0, ':', 2); // sets all the elements in the first row to 2
A.set('::2', 0); // sets all the elements with even indices to 0
A.set([0, -1], [0, -1], [[1, 2], [3, 4]]); // sets four corners to 1, 2, 3, 4
A.set(function (x) { return x < 0; }, 0); // sets all negative elements to 0

A.get(0); // gets the first element
A.get(':', 0); // gets the first column as a 1D vector
// By default, get() will attempt to remove singleton dimensions. If you do not
// want this behavior, you can specify keepDims = true.
A.get(':', 0, true); // gets the first columns as a 2D column vector

A.get('::-1', ':'); // gets a new matrix with the rows reversed
A.get([0, -1], [0, -1]); // gets a new matrix consists of the four corners
A.get([0, 1, 1, 2, 0, 1], ':'); // sample rows
```

## Matrix/Tensor manipulation

JASMAL provides basic tools for tensor manipulation. You can reshape, flatten,
squeeze, tile or concatenate tensors. See the [definition file](src/jasmal/lib/ops/core/definition.ts)
for details.

``` JavaScript
let A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
// Reshaping.
let B = T.reshape(A, [3, 2]);
// The following is equivalent to the above, where we use -1 to indicate that
// the length of this dimension need to be calculated automatically. At most one
// -1 can be used in the new shape.
let B1 = T.reshape(A, [-1, 2]);
// Flattening
let a = T.flatten(A); // should have shape [6]
// Vectorizing
let v = T.vec(A); // should have shape [6, 1]
// Remove singleton dimensions.
let S = T.ones([1, 2, 1, 3]);
T.squeeze(S); // should have shape [2, 3]
// Concatenating at the specified dimension.
let X1 = T.ones([2, 4, 3]);
let X2 = T.zeros([2, 2, 3]);
let X3 = T.ones([2, 1, 3]);
let Z = T.concat([X1, X2, X3], 1); // should have a shape of [2, 7, 3]
/* Tiling. C should be
 *  [[1, 2, 1, 2],
 *   [3, 4, 3, 4],
 *   [1, 2, 1, 2],
 *   [3, 4, 3, 4]] */
let C = T.tile([[1, 2], [3, 4]], [2, 2]);
// Permute axis.
let D1 = T.rand([2, 3, 4]);
let D2 = T.permuteAxis(D1, [2, 1, 0]); // D2 has a shape of [4, 3, 2]
```

## Arithmetic operations

JASMAL supports basic arithmetic operations between tensors of compatible shapes,
including `add()`, `sub()`, `mul()`, `div()`, `neg()`, `reciprocal()`, and
`rem()`. For binary operations, broadcasting is supported.

``` JavaScript
// Only when both operands are scalar, a scalar is returned.
let s = T.add(1, 2); // returns 3

let A = T.ones(3);
// Subtract one from matrix A and return the result as a new matrix.
let B1 = T.sub(A, 1);
// Subtract one from matrix A but do it in-place.
T.sub(A, 1, true);
// Subtract 2+3j from A. The result will be complex.
let B2 = T.sub(A, T.complexNumber(2, 3));

// Broadcasting
let X = T.rand([3, 3]);
let C = T.mul([[1], [2], [3]], X);
// The above operation is equivalent to the following one:
let C1 = T.matmul(T.diag([1, 2, 3]), X);
// or the following one:
let C2 = T.mul(T.tile([[1], [2], [3]], [1, 3]), X);
```

## Math functions

JASMAL supports various math functions. Many of them also accepts complex
inputs. You can check the definitions in the following files:

* [Basic math functions](src/jasmal/lib/ops/math/basic.ts)
* [Logarithm and exponential functions](src/jasmal/lib/ops/math/logexp.ts)
* [Power functions](src/jasmal/lib/ops/math/pow.ts)
* [Rounding functions](src/jasmal/lib/ops/math/rounding.ts)
* [Trigonometry functions](src/jasmal/lib/ops/math/trigonometry.ts)
* [Special functions](src/jasmal/lib/ops/math/special.ts)

``` JavaScript
// Absolute value
T.abs([[1, 2], [-3, 4]]);
// Sine
T.sin(T.linspace(0, Math.PI*2, 100));
// Square root
T.sqrt(-1);
// Element-wise minimum.
T.min2([1, 2], -1);
```

## Random number generation

To support seeding, Jasmal uses the [Mersenne twister](https://en.wikipedia.org/wiki/Mersenne_Twister)
engine by default to generate random numbers.

``` JavaScript
// Specify the seed.
T.seed(42);
// Retrieves an double within (0,1) with 53-bit precision.
let x = T.rand(); 
// Creates a 3x4x5 tensor whose elements sampled from the normal distribution.
let N = T.randn([3, 4, 5]);
// Generate 10 random integers within [0, 10].
let Z = T.randi(0, 10, [10]);
``` 

You can configure JASMAL to use the JavaScript's `Math.random()` using the
following code when creating the JASMAL instance:

``` JavaScript
const T = require('jasmal').JasmalEngine.createInstance({
  rngEngine: 'native'
});
```

## Matrix operations

JASMAL supports various matrix operations. For details, see the definitions
[here](src/jasmal/lib/ops/matrix/definition.ts).

``` JavaScript
let A = T.rand([3, 3]), B = T.rand([3, 3]);
// Construct a complex matrix by combining real and imaginary parts.
let C = T.complex(A, B);
// Matrix multiplication
let AB = T.matmul(A, B);
// You can specify a modifier for B
let ABt = T.matmul(A, B, T.MM_TRANSPOSED);
// Extract diagonal elements.
let d = T.diag(A);
// Construct a diagonal matrix.
let D = diag(d);
// Matrix transpose/Hermitian.
let At = T.transpose(A);
let Ch = T.hermitian(C);
// Kronecker product.
let K = T.kron(A, B);
// Inverse (JASMAL uses LUP decomposition to compute the inverse)
let Ainv = T.inv(A);
// Determinant (JASMAL uses LUP decomposition to compute the determinant)
let detA = T.det(A);
// SVD
let [U1, S1, V1] = T.svd(A);
// SVD also works for complex matrices.
let [U2, S2, V2] = T.svd(C);
// Eigendecomposition for general square matrices.
let [E1, L1] = T.eig(A);
// Eigendecomposition also works for general complex square matrices.
let [E2, L2] = T.eig(C);
// Solve the linear system AX = B.
let X = T.linsolve(A, B);
```

## Data functions

JASMAL also includes several functions for data processing. For details, see
the definitions [here](src/jasmal/lib/ops/data/definition.ts).

``` JavaScript
let A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
// Gets the sum of all the elements.
let sum = T.sum(A);
// Sums each row and returns a column vector. We specify keepDims = true here.
let sums = T.sum(A, 1, true);
// Sorts all the elements in A in descending order and return the indices I
// such that As is given by `A.get(I)`.
let [As, I] = T.sort(A, 'desc', true);
// Histogram (10 bins by default).
let [H, E] = T.hist(T.randn([1000])); // H stores the frequencies and E stores
                                      // the edges of the bins.
```

## Polynomial functions

JASMAL supports polynomial evaluation and polynomial root finding. For details,
see the definitions [here](src/jasmal/lib/ops/poly/definition.ts).

``` JavaScript
// Evaluates x^2+2x+3 at 2 and 4.
let v = T.polyval([1, 2, 3], [2, 4]);
// Evaluates the matrix polynomial X^2 + 2X + 3I.
let V = T.polyvalm([1, 2, 3], [[1, 2], [3, 4]]);
// Finds the least-squares fit of the sine function with a cubic function.
let x = T.linspace(0, 2, 20);
let y = T.sin(x);
let p = T.polyfit(x, y);
// Finds roots of x^3 + x + 1 = 0.
let r = T.roots([1, 0, 1, 1]);
```

## Set functions

JASMAL includes commonly used set functions such as `union()`, `intersect()`, 
and `setdiff()`. For details, see the definitions [here](src/jasmal/lib/ops/set/definition.ts).

``` JavaScript
let A = T.fromArray([5, 4, 4, 3, 0]);
let B = T.fromArray([1, 1, 3, 3]);
// Finds the unique elements in A
let C = T.unique(A); // [0, 3, 4, 5]
// Union of sets.
let U = T.union(A, B); // [0, 1, 3, 4, 5]
// Checks membership.
let M = T.isin(B, A); // Returns a logic vector [0, 0, 1, 1]
// Computes set difference.
let D = T.setdiff(A, B); // [0, 4, 5]
```

# Performance

Unfortunately, JavaScript is relatively slow for dense numerical computations.
Therefore, multiply two 1000 x 1000 matrices or performing the singular value
decomposition of a 500 x 500 matrix may take several seconds in the browser.
Nevertheless, JASMAL is still useful in demonstrating small scale numeric
problems.

## Vectorization and indexing

For vectorized operations, JASMAL's performance should not deviate too far
from that of the native JavaScript. However, for scalar operations, JASMAL can
be slower because of the overheads. For instance, calling `T.sqrt(x)` 1000 times
for a scalar x is slower than calling `T.sqrt(X)` where `X` is a 1000-element
tensor object.

Element by element indexing can also be slow with JASMAL because of the overhead
needed for implementing the flexible indexing schemes. For instance, if you
want to set all the elements in a matrix to zero, `M.set(':', 0)` is much much
faster than the following code:
``` JavaScript
for (let i = 0;i < m;i++) {
    for (let j = 0;j < n;j++) {
        M.set(i, j, 0);
    }
}
```

## Reference copy

For reshaping operations `reshape()`, `flatten()`, and `vec()`, data are not
copied immediately to avoid unnecessary memory allocations:

``` JavaScript
let A = T.randn([200, 200]);
let B = T.reshape([100, 400]); // Now A and B share the same underlying storage.
B.set(0, 0, 100); // A copy of the data in A is made before setting the element at (0, 0),
                  // A and B no longer share the same underlying storage.
```

## Accessing the underlying data storage directly

JASMAL stores multi-dimensional arrays in the row major order. If you want to 
completely bypass the indexing overhead of JASMAL's indexing functions, you can
directly access the underlying storage and manipulate them:

``` JavaScript
// If you want to write to the underlying storage directly, ensure that it is
// not shared.
A.ensureUnsharedLocalStorage();
let re = A.realData;
re[0] = 1;
let im;
// If a matrix does not have complex data storage, accessing its imaginary data
// storage will result in an error.
if (!A.hasComplexStorage()) {
    // Make sure the complex data storage is available.
    A.ensureComplexStorage();
}
im = A.imagData;
im[0] = -1;
```

# License

JASMAL itself is released under the [MIT](LICENSE.md) license.

[eigen.ts](src/jasmal/lib/ops/matrix/decomp/eigen.ts)
contains several subroutines ported from the pristine Fortran code
[EISPACK](http://www.netlib.org/eispack/),
which are distributed using the Modified BSD or MIT license
([source](http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg01379.html)).
For these subroutines, all rights reserved by the authors of EISPACK. 
