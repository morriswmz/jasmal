# Implementation Details

* Different groups of operations are organized in different sub folders.
* The `generator` folder contains tools to dynamically generator unary and binary
  operations based on pre-defined function templates.
* Functions implemented by each provider should not depend on `this` as they
  will be copied to `Jasmal` object. Therefore, in the current implementation,
  all provider interfaces are implemented using object literals.
* Unless otherwise specified, operations will not be in-place and new tensors
  will be returned.