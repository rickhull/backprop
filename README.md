# Backward Propagation

This is a reimplementation of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) in Ruby.
It has been further simplified and some liberties have been taken with naming.

# Rationale

This can be used to train neural nets.  Typically the NN intends to minimize
a loss function.  An efficient way to do this is via gradient descent.
Mathematical derivatives and the chain rule from calculus are used to determine
inputs with the greatest influence on the output.  The inputs are manipulated
to minimize the output, represented as the loss function.  The smallest loss
implies the best performance at a given objective.

# Examples

```ruby
require 'backprop'

include BackProp

# F = ma

mass = Value.new(25, label: 'mass')
acc = Value.new(10, label: 'acc')
force = mass * acc
force.label = 'force'
p force
```

```
force(value=250 gradient=0 *(mass=25, acc=10))
        mass(value=25 gradient=0)
        acc(value=10 gradient=0)
```

Use backward propagation to determine the gradient (derivative with respect
to the caller of `#backward`) for each Value:

```ruby
force.backward
p force
```

```
force(value=250 gradient=1.0 *(mass=25, acc=10))
        mass(value=25 gradient=10.0)
        acc(value=10 gradient=25.0)
```

Notice the gradients have been updated, and the output gradient is 1.0.
