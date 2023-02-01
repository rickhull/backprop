[![Tests](https://github.com/rickhull/backprop/actions/workflows/test.yaml/badge.svg)](https://github.com/rickhull/backprop/actions/workflows/test.yaml)

# Simple Neural Networks

## With Backward Propagation And Gradient Descent

This is a reimplementation of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) in Ruby.
It has been further simplified and some liberties have been taken with naming.
Recursion is used by default for backward propagation and gradient descent.
There is support for operating on flat lists of parameters as well.

# Rationale

This can be used to train neural nets, typically to minimize a loss function.
An efficient way to do this is via gradient descent.
Mathematical derivatives and the chain rule from calculus are used to determine
  inputs with the greatest influence on the output.
The inputs are manipulated to minimize the output, represented as the loss
  function.
That is, the output of the neural net is a prediction.
The error or loss (prediction compared to the ideal, or known output) is
  computed for a variety of cases, and the network weights are adjusted to
  better match the desired output.
The smallest loss implies the best performance at a given objective.

# Fundamentals

## Value

A `Value` is a wrapper around a numeric quantity, typically a float.
This allows us to build mathematical expressions as well as the gradient
  for each Value relative to the larger expression (i.e. the derivative).

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

Because `mass` and `acc` are Values, and we assign the product of these two
  to `force`, now `force` is a Value with an operator `*` and children `mass`
  and `acc`:

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

The gradients have been updated, and the output gradient is 1.0.
We have a tree structure, where our inputs, mass and acceleration, are
  leaf nodes, and they combine via multiplication to make a parent node, or
  root node in this case, force.
By wrapping our numbers in the Value class, whenever we calculate a result,
  we have a tree structure representing that expression, and we can easily
  calculate derivatives for every node in the tree.

# Neural Networks

We will build up an understanding of neural networks one layer at a time.
They will ultimately be composed of many Values that can be thought of as
  a giant arithmetic expression, and we will use the gradients to perform
  gradient descent in order to minimize a loss function.

## Neuron

A neuron has a number of inputs which it combines to yield a single output.
Traditionally, each input has a weight, and the neuron itself has a bias, or
  a fixed amount which is added to each input when considering the output.
Sum each input value times its input weight, add the bias, and apply an
  *activation function* which "normalizes" the output to a predictable value,
  typically between -1.0 and 1.0.
In other words, if you send the right combination of signals, you can get the
  neuron to "fire".

```ruby
require 'perceptron'

include BackProp

# create a new neuron with 3 inputs; initial weights and bias are random
n = Neuron.new(3)

puts n
#=> N(-0.098, 1.000, 0.064) (0.468 relu)

p n
#=> -0.098| 0.000         1.000| 0.000    0.064| 0.000    0.468| 0.000

# send 0 to each input
output = n.apply(0)

puts output
#=> 0.468

# output is positive due to rectified linear unit (ReLU) activation function
output.value >= 0 #=> true

# if bias is positive, zero input should result in bias
(n.bias.value >= 0) ? (output.value == n.bias) : (output.value == 0) #=> true
```

## Layer

A layer is composed of several neurons.
Each neuron has the same number of inputs, so the layer has just a single
  number of inputs.
Each input is sent to each neuron in the layer.
If one layer is to feed into another, then the other layer's neurons must have
  an input count that matches the one layer's neuron count.

```ruby
require 'perceptron'

include BackProp

# create a new layer of 4 neurons with 3 inputs
l = Layer.new(3, 4)

puts l
```

```
N(0.957, 0.650, 0.995)  (-0.530 relu)
N(-0.482, 0.272, -0.467)        (0.905 relu)
N(-0.083, -0.519, -0.921)       (-0.811 relu)
N(-0.369, -0.688, -0.097)       (0.122 relu)
```

```ruby
# send 0 to each input
output = l.apply(0)

# returns an array of outputs, one for each neuron
output.size == 4 #=> true

puts output.map(&:value).join(', ')
#=> 0.0, 0.90522363833711, 0.0, 0.12226124806686789
```

## Multilayer Perceptron (MLP)

First, define a number of inputs.  Say 5 inputs, like temperature, etc.
Often we want a single output, which is the simple case.
Multiple outputs are possible but more complicated.
A single output could represent the recommended setting on a thermostat.
We can define multiple layers of neurons for our neural net which will feed
  on inputs and yield outputs.

```ruby
require 'perceptron'

include BackProp

# create a network with 3 inputs, 2 layers of 4 neurons, and one output neuron
n = MLP.new(3, [4, 4, 1])

puts n
```

```
N(0.660, 0.250, -0.387) (-0.677 relu)
N(0.931, 0.202, 0.596)  (0.861 relu)
N(0.101, 0.611, 0.885)  (-0.295 relu)
N(-0.858, 0.136, 0.091) (-0.309 relu)

N(-0.594, 0.178, 0.484, -0.208) (0.515 relu)
N(-0.295, -0.899, 0.437, -0.812)        (-0.200 relu)
N(-0.478, 0.230, -0.971, 0.897) (-0.858 relu)
N(0.636, 0.719, -0.857, -0.546) (-0.338 relu)

N(0.962, 0.529, 0.475, -0.837)  (-0.362 relu)
```

```ruby
# the first layer has 4 neurons, 3 inputs
n.layers[0].neurons.size == 4 #=> true
n.layers[0].neurons[0].weights.size == 3 #=> true

# next layer has 4 neurons, 4 inputs
n.layers[1].neurons.size == 4 #=> true
n.layers[1].neurons[0].weights.size == 4 #=> true

# final layer has 1 neuron, 4 inputs
n.layers[2].neurons.size == 1 #=> true
n.layers[2].neurons[0].weights.size == 4 #=> true

# send 0 to each input
output = n.apply(0)

# returns an output value corresponding to the output neuron
# output is positive to due to ReLU
output.value >= 0 #=> true

puts output
#=> 0.045
```

## Performance Evaluation

So, our neural net, given some inputs, has yielded an output.  Is it any good?
We need to compare it to a known-good outcome.
We must already have a set of examples where certain inputs correspond to
desired outputs.
Maybe an input signal like `101` is expected to output `1`.  `010` should also yield `1`, but `001` yields `0` (obviously!).
Whatever the desired mapping of inputs to outputs is, the neural net can learn
to provide it, in a very black box sort of manner.

At first, the neural net has random weights, so for `101` it will initially
output something like `0.173`.
And maybe for `010`: `0.865`.  Error is present and quantifiable.
We'll call the error (the difference between the net's output and desired
output) the *loss*, and our job is now to minimize the loss.
There are many ways to quanitify the loss, and the way we choose to quantify
the loss is our so-called *loss function*.

## Loss Function

Typically we have dozens, hundreds, thousands, etc. of training examples or
cases, which map a set of inputs to a desired output.
We'll feed each case's inputs to the neural net and get a prediction.
The difference between prediction and desired output is the error.
The loss function sums the errors to yield a single loss score or number.

A common loss function is *mean squared error*.
That is, for every error (one for every case) positive or negative, square it,
and take the mean of all the squares (sum and divide by the number of cases).

## Forward Calculation

For every case with a known desired output, feed the input to the MLP
(multilayer perceptron, aka neural net) to yield a prediction.
The prediction is merely the evaluation of a gigantic arithmetic expression,
which we have captured by representing all relevant terms as Values,
particularly the weights and bias of each Neuron.  For each prediction, with
a known desired output, calculate the error, and collect all the errors into
the output of the loss function, say *mean squared error*, to represent
our ultimate outcome for this iteration of the MLP: the **loss**.

## Backward Propagation

We will call `loss.backward` to calculate the derivative for every descendent
term in the final, gigantic loss expression, with respect to the loss itself.
This derivative will be called the gradient and is attached to every term by
nature of its representation as a Value object.
A larger gradient means a term has a larger effect on the output prediction.
The sign of the gradient informs which direction the output will change as
a parameter value changes.

## Gradient Descent

For every case, we've run the network forward to generate a *prediction*,
yielding an *error* (relative to the desired output).
Our *loss function* collects the errors into single *loss* number which
represents the ultimate outcome for all cases and this iteration of the MLP.

Now, we will morph the MLP by adjusting the weights and bias for every neuron,
via the process of *gradient descent*.
Simply adjust every value by a small step (e.g. 0.05) multiplied by the
value's gradient.
In keeping with "descent" and "minimization", if the gradient is positive,
the value goes down; negative gradients imply the value should increase.

Now we have a new MLP with all neuron weights and biases slightly different
than before.  Its predictions should be slightly better and yield slightly
smaller loss.

## Forward and Backward

Loop:

1. Run the network forward to generate a new output.
2. Determine the loss; it should be smaller over time
3. Backward propagate the gradients
   (derivatives for each term (Value) with respect to the loss)
4. Adjust all weights slightly, according to their gradients.

## Further Reading

* [demo/loss.rb](demo/loss.rb)
