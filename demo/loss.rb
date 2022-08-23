require 'perceptron'

include BackProp

num_inputs = 3
num_examples = 9
net_structure = [4, 4, 1]
gradient_step = 0.1
iterations = 999

# binary classifier; 9 sets of inputs that map to 1 or 0
inputs = BackProp.rand_inputs(num_inputs, num_examples, (-1.0..1.0))
outputs = BackProp.rand_outputs(num_examples, 2)
predictions = []

n = MLP.new(num_inputs, net_structure)
inputs.each.with_index { |input, i|
  puts format("%s = %s", input.join(', '), outputs[i].value.inspect)
}
puts n

puts "Press Enter to continue"
gets

999.times { |i|
  predictions = inputs.map { |input| n.apply(input) }
  loss = BackProp.mean_squared_error(outputs, predictions)
  puts loss

  # propagate the derivatives (gradients) backwards
  loss.backward

  if i % 100 == 0
    p predictions.map(&:value)
    p n
    puts
    gets
  end

  # nudge all parameters according to the new gradient
  n.parameters.each { |val|
    val.value += -1 * gradient_step * val.gradient
  }
}

p predictions.map(&:value)
p n
