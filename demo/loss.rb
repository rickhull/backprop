require 'perceptron'

include BackProp

num_inputs = 3
num_examples = 9
net_structure = [4, 4, 1]
gradient_step = 0.1
iterations = 999
afn = [:tanh, :sigmoid, :relu].sample

# binary classifier; 9 sets of inputs that map to 1 or 0
inputs = BackProp.rand_inputs(num_inputs, num_examples, (-1.0..1.0))
outputs = BackProp.rand_outputs(num_examples, 2)
predictions = []

n = MLP.new(num_inputs, net_structure, activation: afn)

puts "Training Cases:"
inputs.each.with_index { |input, i|
  puts format("%s = %s", input.join(', '), outputs[i].value.inspect)
}
puts

puts "Neural Net:"
puts n
puts

puts "Press Enter to continue"
gets

999.times { |i|
  # 1. apply inputs to the net to yield predictions
  # 2. calculate the loss
  # 3. backward propagate the gradients
  # 4. adjust every neuron in the direction of minimizing loss

  # 1. apply inputs
  predictions = inputs.map { |input| n.apply(input).first }

  # 2. calculate loss
  loss = BackProp.mean_squared_error(outputs, predictions)
  puts loss

  # 3. propagate the derivatives (gradients) backwards
  loss.backward

  # output every so often
  if i % 100 == 0
    p outputs.map(&:value)
    p predictions.map(&:value)
    puts
    p n
    gets
  end

  # 4. adjust all weights and biases towards minimizing loss function
  n.descend(gradient_step)
}

p outputs.map(&:value)
p predictions.map(&:value)
puts n
p n
