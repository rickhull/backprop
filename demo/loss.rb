require 'perceptron'

include BackProp

num_inputs = 3
num_examples = 9
net_structure = [4, 4, 1]

# binary classifier; 9 sets of inputs that map to 1 or 0
inputs = BackProp.rand_inputs(num_inputs, num_examples, (-1.0..1.0))
outputs = BackProp.rand_outputs(num_examples, 2)

n = MLP.new(num_inputs, net_structure)
puts n

preds = inputs.map { |input| n.apply(input) }
loss = BackProp.mean_squared_error(outputs, preds)
puts loss


loss.backward
p n
