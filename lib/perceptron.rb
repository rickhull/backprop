require 'backprop'

module BackProp
  def self.mean_squared_error(a1, a2)
    a1.map.with_index { |a, i|
      (a - a2[i]) ** 2
    }.inject(Value.new(0)) { |memo, val| memo + val } / a1.size
  end

  def self.rand_inputs(num_inputs, num_examples, rand_arg)
    Array.new(num_examples) {
      Array.new(num_inputs) { Value.new rand(rand_arg) }
    }
  end

  def self.rand_outputs(num_examples, rand_arg)
    Array.new(num_examples) { Value.new rand(rand_arg) }
  end

  class Neuron
    # available activation functions for Value objects
    ACTIVATION = {
      tanh: :tanh,
      sigmoid: :sigmoid,
      relu: :relu,
    }

    attr_reader :weights, :bias, :activation

    def initialize(input_count, activation: :tanh)
      @weights = Array.new(input_count) { Value.new(rand(-1.0..1.0)) }
      @bias = Value.new(rand(-1.0..1.0))
      @activation = ACTIVATION.fetch(activation)
    end

    def apply(x = 0)
      x = Array.new(@weights.size) { x } if !x.is_a? Enumerable
      sum = @weights.map.with_index { |w, i|
        w * x[i]
      }.inject(Value.new(0)) { |memo, val| memo + val } + @bias
      sum.send(@activation)
    end

    def parameters
      @weights + [@bias]
    end

    def to_s
      format("N(%s)\t(%s %s)", @weights.join(', '), @bias, @activation)
    end

    def inspect
      @weights.map { |wval|
        format("% .3f|% .3f", wval.value, wval.gradient)
      }.join("\t")
    end
  end

  class Layer
    attr_reader :neurons

    def initialize(input_count, output_count, activation: :tanh)
      @neurons = Array.new(output_count) {
        Neuron.new(input_count, activation: activation)
      }
    end

    def apply(x = 0)
      @neurons.map { |n| n.apply(x) }
    end

    def parameters
      @neurons.map { |n| n.parameters }.flatten
    end

    def to_s
      @neurons.join("\n")
    end

    def inspect
      @neurons.map(&:inspect).join("\n")
    end
  end

  class MLP
    attr_reader :layers

    # MLP.new(3, [4, 4, 1])
    def initialize(input_count, output_counts, activation: :tanh)
      flat = [input_count, *output_counts]
      @layers = output_counts.map.with_index { |oc, i|
        Layer.new(flat[i], flat[i+1], activation: activation)
      }
    end

    def apply(x = 0)
      @layers.each { |layer| x = layer.apply(x) }
      x.size == 1 ? x.first : x
    end

    def parameters
      @layers.map { |l| l.parameters }.flatten
    end

    def to_s
      @layers.join("\n\n")
    end

    def inspect
      @layers.map(&:inspect).join("\n\n")
    end
  end
end
