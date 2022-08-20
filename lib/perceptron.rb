require 'backprop'

module BackProp
  class Neuron
    def self.activation(x, fn = :tanh)
      case fn
      when :tanh
        Math.tanh(x)
      when :sigmoid
        BackProp.sigmoid(x)
      when :ReLU
        raise "not yet"
      else
        raise "unknown activation function: #{fn}"
      end
    end

    attr_reader :weights, :bias, :activation

    def initialize(input_count, activation: :tanh)
      @weights = Array.new(input_count) { Value.new(rand(-1.0..1.0)) }
      @bias = Value.new(rand(-1.0..1.0))
      @activation = :tanh
    end

    def apply(x = 0)
      x = Array.new(@weights.size) { x } if !x.is_a? Enumerable
      sum = @weights.map.with_index { |w, i| w.value * x[i] }.sum + @bias.value
      self.class.activation(sum, @activation)
    end

    def parameters
      @weights + [@bias]
    end

    def to_s
      format("Neuron(%s) (%s %s)",
             @weights.join(', '), @bias, @activation)
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
      x
    end

    def parameters
      @layers.map { |l| l.parameters }.flatten
    end

    def to_s
      @layers.join("\n\n")
    end
  end
end
