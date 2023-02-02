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

  class Value
    def self.wrap(other)
      other.is_a?(Value) ? other : Value.new(other)
    end

    attr_reader :children
    attr_accessor :value, :label, :gradient, :backstep, :op

    # op implies children, converse, and inverse
    def initialize(float, label: '', op: nil, children: [])
      @value = float.to_f
      @gradient = 0
      @children = children
      if @children.empty?
        raise "op #{op.inspect} has no children" unless op.nil?
      else
        raise "op is required" if op.nil?
      end
      @op = op
      @label = label
      @backstep = -> {}
    end

    # string representations

    def to_s
      @label.empty? ? ("%.3f" % @value) : format("%s=%.3f", @label, @value)
    end

    def display
      format("%s(%.3f gradient=%.3f", self.display_label, @value, @gradient) +
        display_children + ')'
    end

    def display_label
      @label.empty? ? @op || 'Value' : @label
    end

    def display_children
      (@op.nil? ? '' : format(" %s(%s)", @op, @children.join(', ')))
    end

    def inspect
      @children.empty? ? self.display :
        [self.display, @children.map(&:inspect).join("\n\t")].join("\n\t")
    end

    #
    # Primary operations; notice every Value.new(op:) also defines a backstep
    #   The backstep closes over the environment of the method so it can
    #   refer to values present when the method executes
    #

    def +(other)
      other = Value.wrap(other)
      val = Value.new(@value + other.value, children: [self, other], op: :+)

      # What we're about to do here is pretty twisted.  We're going to refer
      # to this execution context in the definition of a lambda, but we'll
      # evaluate it later.
      # Backstep is a lambda attached to val, which will be the return value
      # here. When val.backstep is called later, it will update the gradients
      # on both self and other.
      val.backstep = -> {
        # gradients accumulate for handling a term used more than once
        # chain rule says to multiply val's gradient and the op's derivative
        # derivative of addition is 1.0; pass val's gradient to children
        self.gradient += val.gradient
        other.gradient += val.gradient
      }
      val
    end

    def *(other)
      other = Value.wrap(other)
      val = Value.new(@value * other.value, children: [self, other], op: :*)
      val.backstep = -> {
        # derivative of multiplication is the opposite term
        self.gradient += val.gradient * other.value
        other.gradient += val.gradient * self.value
      }
      val
    end

    # Mostly we are squaring(2) or dividing(-1)
    # We don't support expressions, so Value is not supported for other
    # This will look like a unary op in the tree
    def **(other)
      raise("Value is not supported") if other.is_a? Value
      val = Value.new(@value ** other, children: [self], op: :**)
      val.backstep = -> {
        # accumulate, chain rule, derivative; as before
        self.gradient += val.gradient * (other * self.value ** (other - 1))
      }
      val
    end

    # e^x - unary operation
    def exp
      val = Value.new(Math.exp(@value), children: [self], op: :exp)
      val.backstep = -> {
        self.gradient += val.gradient * val.value
      }
      val
    end

    #
    # Secondary operations defined in terms of primary
    # These return differentiable Values but with more steps
    #

    def -(other)
      self + (Value.wrap(other) * Value.new(-1))
    end

    def /(other)
      self * (Value.wrap(other) ** -1)
    end

    #
    # Activation functions
    # Unary operations
    #

    def tanh
      val = Value.new(Math.tanh(@value), children: [self], op: :tanh)
      val.backstep = -> {
        self.gradient += val.gradient * (1 - val.value ** 2)
      }
      val
    end

    # 1 / 1 + e^-x
    def sigmoid
      ((self * -1).exp + 1) ** -1
    end

    # rectified linear unit; not susceptible to vanishing gradient like above
    def relu
      neg = @value < 0
      val = Value.new(neg ? 0 : @value, children: [self], op: :relu)
      val.backstep = -> {
        self.gradient += val.gradient * (neg ? 0 : 1)
      }
      val
    end

    #
    # Backward propagation
    #

    # Generally, this is called on the final output, say of a loss function
    # It will initialize the gradients and then update the gradients on
    # all dependent Values via back propagation
    def backward
      self.reset_gradient # set gradient to zero on all descendants
      @gradient = 1.0     # this node's gradient is 1.0
      self.backprop       # call backstep on all descendants
    end

    # recursive call; visits all descendants; sets gradient to zero
    def reset_gradient
      @gradient = 0.0
      @children.each(&:reset_gradient)
      self
    end

    # recursive call; visits all descendants; updates gradients via backstep
    def backprop
      self.backstep.call
      @children.each(&:backprop)
      self
    end

    def descend(step_size = 0.1)
      @value += -1 * step_size * @gradient
      self
    end

    def descend_recursive(step_size = 0.1)
      self.descend(step_size)
      @children.each { |c| c.descend_recursive(step_size) }
      self
    end
  end
end
