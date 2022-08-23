module BackProp
  class Value
    def self.wrap(other)
      other.is_a?(Value) ? other : Value.new(other)
    end

    attr_reader :children
    attr_accessor :value, :label, :gradient, :backstep, :op

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

    def to_s
      @label.empty? ? ("%.3f" % @value) : format("%s=%.3f", @label, @value)
    end

    def display
      format("%s(%.3f gradient=%.3f",
             @label.empty? ? @op || 'Value' : @label, @value, @gradient) +
        (@op.nil? ? '' :
           format(" %s(%s)", @op, @children.join(', '))) + ')'
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
      val.backstep = -> {
        # gradients accumulate to handle a value used multiple times
        self.gradient += val.gradient
        other.gradient += val.gradient
      }
      val
    end

    def *(other)
      other = Value.wrap(other)
      val = Value.new(@value * other.value, children: [self, other], op: :*)
      val.backstep = -> {
        self.gradient += val.gradient * other.value
        other.gradient += val.gradient * self.value
      }
      val
    end

    # Mostly we are squaring(2) or dividing(-1)
    def **(other)
      raise("Value is not supported") if other.is_a? Value
      val = Value.new(@value ** other, children: [self], op: :**)
      val.backstep = -> {
        self.gradient += val.gradient * (other * self.value ** (other - 1))
      }
      val
    end

    def exp
      val = Value.new(Math.exp(@value), children: [self], op: :exp)
      val.backstep = -> {
        self.gradient += val.gradient * val.value
      }
      val
    end

    #
    # Secondary operations defined in terms of primary
    #

    def -(other)
      self + (Value.wrap(other) * Value.new(-1))
    end

    def /(other)
      self * (Value.wrap(other) ** -1)
    end

    #
    # Activation functions
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

    def backward
      self.reset_gradient
      @gradient = 1.0
      self.backprop
    end

    def reset_gradient
      @gradient = 0.0
      @children.each(&:reset_gradient)
      self
    end

    def backprop
      self.backstep.call
      @children.each(&:backprop)
      self
    end
  end
end
