module BackProp
  def v(numeric, label: '')
    Value.new(numeric.to_f, label: label)
  end

  def self.tanh(x)
    e2x = Math.exp(2 * x).to_f
    (e2x - 1) / (e2x + 1)
  end

  class Value
    attr_reader :value, :children, :op
    attr_accessor :label, :gradient, :backstep

    def initialize(float, label: '', op: nil, children: [])
      @value = float
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

    def display
      @label.empty? ? @value.to_s : format("%s=%.3f", @label, @value)
    end

    def to_s
      format("%s(value=%.3f gradient=%.3f",
             @label.empty? ? @op || 'Value' : @label, @value, @gradient) +
        (@op.nil? ? '' :
           format(" %s(%s)", @op, @children.map(&:display).join(', '))) + ')'
    end

    def inspect
      @children.empty? ? self.to_s :
        [self.to_s, @children.map(&:inspect).join("\n\t")].join("\n\t")
    end

    def +(other)
      val = Value.new(@value + other.value, children: [self, other], op: :+)
      val.backstep = -> {
        self.gradient += val.gradient
        other.gradient += val.gradient
      }
      val
    end

    def *(other)
      val = Value.new(@value * other.value, children: [self, other], op: :*)
      val.backstep = -> {
        self.gradient += val.gradient * other.value
        other.gradient += val.gradient * self.value
      }
      val
    end

    def -(other)
      self + (other * Value.new(-1))
    end

    def **(other)
      raise("Value is not supported") if other.is_a? Value
      val = Value.new(@value ** other, children: [self], op: :**)
      val.backstep = -> {
        self.gradient += val.gradient * (other * self.value ** (other - 1))
      }
      val
    end

    def /(other)
      self * (other ** -1)
    end

    def tanh
      val = Value.new(BackProp.tanh(@value), children: [self], op: :tanh)
      val.backstep = -> {
        self.gradient += val.gradient * (1 - val.value ** 2)
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

    def backward
      @gradient = 1.0
      self.backprop
    end

    def backprop
      self.backstep.call
      @children.each(&:backprop)
      self
    end
  end
end
