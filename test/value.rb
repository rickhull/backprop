require 'minitest/autorun'
require 'backprop/value'

include BackProp

describe Value do
  describe "basics" do
    before do
      @flt = 2.3
      @v = Value.new(2.3)
    end

    it "wraps numeric values, primarily floats" do
      expect(@v).must_be_kind_of Value
      expect(@v.value).must_be_kind_of Float
      expect(@v.value).must_equal @flt
    end

    it "has several string representations" do
      expect(@v.to_s).must_be_kind_of String
      expect(@v.display).must_be_kind_of String
      expect(@v.inspect).must_be_kind_of String
    end

    it "creates a tree structure when joined by an operator" do
      expect(@v.children).must_be_empty
      sum = @v + 3
      expect(sum).must_be_kind_of Value
      expect(sum.children).wont_be_empty
      expect(sum.op).must_equal :+
      expect(sum.value).must_be_within_epsilon(@flt + 3)
    end

    it "keeps track of a gradient value, initialized to zero" do
      expect(@v.gradient).must_equal 0
    end
  end

  describe "operations" do
    it "updates the gradient value when used in a calculation" do

    end

    describe "addition" do
      before do
        @a = Value.new(1.0)
        @b = Value.new(2.0)
        @sum = @a + @b
      end

      it "yields a Value" do
        expect(@sum).must_be_kind_of Value
        expect(@sum.value).must_be_within_epsilon 3.0
      end

      it "has a sum parent with _a_ and _b_ as children" do
        expect(@sum.children).must_include @a
        expect(@sum.children).must_include @b
        expect(@sum.op).must_equal :+
      end

      it "updates child gradients upon back propagation" do
        expect(@a.gradient).must_equal 0
        expect(@b.gradient).must_equal 0
        expect(@sum.gradient).must_equal 0

        @sum.backward
        expect(@sum.gradient).must_equal 1  # by definition
        expect(@a.gradient).must_equal 1    # via chain rule for addition
        expect(@b.gradient).must_equal 1    # via chain rule for addition
      end
    end

    describe "multiplication" do
      before do
        @a = Value.new(-1)
        @b = Value.new(2.5)
        @prod = @a * @b
      end

      it "yields a Value" do
        expect(@prod).must_be_kind_of Value
        expect(@prod.value).must_be_within_epsilon(-2.5)
      end

      it "has a prod parent with _a_ and _b_ and children" do
        expect(@prod.children).must_include @a
        expect(@prod.children).must_include @a
        expect(@prod.op).must_equal :*
      end

      it "updates child gradients upon back propagation" do
        expect(@a.gradient).must_equal 0
        expect(@b.gradient).must_equal 0
        expect(@prod.gradient).must_equal 0

        @prod.backward
        expect(@prod.gradient).must_equal 1
        expect(@a.gradient).must_equal @b.value # via chain rule
        expect(@b.gradient).must_equal @a.value # via chain rule
      end
    end

    describe "subtraction" do
      before do
        @a = Value.new(10)
        @b = Value.new(4)
        @diff = @a - @b
      end

      it "combines addition with multiplication for negation" do
        # @a + @b * -1
        expect(@diff.value).must_be_within_epsilon 6.0
        expect(@diff.op).wont_equal :-
        expect(@diff.op).must_equal :+
        expect(@diff.children).must_include @a
        expect(@diff.children).wont_include @b
      end
    end

    describe "pow" do
      before do
        @a = Value.new 2
        @b = 10
        @pow = @a ** @b
      end

      it "does not work with right-side Values" do
        expect { @a ** Value.new(3) }.must_raise
      end

      it "yields a Value" do
        expect(@pow).must_be_kind_of Value
        expect(@pow.value).must_be_within_epsilon 1024.0
      end

      it "has a pow parent without _b_ in children" do
        expect(@pow.children).must_include @a
        expect(@pow.children).wont_include @b
        expect(@pow.op).must_equal :**
      end

      it "updates child gradient upon back propagation" do
        expect(@a.gradient).must_equal 0
        expect(@pow.gradient).must_equal 0

        @pow.backward
        expect(@pow.gradient).must_equal 1
        expect(@a.gradient).must_be_within_epsilon @b * @a.value ** (@b - 1)
      end
    end

    describe "division" do
      before do
        @a = Value.new 19.1
        @b = Value.new 2.3
        @quot = @a / @b
      end

      it "uses pow(-1)" do
        # @a * @b ** -1
        expect(@quot.value).must_be_within_epsilon(19.1 / 2.3)
        expect(@quot.op).wont_equal :/
        expect(@quot.op).must_equal :*
        expect(@quot.children).must_include @a
        expect(@quot.children).wont_include @b
      end
    end

    describe "exp" do
      before do
        @a = Value.new 2.4
        @exp = @a.exp
      end

      it "yields a Value" do
        expect(@exp).must_be_kind_of Value
        expect(@exp.value).must_be_within_epsilon Math.exp(2.4)
      end

      it "has exp parent with _a_ in children" do
        expect(@exp.children).must_include @a
        expect(@exp.op).must_equal :exp
      end

      it "updates child gradient upon back propagation" do
        expect(@a.gradient).must_equal 0
        expect(@exp.gradient).must_equal 0

        @exp.backward
        expect(@exp.gradient).must_equal 1
        expect(@a.gradient).must_equal @exp.value # chain rule / derivative
      end
    end
  end

  describe "activation functions" do
  end

  describe "backward propagation" do
  end
end
