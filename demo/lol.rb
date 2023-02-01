require 'backprop/value'

include BackProp

a = Value.new(2, label: :a)
b = Value.new(-3, label: :b)
c = Value.new(10, label: :c)
e = a * b; e.label = :e
d = e + c; d.label = :d
f = Value.new(-2, label: :f)
l = d * f; l.label = :L

puts "Setup:"
p l
puts


puts "Calculate gradient by hand:"

l.gradient = 1.0

# l = d * f; derivative dl/dd = f; dl/df = d
f.gradient = d.value
d.gradient = f.value


# now c.gradient
# that is dL/dc

# dL/dd is -2
# dd/dc is 1
# by chain rule (multiply): dL/dc is -2 * 1 = -2

c.gradient = d.gradient * l.gradient
e.gradient = d.gradient * l.gradient

# now b.gradient (and a.gradient)
# e = a * b

# dL/da = dL/de * de/da
a.gradient = e.gradient * b.value
b.gradient = e.gradient * a.value

p l
puts

puts "Reset gradients"
l.reset_gradient
p l
puts

puts "Calculate gradient via backward:"

l.backward

p l
