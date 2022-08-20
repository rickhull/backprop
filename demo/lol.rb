require 'backprop'

include BackProp

a = v(2, label: :a)
b = v(-3, label: :b)
c = v(10, label: :c)
e = a * b; e.label = :e
d = e + c; d.label = :d
f = v(-2, label: :f)
l = d * f; l.label = :L

p l

l.backward

p l

if false

l.gradient = 1.0
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
end
