require 'backprop'

include BackProp

# inputs x1, x2
x1 = v 2, label: :x1
x2 = v 0, label: :x2

# weights w1, w2
w1 = v -3, label: :w1
w2 = v 1, label: :w2

# neuron bias
b = v 6.8813735870195432, label: :b

xw1 = x1*w1; xw1.label = :xw1
xw2 = x2*w2; xw2.label = :xw2

sum = xw1 + xw2; sum.label = :sum
n = sum + b; n.label = :n

o = n.tanh; o.label = :o

# now try backprop

o.backward

p o



if false
o.gradient = 1

# do/dn
# d/dx tanh x = 1 - tanh(x)^2

# 1 - o**2

n.gradient = 1 - o.value ** 2

# n = sum + b
sum.gradient = n.gradient
b.gradient = n.gradient

# sum = xw1 + xw2
xw1.gradient = sum.gradient
xw2.gradient = sum.gradient

# xw1 = x1 * w1
x1.gradient = xw1.gradient * w1.value
w1.gradient = xw1.gradient * x1.value

# xw2 = x2 * w2
x2.gradient = xw2.gradient * w2.value
w2.gradient = xw2.gradient * x2.value

p o
end
