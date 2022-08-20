require 'backprop'

include BackProp

f = v(68.5, label: :f)
c = (f - 32) * 5 / 9; c.label = :c
c.backward

p c
