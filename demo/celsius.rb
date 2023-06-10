require 'backprop'

include BackProp

deg_f = 68.5

puts "Convert #{deg_f} to celsius using arithmetic (F - 32 * 5/9):"
puts

f = Value.new(68.5, label: :f)
c = (f - 32) * 5 / 9; c.label = :c

puts "f = Value.new(68.5, label: :f)"
puts "c = (f - 32) * 5 / 9; c.label = :c"
puts
p c
