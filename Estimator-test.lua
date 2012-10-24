-- Estimator-test.lua

require 'all'

tester = Tester()
test = {}

function test.one()
   local xs = torch.rand(3, 4)
   local ys = torch.rand(3)

   e = Estimator(xs, ys)
   tester:assert(e ~= nil)
   tester:asserteq(torch.typename(e), 'Estimator')
end -- one




tester:add(test)
tester:run(true) -- true ==> verbose