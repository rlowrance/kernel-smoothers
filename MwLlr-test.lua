-- MwLlr-test.lua
-- unit test

require 'all'

test = {}
tester = Tester()

function test.one()
   -- this is a very weak tests, it checks for completion
   -- figuring out a problem to solve by hand seems complicated
   local v = makeVerbose(true, 'test.one')
   local nObs = 3
   local nDims = 2
   local xs = torch.Tensor(nObs, nDims)
   local ys = torch.Tensor(nObs)
   for i = 1, nObs do
      ys[i] = 100 * i
      for d = 1, nDims do
         xs[i][d] = 10 * i + d
      end
   end

   local llr = MwLlr(xs, ys, 'epanechnikov quadratic')
   
   local query = torch.Tensor(nDims)
   query[1] = 1
   query[2] = 2

   local lambda = 50
   local ok, estimate = llr:estimate(query, lambda)
   v('estimate', estimate)
   tester:assert(ok)
   tester:assertgt(estimate, 0)
end -- test.one


tester:add(test)
tester:run(true) -- true ==> verbose