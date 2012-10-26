-- LLr-test.lua
-- unit test

require 'all'

test = {}
tester = Tester()

function test.one()
   -- this is a very weak tests, it checks for completion
   -- figuring out a problem to solve by hand seems complicated
   local v = makeVerbose(true, 'test.one')
   local nObs = 10
   local nDims = 2
   local xs = torch.Tensor(nObs, nDims)
   local ys = torch.Tensor(nObs)
   for i = 1, nObs do
      ys[i] = 100 * i
      for d = 1, nDims do
         xs[i][d] = 10 * i + d
      end
   end

   -- build the cache
   local nShards = 1
   local nncb = Nncachebuilder(xs, nShards)
   local filePathPrefix = '/tmp/SmootherLlr-test-'
   nncb:createShard(1, filePathPrefix)
   Nncachebuilder.mergeShards(nShards, filePathPrefix)
   local nncache = Nncache.loadUsingPrefix(filePathPrefix)

   local visible = torch.Tensor(nObs):fill(1)

   local llr = SmootherLlr(xs, ys, visible, nncache, 'epanechnikov quadratic')
   
   local queryIndex = 5

   local params = {}
   params.k = 5
   params.regularizer = 1e-7
   local ok, estimate = llr:estimate(queryIndex, params)
   v('estimate', estimate)
   tester:assert(ok)
   tester:assertgt(estimate, 0)
end -- test.one


tester:add(test)
tester:run(true) -- true ==> verbose