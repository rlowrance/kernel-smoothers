-- Nncachebuilder-test.lua
-- unit test

require 'all'

setRandomSeeds(27)

tester = Tester()
test = {}

function test.integrated()
   local v = makeVerbose(false, 'test.integrated')
   local nObs = 300
   local nDims = 10
   local xs = torch.rand(nObs, nDims)
   local nShards = 5

   local nnc = Nncachebuilder(xs, nShards)
   tester:assert(nnc ~= nil)

   local filePathPrefix = '/tmp/Nncache-test'
   for n = 1, nShards do
      nnc:createShard(n, filePathPrefix)
   end

   nnc:mergeShards(filePathPrefix)

   local cache = Nncachebuilder.read(filePathPrefix)
   --print('cache', cache)
   --print('type(cache)', type(cache))
   v('cache', cache)
   tester:assert(check.isTable(cache))
   local count = 0
   for key, value in pairs(cache) do
      count = count + 1
      tester:assert(check.isIntegerPositive(key))
      tester:assert(check.isTensor1D(value))
      tester:asserteq(math.min(nObs,256), value:size(1))
   end
   tester:asserteq(nObs, count)
end -- test.integrated

print('**********************************************************************')
tester:add(test)
tester:run(true)  -- true ==> verbose
