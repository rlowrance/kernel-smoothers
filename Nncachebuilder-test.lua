-- Nncachebuilder-test.lua
-- unit test

require 'all'

setRandomSeeds(27)

tester = Tester()
test = {}

function test.cache()
   local v = makeVerbose(false, 'test.cache')
   local nObs = 10
   local nDims = 1
   local xs = torch.Tensor(nObs, nDims)
   for i = 1, nObs do
      xs[i][1] = i
   end

   local nShards = 1
   local nncb = Nncachebuilder(xs, nShards)
   local filePathPrefix = '/tmp/Knn-test-cache'
   nncb:createShard(1, filePathPrefix)
   Nncachebuilder.mergeShards(nShards, filePathPrefix)
   local cache = Nncachebuilder.read(filePathPrefix)
   v('cache', cache)
   if isVerbose then
      for key, value in pairs(cache) do
         print(string.format('cache[%d] = %s', key, tostring(value)))
      end
   end

   -- first element should be the obs index
   for i = 1, 10 do
      tester:asserteq(i, cache[i][1])
   end

   -- test last element in cache line
   tester:asserteq(10, cache[1][10])
   tester:asserteq(10, cache[2][10])
   tester:asserteq(10, cache[3][10])
   tester:asserteq(10, cache[4][10])
   tester:asserteq(10, cache[5][10])
   tester:asserteq(1, cache[6][10])
   tester:asserteq(1, cache[7][10])
   tester:asserteq(1, cache[8][10])
   tester:asserteq(1, cache[9][10])
   tester:asserteq(1, cache[10][10])

end -- test.cache
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

   Nncachebuilder.mergeShards(nShards, filePathPrefix)

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
      tester:asserteq(key, value[1]) -- obsIndex always nearest to itself
   end
   tester:asserteq(nObs, count)
end -- test.integrated

print('**********************************************************************')
tester:add(test)
tester:run(true)  -- true ==> verbose
