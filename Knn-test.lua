-- Knn-test.lua
-- unit tests for class Knn

require 'all'

tests = {}

tester = Tester()

function makeExample()
   local nsamples = 10
   local ndims = 3
   local xs = torch.Tensor(nsamples, ndims)
   local ys = torch.Tensor(nsamples)
   for i = 1, nsamples do
      for d = 1, ndims do
         xs[i][d] = i
         ys[i] = i
      end
   end
   return nsamples, ndims, xs, ys
end -- makeExample


function tests.seeNeighbors()
   if true then return end
   -- check on neighbor indices returned from method smooth
   --if true then return end
   local v = makeVerbose(false, 'tests.seeNeighbors')
   local nSamples, nDims, xs, ys = makeExample()
   local enableCache = true
   local knn = Knn(xs, ys, not enableCache)
   local queryIndex = 3
   local k = 3
   local useQueryPoint = true
   local ok, estimate, cacheHit, nearestIndices = 
      knn:smooth(queryIndex, k, useQueryPoint)
   tester:assert(ok)
   tester:assert(estimate == 3)
   tester:assert(not cacheHit)

   v('nearestIndices', nearestIndices)
   tester:assert(check.isTensor1D(nearestIndices))
   tester:asserteq(3, nearestIndices[1])
   tester:assert(2 == nearestIndices[2] or 4 == nearestIndices[2])
   tester:assert(2 == nearestIndices[3] or 4 == nearestIndices[3])
end -- tests.seeNeighbors
   

function tests.cacheEstimate()
   local v = makeVerbose(true, 'tests.cacheEstimate')

   local nSamples, nDims, xs, ys = makeExample()
   local kmax = 3
   local enableCache = true
   local knn = Knn(xs, ys, enableCache)
   
   local k = 2
   local ok, estimate = knn:estimate(xs[1], k)
   tester:assert(ok)
   tester:asserteq(1.5, estimate)

   local ok, estimate = knn:estimate(xs[1], k)
   tester:assert(ok)
   tester:asserteq(1.5, estimate)

end -- tests.cacheEstimate

function tests.cacheSmooth()
   if true then 
      -- skip tests of method smooth which is deprecated
      return 
   end
   if false then
      print('STUB')
      return
   end
   -- see if get same answer with and without cache
   -- uses live data, so slow
   local v = makeVerbose(false, 'tests.cache')

   local log = Log('tmp')
   local nObs, data = readTrainingData('../../data/',
                                       log,
                                       0,   -- no input limit
                                       '1A')
   v('nObs', nObs)
   v('data', data)

   local k = 15
   local enableCache = true
   v('data.xs typename', torch.typename(data.xs))
   local knnUseCache = Knn(data.features, data.prices, k, enableCache)
   local knnNoCache = Knn(data.features, data.prices, k, not enableCache)
   
   -- test for few queryIndices
   for queryIndex = 1, 10 do
      local useQueryPoint = false

      -- using the cache, but there is no entry at first
      local ok, estimateUseCache1, cacheHit1 = 
         knnUseCache:smooth(queryIndex,
                            k,
                            useQueryPoint)
      if not ok then
         error(estimatedUseCache1)
      end
      tester:assert(not cacheHit1)

      -- use the cache and hit the entry
      local ok, estimateUseCache2, cacheHit2 =
         knnUseCache:smooth(queryIndex,
                            k,
                            useQueryPoint)
       if not ok then
          error(estimatedUseCache2)
       end
       tester:assert(cacheHit2)
       v('estimateUseCache1', estimateUseCache1)
       v('estimateUseCache2', estimateUseCache2)
       tester:asserteq(estimateUseCache1, 
                       estimateUseCache2, 
                       'should be identical')
       
       -- don't use the cache
       local ok, estimateNoCache, cacheHit =
          knnNoCache:smooth(queryIndex,
                            k,
                            useQueryPoint)

       if not ok then
          error(estimateNoCache)
       end
       tester:assert(not cacheHit)
       v('estimateNoCache', estimateNoCache)
       tester:asserteq(estimateUseCache1,
                       estimateNoCache,
                       'should be identical')
   end -- loop of queryIndices
end -- tests.cache   

function tests.testEstimate()
   --if true then return end
   local nSamples, nDims, xs, ys = makeExample()
   local query = torch.Tensor(nDims):zero()
   
   local expectedSum = 0
   for k = 1, 0 do
      expectedSum = expectedSum + ys[k]
      local knn = Knn(xs, ys, 9)
      local ok, actual = knn:estimate(query, k)
      tester:assert(ok, 'k=' .. k)
      local expected = expectedSum / k
      tester:asserteq(expected, actual, 'k=' .. k)
   end
end

function smoothOLD(queryIndex, k, useQuery)
   if true then 
      -- skip tests of method smooth which is deprecated
      return
   end
   assert(queryIndex)
   assert(k)
   assert(useQuery ~= nil)
   local knn = Knn(xs, ys, 254)
   local ok, value, hitCache = knn:smooth(queryIndex, k, useQuery)
   tester:assert(ok, 'k=' .. k)
   return value, hitCache
end -- smoothOLD

-- test smoothing without using the query point
function tests:testSmooth1()
   if true then return end
   local nSamples, nDims, xs, ys = makeExample()
   local knn = Knn(xs, ys, xs:size(1) - 1)
   local function smooth(queryIndex, k, useQuery)
      local ok, value, hitCache = knn:smooth(queryIndex, k, useQuery)
      tester:assert(ok)
      return value
   end
   local useQuery = false
   tester:asserteq(2,   smooth(1, 1, useQuery), 'nearest neighbors = [2]')
   tester:asserteq(2.5, smooth(1, 2, useQuery), 'nearest neighbors = [2,3]')
   tester:asserteq(3,   smooth(1, 3, useQuery), 'nearest neighbors = [2,3,4]')

   tester:asserteq((2+4)/2,   smooth(3, 2, useQuery), 
                   'nearest neighbors = [2,4]')
   tester:asserteq((1+2+4+5)/4,   smooth(3, 4, useQuery), 
                   'nearest neighbors = [1,2,4,5]')
   tester:asserteq((55-3)/9, smooth(3, 9, useQuery), 
                   'nearest neighbors = [1,2,4,5,6,7,8,9,10]')

end

-- test smoothing using the query point
function tests:testSmooth2()
   if true then return end
   local nSamples, nDims, xs, ys = makeExample()
   local knn = Knn(xs, ys, xs:size(1) - 1)
   local function smooth(queryIndex, k, useQuery)
      local ok, value, hitCache = knn:smooth(queryIndex, k, useQuery)
      tester:assert(ok)
      return value
   end
   
   local useQuery = true
   tester:asserteq(1,   smooth(1, 1, useQuery), 'nearest neighbors = [2]')
   tester:asserteq(1.5, smooth(1, 2, useQuery), 'nearest neighbors = [2,3]')
   tester:asserteq(2,   smooth(1, 3, useQuery), 'nearest neighbors = [2,3,4]')

   tester:asserteq((2+3+4)/3,   smooth(3, 3, useQuery), 
                   'nearest neighbors = [2,3,4]')
   tester:asserteq((1+2+3+4+5)/5,   smooth(3, 5, useQuery), 
                   'nearest neighbors = [1,2,3,4,5]')
   tester:asserteq((55-10)/9, smooth(3, 9, useQuery), 
                   'nearest neighbors = [1,2,3,4,5,6,7,8,9]')

end

-- test whether smoothing using the cache
function tests:testSmooth3()
   if true then return end
   --if true then return end
   local v, trace = makeVerbose(false, 'tests:testSmooth3')

   local nSamples, nDims, xs, ys = makeExample()

   local useQuery = true

   -- first try: should get expected value and not use the cache
   local useQuery = true
   local knn = Knn(xs, ys, xs:size(1) - 1)

   local function printCache(msg)
      if not trace then return end
      v('printCache', msg)
      v('knn.cache', knn.cache)
      for key, value in pairs(knn.cache) do
         v('key', key)
         v('value', value)
      end
   end

   -- queryIndex == 1 k == 1
   local ok, estimate, hitCache = knn:smooth(1, 1, useQuery)
   tester:assert(ok, 'works')
   tester:asserteq(1, estimate, 'expected')
   tester:assert(not hitCache, 'no cache on first probe')
   printCache('a')

   -- retry queryIndex == 1 k == 2
   ok, estimate, hitCache = knn:smooth(1, 2, useQuery)
   tester:assert(ok, 'works')
   tester:assert(1, estimate, 'expected')
   tester:assert(hitCache, 'cache hit on second probe')
   printCache('b')

   -- queryIndex == 2 k == 1
   useQuery = false
   ok, estimate, hitCache = knn:smooth(2, 1, useQuery)
   tester:assert(ok, 'works')
   tester:assert(1.5, estimate, 'expected')
   tester:assert(not hitCache, 'no cache on first probe')
   printCache('c')

   -- queryIndex == 2 k == 1
   ok, estimate, hitCache = knn:smooth(2, 1, useQuery)
   tester:assert(ok, 'works')
   tester:assert(1.5, estimate, 'expected')
   tester:assert(hitCache, 'no cache on first probe')
   printCache('d')
 
   local cache = knn._cacheSmooth  -- don't do this in production code!
   v('knn', knn)
   v('cache', cache)
   for k, value in pairs(cache) do
      v('k', k)
      v('v', value)
      tester:assert(k == 1 or k == 2, 'what we queried')
      tester:assert(torch.typename(value) == 'torch.IntTensor', 
                    'expected type')
   end
end

function tests.bugZeroIndexValues()
   if true then return end
   -- sometimes a zero index value is generated in the cache
   -- this cannot happen!
   -- Hyp: the cache indices need to be 32 bits wide, not 8
   local nObs = 300
   local nDims = 3
   local xs = torch.rand(nObs, nDims)
   local ys = torch.rand(nObs)
   local queryIndex = 290
   local k = 10
   local kmax = 11
   local useQueryPoint = false

   local enableCache = true
   local knn = Knn(xs, ys, not enableCache)
   -- following line should generate an error before bug is fixed
   -- Since the test depends on random numbers used to initialize the xs,
   -- It may not always fail
   -- Hence run it 10 times
   for time = 1, 10 do
      local ok, estimate, cacheHit = knn:smooth(queryIndex,
                                                k,
                                                useQueryPoint)
   end
   tester:assert(true, 'got this far')
end

-- run unit tests
if false then
   --tester:add(tests.bugZeroIndexValues, 'tests.bugZeroIndexValues')
   tester:add(tests._euclideanDistances, 'tests._euclideanDistances')
   print('STUB: did not run all unit tests')
else
   tester:add(tests)
end
tester:run(true)  -- true ==> verbose



   
