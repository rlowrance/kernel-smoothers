-- Knn-test.lua
-- unit tests for class Knn

require 'all'

tests = {}

tester = Tester()

-- set global variables nsamples, ndims, xs, ys
function makeExample()
   nsamples = 10
   ndims = 3
   xs = torch.Tensor(nsamples, ndims)
   ys = torch.Tensor(nsamples)
   for i = 1, nsamples do
      for d = 1, ndims do
         xs[i][d] = i
         ys[i] = i
      end
   end
   return nsamples, ndims, xs, ys
end

function tests.cache()
   -- see if get same answer with and without cache
   -- uses live data, so slow
   local v = makeVerbose(false, 'tests.cache')

   local log = Log('tmp')
   local nObs, data = readTrainingData('../../data/',
                                       log,
                                       0,   -- no input limit
                                       '1A')
   local k = 15
   local disableCache = true
   local knnUseCache = Knn(k, not disableCache)
   local knnNoCache = Knn(k, disableCache)
   
   -- test for few queryIndices
   for queryIndex = 1, 10 do
      local useQueryPoint = false

      -- using the cache, but there is no entry at first
      local ok, estimateUseCache1, cacheHit1 = 
         knnUseCache:smooth(data.features,
                            data.prices,
                            queryIndex,
                            k,
                            useQueryPoint)
      if not ok then
         error(estimatedUseCache1)
      end
      tester:assert(not cacheHit1)

      -- use the cache and hit the entry
      local ok, estimateUseCache2, cacheHit2 =
         knnUseCache:smooth(data.features,
                            data.prices,
                            queryIndex,
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
          knnNoCache:smooth(data.features,
                            data.prices,
                            queryIndex,
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
   makeExample()
   local query = torch.Tensor(ndims):zero()
   
   local expectedSum = 0
   for k = 1, 10 do
      expectedSum = expectedSum + ys[k]
      local knn = Knn(254)
      local ok, actual = knn:estimate(xs, ys, query, k)
      tester:assert(ok, 'k=' .. k)
      local expected = expectedSum / k
      tester:asserteq(expected, actual, 'k=' .. k)
   end
end

function smooth(queryIndex, k, useQuery)
   assert(queryIndex)
   assert(k)
   assert(useQuery ~= nil)
   local knn = Knn(254)
   local ok, value, hitCache = knn:smooth(xs, ys, queryIndex, k, useQuery)
   tester:assert(ok, 'k=' .. k)
   return value, hitCache
end

-- test smoothing without using the query point
function tests:testSmooth1()
   --if true then return end
   makeExample()

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
   makeExample()

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
   --if true then return end
   local v, trace = makeVerbose(false, 'tests:testSmooth3')

   makeExample()

   local useQuery = true

   -- first try: should get expected value and not use the cache
   local useQuery = true
   local knn = Knn(254)

   local function printCache(msg)
      if not trace then return end
      v('cache', msg)
      for k, v in pairs(knn.cacheSortedIndices) do
         v('key', k)
         v('value', v)
      end
   end

   -- queryIndex == 1 k == 1
   local ok, estimate, hitCache = knn:smooth(xs, ys, 1, 1, useQuery)
   tester:assert(ok, 'works')
   tester:asserteq(1, estimate, 'expected')
   tester:assert(not hitCache, 'no cache on first probe')
   printCache('a')

   -- retry queryIndex == 1 k == 2
   ok, estimate, hitCache = knn:smooth(xs, ys, 1, 2, useQuery)
   tester:assert(ok, 'works')
   tester:assert(1, estimate, 'expected')
   tester:assert(hitCache, 'cache hit on second probe')
   printCache('b')

   -- queryIndex == 2 k == 1
   useQuery = false
   ok, estimate, hitCache = knn:smooth(xs, ys, 2, 1, useQuery)
   tester:assert(ok, 'works')
   tester:assert(1.5, estimate, 'expected')
   tester:assert(not hitCache, 'no cache on first probe')
   printCache('c')

   -- queryIndex == 2 k == 1
   ok, estimate, hitCache = knn:smooth(xs, ys, 2, 1, useQuery)
   tester:assert(ok, 'works')
   tester:assert(1.5, estimate, 'expected')
   tester:assert(hitCache, 'no cache on first probe')
   printCache('d')
 
   local cache = knn.cacheSortedIndices  -- don't do this in production code!
   for k, value in pairs(cache) do
      v('k', k)
      v('v', value)
      tester:assert(k == 1 or k == 2, 'what we queried')
      tester:assert(torch.typename(value) == 'torch.IntTensor', 
                    'expected type')
   end
end

function tests.bugZeroIndexValues()
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

   local knn = Knn(kmax)
   -- following line should generate an error before bug is fixed
   -- Since the test depends on random numbers used to initialize the xs,
   -- It may not always fail
   -- Hence run it 10 times
   for time = 1, 10 do
      local ok, estimate, cacheHit = knn:smooth(xs,
                                                ys,
                                                queryIndex,
                                                k,
                                                useQueryPoint)
   end
   tester:assert(true, 'got this far')
end

-- run unit tests
--tester:add(tests.bugZeroIndexValues, 'test.bugZeroIndexValues')
tester:add(tests)
tester:run(true)  -- true ==> verbose



   
