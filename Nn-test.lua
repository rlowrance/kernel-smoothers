-- Knn-test.lua
-- unit tests for class KnnEstimator and KnnSmoother

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
         ys[i] = i * 10
      end
   end
   return nsamples, ndims, xs, ys
end -- makeExample

function tests.estimator()
   local v = makeVerbose(false, 'tests.estimator')
   local nSamples, nDims, xs, ys = makeExample()
   local query = torch.Tensor(nDims):fill(3)

   local function test(k, expectedSimpleAverage, expectedWeightedAverage)
      -- simple average
      --v('xs', xs)
      local knn = NnAvgEstimator(xs, ys)
      local ok, estimate = knn:estimate(query, k)
      v('ok,estimate', ok, estimate)
      tester:assert(ok)
      tester:asserteq(expectedSimpleAverage, estimate)

      -- weighted average
      if true then return end
      if expectedWeightedAverage then
         local knn = KnnEstimatorKwavg(xs, ys)
         local ok, estimate = knn:estimate(query, k)
         local tol = 1e-3
         if expectedWeightedAverage then
            tester:assert(ok)
            tester:assertle(math.abs(expectedWeightedAverage - estimate), tol)
         else
            tester:assert(not ok)
         end
      end
   end

   -- see lab book for 2012-10-18 for calculations
   test(1, 30, false)
   test(3, 30, 30)
   test(5, 30, 30) 
   test(6, 35, 30)    
   test(7, 40, 32.7277) 
   test(8, 45)
   test(9, 50)
   test(10, 55)
end -- KnnEstimator

function tests.smoother()
   --if true then return end
   local v, isVerbose = makeVerbose(true, 'tests.smoother')
   local nSamples, nDims, xs, ys = makeExample()
   
   -- build up the nearest neighbors cache
   local nShards = 1
   local nncb = Nncachebuilder(xs, nShards)
   local filePathPrefix = '/tmp/Nn-test-cache-'
   nncb:createShard(1, filePathPrefix)
   Nncachebuilder.mergeShards(nShards, filePathPrefix)
   local cache = Nncache.loadUsingPrefix(filePathPrefix)

   v('cache', cache)

   local function p(key, value)
      print(string.format('cache[%d] = %s', key, tostring(value)))
   end
     
   if isVerbose then
      cache:apply(p)
   end
   
   local selector = torch.ByteTensor(nSamples):fill(0)
   for i = 1, nSamples / 2 do
      selector[i] = 1
   end
   v('selector', selector)
      
   v('xs', xs)
   
   local queryIndex = 5

   local function test(k, expectedSimpleAverage, expectedKwavg)
      -- test KnnSmootherAvg
      local knn = NnAvgSmoother(xs, ys, selector, cache)
      local ok, estimate = knn:estimate(queryIndex, k)
      tester:assert(ok)
      tester:asserteq(expectedSimpleAverage, estimate)

      -- test KnnSmootherKwavg
      if true then return end
      if expectedKwavg then
         local knn = KnnSmootherKwavg(xs, ys, selector, cache)
         local ok, estimate = knn:estimate(queryIndex, k)
         tester:assert(ok, 'not ok; error message = ' .. tostring(estimate))
         local tol = 1e-2
         v('estimate', estimate)
         tester:assertlt(math.abs(expectedKwavg - estimate), tol)
      end
   end

   -- hand calculation for expectedKwavg are in lab book 2012-10-20
   if true then
   test(1, 50)  -- no kwavg estimate for k = 1
   test(2, 45, 50) 
   test(3, 40, 45.7143) 
   test(4, 35, 41.8167)
   end
   test(5, 30, 38.0008)
end -- KnnSmoother

-- run unit tests
print('*********************************************************************')
if false then
   --tester:add(tests.bugZeroIndexValues, 'tests.bugZeroIndexValues')
   tester:add(tests._euclideanDistances, 'tests._euclideanDistances')
   print('STUB: did not run all unit tests')
else
   tester:add(tests)
end
tester:run(true)  -- true ==> verbose



   
