-- Kwavg-test.lua
-- unit tests for class Kwavg

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
end

function tests._determineWeights()
   local nObs = 3
   local nDims = 2
   local xs = torch.Tensor(nObs, nDims)
   xs[1] = torch.Tensor(nDims):fill(1)
   xs[2] = torch.Tensor(nDims):fill(2)
   xs[3] = torch.Tensor(nDims):fill(3)

   local kwavg = Kwavg(xs, torch.Tensor(nObs), 'epanechnikov quadratic')

   local query = torch.Tensor(nDims):fill(0)
   local lambda = 2
   local weights = kwavg:_determineWeights(query, lambda)

   local tol = 1e-6
   tester:asserteq(3, weights:size(1))
   tester:assert(math.abs(0.3750 - weights[1]) < tol)
   tester:asserteq(0, weights[2])
   tester:asserteq(0, weights[3])

   query = torch.Tensor(nDims):fill(2)
   weights = kwavg:_determineWeights(query, lambda)

   tester:asserteq(3, weights:size(1))
   tester:assert(math.abs(0.3750 - weights[1]) < tol)
   tester:assert(math.abs(0.75 - weights[2]) < tol)
   tester:assert(math.abs(0.3750 - weights[3]) < tol)
end -- _determineWeights

function tests._weightedAverage()
   local nObs = 3
   local ys = torch.Tensor(nObs)
   ys[1] = 1
   ys[2] = 2
   ys[3] = 3
   
   local weights = torch.Tensor(nObs)
   weights[1] = 0
   weights[2] = 20
   weights[3] = 10
   
   local kwavg = Kwavg(torch.Tensor(nObs, 2), ys, 'epanechnikov quadratic')
   local tol = 1e-6
   local ok, estimate =  kwavg:_weightedAverage(weights)
   tester:assert(ok)
   tester:assert(math.abs(2.333333333 - estimate) < tol)

   weights = torch.Tensor(nObs):fill(0)
   local ok, estimate = kwavg:_weightedAverage(weights)
   tester:assert(not ok)
   tester:asserteq(estimate, 'all weights used were 0')
end -- _weightedAverage

function tests.estimate()
   local v = makeVerbose(false, 'test.estimate')
   local trace = false
   local nsamples, ndims, xs, ys = makeExample()
   local query = torch.Tensor(ndims):zero()

   -- for calculations, see lab book date 2012-08-23
   -- lambda:             1  2  3       4       5       6
   local expectedSeq = {0/0, 1, 1, 1.2353, 1.3714, 1.6364}
   for lambda = 1, 6 do
      local kwavg = Kwavg(xs, ys, 'epanechnikov quadratic')
      v('lambda', lambda)
      local ok, actual = kwavg:estimate(query, lambda)
      v('ok', ok)
      v('actual', actual)
      if lambda == 1 then
         tester:assert(not ok, 'lambda=' .. lambda)
      else
         tester:assert(ok, 'lambda=' .. lambda)
         local expected = expectedSeq[lambda]
         if trace then
            print('lambda, actual, expected', lambda, actual, expected)
         end
         local tolerance = 0.0001
         tester:assert(math.abs(expected - actual) < tolerance, 
                          'lambda=' .. lambda)
      end
   end
end

-- test smoothing without using the query point
function tests.smooth1()
   local trace = false
   local nsamples, ndims, xs, ys = makeExample()
   local query = torch.Tensor(ndims):zero()
   
   -- for calculations, see lab book date 2012-08-23
   -- i:                  1  2  3       4
   local expectedSeq = {0/0, 2, 3, 3.9999}
   for i = 1, 4 do
      local kwavg = Kwavg(xs, ys, 'epanechnikov quadratic')
      local errorIfZeroSumWeights = false
      local useQueryPoint = false
      local ok, actual = kwavg:smooth(i, i, useQueryPoint)
      if i == 1 then
         tester:assert(not ok, 'i=' .. i)
      else
         tester:assert(ok, 'i=' .. i)
         local expected = expectedSeq[i]
         if trace then 
            print('i, actual, expected', i, actual, expected)
         end
         local tolerance = 0.0002
         tester:assert(math.abs(expected - actual) < tolerance, 
                       'i=' .. i)
      end
   end
end

-- test smoothing using the query point
function tests.smooth2()
   local trace = false
   local nsamples, ndims, xs, ys = makeExample()
   local query = torch.Tensor(ndims):zero()
   
   -- for calculations, see lab book date 2012-08-23
   local function test(queryIndex, lambda, expected)
      local kwavg = Kwavg(xs, ys, 'epanechnikov quadratic')
      local useQueryPoint = true
      local ok, actual = kwavg:smooth(queryIndex, lambda, useQueryPoint)
      tester:assert(ok, 'no error')
      if trace then
         print('queryIndex, lambda, actual, expected', 
               queryIndex, lambda, actual, expected)
      end
      local tolerance = 0.0001
      tester:assert(math.abs(expected - actual) < tolerance, 
                    'queryIndex=' .. queryIndex .. ' lambda=' .. lambda)
      end

   test(3, 0.4, 3.0)
   test(1, 3, 1.4)
end


-- run unit tests
if false then
   --tester:add(tests.estimate, 'test.estimate')
   --tester:add(tests.smooth1, 'test.smooth1')
   tester:add(tests.smooth2, 'test.smooth2')
else
   tester:add(tests)
end
tester:run(true) -- true ==> verbose



   
