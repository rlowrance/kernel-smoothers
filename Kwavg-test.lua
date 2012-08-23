-- Kwavg-test.lua
-- unit tests for class Kwavg

require 'Kwavg'

tests = {}

tester = torch.Tester()

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

function tests.estimate()
   local trace = false
   local nsamples, ndims, xs, ys = makeExample()
   local query = torch.Tensor(ndims):zero()

   -- for calculations, see lab book date 2012-08-23
   -- lambda:             1  2  3       4       5       6
   local expectedSeq = {0/0, 1, 1, 1.2353, 1.3714, 1.6364}
   for lambda = 1, 6 do
      local kwavg = Kwavg('epanechnikov quadratic')
      local errorIfZeroSumWeights = false
      local actual = kwavg:estimate(xs, ys, query, lambda,
                                    errorIfZeroSumWeights)
      local expected = expectedSeq[lambda]
      if trace then
         print('lambda, actual, expected', lambda, actual, expected)
      end
      if expected ~= expected then -- if expected == nan
         tester:assert(actual ~= actual, 'lambda = 1')
      else
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
      local kwavg = Kwavg('epanechnikov quadratic')
      local errorIfZeroSumWeights = false
      local useQueryPoint = false
      local actual = kwavg:smooth(xs, ys, i, i,
                                  useQueryPoint,
                                  errorIfZeroSumWeights)
      local expected = expectedSeq[i]
      if trace then 
         print('i, actual, expected', i, actual, expected)
      end
      if expected ~= expected then -- if expected == nan
         tester:assert(actual ~= actual, 'i = 1')
      else
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
      local kwavg = Kwavg('epanechnikov quadratic')
      local errorIfZeroSumWeights = false
      local useQueryPoint = true
      local actual = kwavg:smooth(xs, ys, queryIndex, lambda,
                                  useQueryPoint,
                                  errorIfZeroSumWeights)
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
tester:run()



   
