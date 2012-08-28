-- Knn-test.lua
-- unit tests for class Knn

require 'Knn'

tests = {}

tester = torch.Tester()

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

function tests.testEstimate()
   makeExample()
   local query = torch.Tensor(ndims):zero()
   
   local expectedSum = 0
   for k = 1, 10 do
      expectedSum = expectedSum + ys[k]
      local knn = Knn
      local ok, actual = knn:estimate(xs, ys, query, k)
      tester:assert(ok, 'k=' .. k)
      local expected = expectedSum / k
      tester:asserteq(expected, actual, 'k=' .. k)
   end

   -- test k = 0
   local ok, message = Knn():estimate(xs, ys, query, 0)
   tester:assert(not ok, 'k=0')
end

function smooth(queryIndex, k, useQuery)
   assert(queryIndex)
   assert(k)
   assert(useQuery ~= nil)
   local knn = Knn()
   local ok, value = knn:smooth(xs, ys, queryIndex, k, useQuery)
   if k == 0 then
      tester:assert(not ok, 'k=' .. k)
   else
      tester:assert(ok, 'k=' .. k)
   end
   return value
end

-- test smoothing without using the query point
function tests:testSmooth1()
   makeExample()

   local useQuery = false
   smooth(1, 0, useQuery) -- should generate an error
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

-- run unit tests
tester:add(tests)
tester:run()



   
