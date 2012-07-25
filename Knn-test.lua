-- knn-test.lua
-- unit tests for class Knn

require 'Knn'

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

function tests.testEstimate()
   local nsamples, ndims, xs, ys = makeExample()
   local query = torch.Tensor(ndims):zero()
   
   local expectedSum = 0
   for k = 1, 10 do
      expectedSum = expectedSum + ys[k]
      local actual = Knn:estimate(xs, ys, query, k)
      local expected = expectedSum / k
      tester:asserteq(expected, actual, 'k=' .. k)
   end
end

function tests:testSmooth()
   local nsamples, ndims, xs, ys = makeExample()

   local function smooth(queryIndex, k)
      return Knn:smooth(xs, ys, queryIndex, k)
   end

   tester:asserteq(2,   smooth(1,1), 'nearest neighbors = [2]')
   tester:asserteq(2.5, smooth(1,2), 'nearest neighbors = [2,3]')
   tester:asserteq(3,   smooth(1,3), 'nearest neighbors = [2,3,4]')

   tester:asserteq((2+4)/2,   smooth(3,2), 
                   'nearest neighbors = [2,4]')
   tester:asserteq((1+2+4+5)/4,   smooth(3,4), 
                   'nearest neighbors = [1.2,4,5]')
   tester:asserteq((55-3)/9, smooth(3,9), 
                   'nearest neighbors = [1,2,4,5,6,7,8,9,10]')

end

-- run unit tests
tester:add(tests)
tester:run()



   
