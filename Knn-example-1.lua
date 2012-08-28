-- Knn-example-1.lua
-- smoothing a bunch of points

require 'Knn'

local nObservations = 100
local nDims = 10

-- initialize to random value drawn from Normal(0,1)
local xs = torch.randn(nObservations, nDims)
local ys = torch.randn(nObservations)

local smoothed = torch.Tensor(nObservations)
local knn = Knn()          -- create instance of Knn; NOTE: no parameters
local useQueryPoint = true -- use the query point to estimate itself

-- return RMSE for k nearest neighbors
local function computeRmse(k, xs, ys)
   local knn = Knn()           -- create instance of Knn; NOTE: no parameters
   local useQueryPoint = false -- don't use the query point to estimate itself
   local sumSquaredErrors = 0
   for queryIndex = 1, nObservations do
      local ok, value = knn:smooth(xs, ys, queryIndex, k, useQueryPoint)
      if not ok then
         -- the estimate could not be provided
         -- value says why
         error('no estimate; queryIndex=' .. queryIndex .. ' reason=' .. value)
      else
         -- the estimate was provided
         -- it's in value
         local estimate = value
         local error = ys[queryIndex] - estimate
         sumSquaredErrors = sumSquaredErrors + error * error
      end
   end
   return math.sqrt(sumSquaredErrors)
end

-- test:
--  increasing k usually reduces the RMSE for a while and then makes it worse
local lowestRmse = math.huge
local lowestK
for k = 1, 40 do
   local rmse = computeRmse(k, xs, ys)
   if rmse < lowestRmse then
      lowestK = k
      lowestRmse = rmse
   end
   print('k, RMSE', k, rmse)
end
print('best k value', lowestK)
