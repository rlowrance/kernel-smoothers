-- knn.lua
-- k nearest neighbors algorithm

require 'affirm'

-- API overview
if false then
   local knn = Knn()

   -- average of k nearest neighbors using Euclidean distance in xs to query
   -- xs are the inputs
   -- ys are the targets
   local ok, estimate = Knn:estimate(xs, ys, query, k)
   if not ok then
      handleError()
   end

   -- re-estimate xs[queryIndex] using k nearest neighbor
   -- exclude ys[queryIndex] from this estimate
   local useQueryPoint = false
   local ok, smoothedEstimate = Knn:smooth(xs, ys, queryIndex, k,
                                           useQueryPoint)
   if not ok then 
      handleError()
   end
end -- API overview

--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

local Knn = torch.class('Knn')

-- nothing to initialize
function Knn:__init()
end

-----------------------------------------------------------------------------
-- PUBLIC METHODS
-----------------------------------------------------------------------------

-- estimate y for a new query point using the Euclidean distance
-- ARGS:
-- xs    : 2D Tensor
--         the i-th input sample is xs[i]
-- ys    : 1D Tensor or array of numbers
--         y[i] is the known value (target) of input sample xs[i]
--         number of ys must equal number of rows in xs
-- query : 1D Tensor
-- k     : number >= 1 
--          math.floor(k) neighbors are considered
-- RESULTS:
-- true, estimate : estimate is the estimate for the query
--                  estimate is a number
-- false, reason  : no estimate was produced
--                  rsult is a string
function Knn:estimate(xs, ys, query, k)
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, k)
   affirm.isTensor1D(ys, 'ys')

   local distances = self:_determineEuclideanDistances(xs, query)
   
   return self:_averageKNearest(distances, ys, k)
end

-- re-estimate y for an existing xs[queryIndex]
-- ARGS:
-- xs            : 2D Tensor
--                 the i-th input sample is xs[i]
-- ys            : 1D Tensor or array of numbers
--                 y[i] is the known value (target) of input sample xs[i]
--                 number of ys must equal number of rows in xs 
-- queryIndex    : number >= 1
--                 xs[math.floor(queryIndex)] is re-estimated
-- k             : number >= 1 
--                 math.floor(k) neighbors are considered
-- useQueryPoint : boolean
--                if true, use the point at queryIndex as a neighbor
--                if false, don't
-- RESULTS:
-- true, estimate : estimate is the estimate for the query
--                  estimate is a number
-- false, reason  : no estimate was produced
--                  rsult is a string
function Knn:smooth(xs, ys, queryIndex, k, useQueryPoint)
   local trace = false
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, k)
   affirm.isIntegerPositive(queryIndex, 'queryIndex')
   affirm.isBoolean(useQueryPoint, 'useQueryPoint')

   assert(queryIndex <= xs:size(1),
          'queryIndex cannot exceed number of samples = ' .. xs:size(1))

   local distances = 
      self:_determineEuclideanDistances(xs, xs[queryIndex])
   
   -- make the distance from the query index to itself large, if
   -- we are not using the query point as a neighbor
   if not useQueryPoint then
      distances[queryIndex] = math.huge
   end

   local ok, value = self:_averageKNearest(distances, ys, k)

   if trace then
      print('Knn:smooth')
      print('xs\n', xs)
      print('ys\n', ys)
      print('queryIndex', queryIndex)
      print('xs[math.floor(queryIndex)]', xs[math.floor(queryIndex)])
      print('k', k)
      print('ok', ok)
      print('value', value)
   end

   return ok, value
end

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

-- return average of the k nearest samples
function Knn:_averageKNearest(distances, ys, k)
   -- determine indices that sort the distances in increasing order
   local _, sortIndices = torch.sort(distances)

   -- sum the y values for the k nearest neighbors
   k = math.floor(k)
   local sum = 0
   for index = 1, k do
      sum = sum + ys[sortIndices[index]]
   end

   -- return average of the k nearest neighbors or an error if k is 0
   if k < 0 then
      return false, 'k, once rounded to an integer, is negative'
   elseif k == 0 then
      return false, 'k, once rounded to an integer, is zero'
   else
      return true, sum / k
   end
end

-- return 1D tensor such that result[i] = EuclideanDistance(xs[i], query)
-- We require use of Euclidean distance so that this code will work.
-- It computes all the distances from the query point at once
-- using Clement Farabet's idea to speed up the computation.
function Knn:_determineEuclideanDistances(xs, query)
   local trace = false

   query = query:clone()  -- in case the query is a view of the xs storage
   
   -- create a 2D Tensor where each row is the query
   -- This construction is space efficient relative to replicating query
   -- queries[i] == query for all i in range
   -- Thanks Clement!
   local queries = 
      torch.Tensor(query:storage(),
                   1,               -- offset
                   xs:size(1), 0,   -- row index offset and stride
                   xs:size(2), 1)   -- col index offset and stride
      
      local distances = torch.add(queries, -1 , xs) -- queries - xs
      distances:cmul(distances)                     -- (queries - xs)^2
      distances = torch.sum(distances, 2):squeeze() -- \sum (queries - xs)^2
      distances = distances:sqrt()                  -- Euclidean distances
      
      if trace then
         print('Knn:_determineEuclideanDistances')
         print('xs\n', xs)
         print('query\n', query)
         print('distances\n', distances)
      end

      return distances
end

-- verify type and values of certain arguments
function Knn:_typeAndValueCheck(xs, ys, k)
   -- type and value check xs
   affirm.isTensor2D(xs, 'xs')
   affirm.isTensor1D(ys, 'ys')
   affirm.isInteger(k, 'k')
end
