-- knn.lua
-- k nearest neighbors algorithm

require 'torch'

-- examples
if false then
   local knn = Knn()

   -- average of k nearest neighbors in xs to query
   local estimate = Knn:estimate(xs, ys, query, k) -- use Euclidean distance

   -- re-estimate xs[queryIndex] using k nearest neighbor
   -- exclude ys[queryIndex] from this estimate
   local smoothedEstimate = Knn:smooth(xs, ys, queryIndex, k)
end

-- create the class object
local Knn = torch.class('Knn')

-----------------------------------------------------------------------------
-- intializer
-----------------------------------------------------------------------------

-- nothing to initialize
function Knn:__init()
end

-----------------------------------------------------------------------------
-- estimate
-----------------------------------------------------------------------------

-- estimate y for a new query point using the Euclidean distance
-- ARGS:
-- xs:    2D Tensor
--        the i-th sample is xs[i]
-- ys:    1D Tensor or array of numbers
--        y[i] is the known value of sample xs[i]
--        number of ys must equal number of xs
-- query: 1D Tensor
-- k    : number, >= 1 
--        math.floor(k) neighbors are considered
--
-- RESULTS:
-- estimate: scalar
--           average y value of the k nearest neighbors in the xs
--           from the query using the Euclidean distance 
function Knn:estimate(xs, ys, query, k)
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, k)

   assert(query:dim() == 1,
          'query must be 1D Tensor')

   local distances = self:_determineEuclideanDistances(xs, query)
   
   return self:_averageKNearest(distances, ys, k)
end

-----------------------------------------------------------------------------
-- smooth
-----------------------------------------------------------------------------

-- re-estimate y for an existing xs[queryIndex]
-- ARGS:
-- xs:         2D Tensor
--             the i-th sample is xs[i]
-- ys:         1D Tensor or array of numbers
--             y[i] is the known value of sample xs[i]
--             number of ys must equal number of xs
-- queryIndex: number >= 1
--             xs[math.floor(queryIndex)] is re-estimated
-- k:          number, >= 1 
--             math.floor(k) neighbors are considered
--
-- RESULTS:
-- estimate: scalar
--           average y value of the k nearest neighbors in the xs
--           from the query using the Euclidean distance 
--
-- NOTES:
-- (1) This all-at-once code is based on Clement Farabet's original design
function Knn:smooth(xs, ys, queryIndex, k)
   local trace = false
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, k)

   assert(type(queryIndex) == 'number',
          'queryIndex must be a number')
   assert(queryIndex >= 1,
          'queryIndex must be at least 1')
   assert(queryIndex <= xs:size(2),
          'queryIndex cannot exceed number of samples (=xs:size(2))')

   local distances = 
      self:_determineEuclideanDistances(xs, xs[math.floor(queryIndex)])
   
   -- make the distance from the query index to itself large
   distances[queryIndex] = math.huge
   local result = self:_averageKNearest(distances, ys, k)

   if trace then
      print('Knn:smooth')
      print('xs\n', xs)
      print('ys\n', ys)
      print('queryIndex', queryIndex)
      print('xs[math.floor(queryIndex)]', xs[math.floor(queryIndex)])
      print('k', k)
      print('result', result)
   end

   return result
end

-----------------------------------------------------------------------------
-- private methods
-----------------------------------------------------------------------------

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

   -- return average of the k nearest neighbors
   return sum / k
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

-- verify type and values of arguments
function Knn:_typeAndValueCheck(xs, ys, k)
   assert(xs,
          'xs must be supplied')
   assert(xs:dim() == 2,
          'xs must be 2D Tensor')

   assert(ys,
          'ys must be supplied')
   assert(ys[1],
          'ys must be subscriptable by one value')
   if type(ys) == 'table' then
      assert(#ys == xs:size(1), 
             'number of ys must equal number of samples')
   else
      assert(ys:size(1) == xs:size(1),
             'number of ys must equal number of samples')
   end

   assert(type(k) == 'number',
          'k must be a number')
   assert(k >= 1,
          'k must be at least 1')

end


