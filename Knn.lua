-- knn.lua
-- k nearest neighbors algorithm

require 'torch'

-- examples
if false then
   local knn = Knn()

   -- average of k nearest neighbors in xs to query
   -- xs are the inputs
   -- ys are the targets
   local estimate = Knn:estimate(xs, ys, query, k) -- use Euclidean distance

   -- re-estimate xs[queryIndex] using k nearest neighbor
   -- exclude ys[queryIndex] from this estimate
   loca useQueryPoint = false
   local smoothedEstimate = Knn:smooth(xs, ys, queryIndex, k,
                                      useQueryPoint)
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
-- xs    : 2D Tensor
--         the i-th input sample is xs[i]
-- ys    : 1D Tensor or array of numbers
--         y[i] is the known value (target) of input sample xs[i]
--         number of ys must equal number of rows in xs
-- query : 1D Tensor
-- k     : number >= 1 
--          math.floor(k) neighbors are considered
--
-- RESULTS:
-- estimate : number
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
-- estimate : number
--            average y value of the k nearest neighbors in the xs
--            from the query using the Euclidean distance 
function Knn:smooth(xs, ys, queryIndex, k, useQueryPoint)
   local trace = false
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, k)

   assert(type(queryIndex) == 'number',
          'queryIndex must be a number')
   assert(queryIndex >= 1,
          'queryIndex must be at least 1')
   assert(queryIndex <= xs:size(1),
          'queryIndex cannot exceed number of samples = ' .. xs:size(1))

   assert(type(useQueryPoint) == 'boolean')

   local distances = 
      self:_determineEuclideanDistances(xs, xs[math.floor(queryIndex)])
   
   -- make the distance from the query index to itself large, if
   -- we are not using the query point is a neighbor
   if not useQueryPoint then
      distances[queryIndex] = math.huge
   end

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

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- _averageNearest
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

   -- return average of the k nearest neighbors
   return sum / k
end

--------------------------------------------------------------------------------
-- _determineEuclideanDistances
--------------------------------------------------------------------------------

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

--------------------------------------------------------------------------------
-- _typeAndValueCheck
--------------------------------------------------------------------------------

-- verify type and values of arguments
function Knn:_typeAndValueCheck(xs, ys, k)
   -- type and value check xs
   assert(xs,
          'xs must be supplied')
   assert(string.match(torch.typename(xs), 'torch%..*Tensor'),
          'xs must be a Tensor')
   assert(xs:dim() == 2,
          'xs must be 2D Tensor')

   -- type and value check ys
   assert(ys,
          'ys must be supplied')
   if type(ys) == 'table' then
      assert(#ys == xs:size(1), 
             'number of ys must equal number of xs')
   elseif string.match(torch.typename(ys), 'torch%..*Tensor') then
      assert(ys:dim() == 1, 'ys must be a 1D Tensor')
      assert(ys:size(1) == xs:size(1),
             'number of ys must equal number of xs')
   else
      assert(false, 'ys must be an array or a 1D Tensor')
   end
   assert(ys[1],
          'ys must be subscriptable by one value')

   -- type and value check k
   assert(type(k) == 'number',
          'k must be a number')
   assert(k >= 1,
          'k must be at least 1')

end
