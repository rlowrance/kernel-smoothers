-- knn.lua
-- k nearest neighbors algorithm

require 'affirm'
require 'makeVerbose'

-- API overview
if false then
   -- kmax is the maximum value of k that will be used
   -- it effects the cache size used by the smooth method
   local knn = Knn(kmax)

   -- average of k nearest neighbors using Euclidean distance in xs to query
   -- xs are the inputs
   -- ys are the targets
   local ok, estimate = knn:estimate(xs, ys, query, k)
   if not ok then
      error(estimate)  -- in this case, estimate is a string
   end

   -- re-estimate xs[queryIndex] using k nearest neighbor
   -- exclude ys[queryIndex] from this estimate
   local useQueryPoint = false
   local ok, smoothedEstimate, cacheHit = knn:smooth(xs, ys, queryIndex, k,
                                                     useQueryPoint)
   if not ok then 
      error(smoothedEstimate) -- smoothedEstimate is a string 
   end
end -- API overview

--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

local Knn = torch.class('Knn')

-- maintain a cache of distances from each queryIndex to all other points
-- this can speed up cross validation studies using the smooth method
function Knn:__init(kmax)
   affirm.isIntegerPositive(kmax, 'kmax')
   assert(kmax < 256, 'kmax must fit into 8 unsigned bits')

   self.kmax = kmax
   -- cache the kmax nearest sorted indices to queryIndex in
   -- self.cacheSortedIndices[queryIndex]
   -- NOTE: If the amount of RAM becomes a problem, a short int will suffices
   -- for the storage (or maybe even a byte if kmax <= 256)
   self.cacheSortedIndices = {}
end -- __init

-----------------------------------------------------------------------------
-- PUBLIC METHODS
-----------------------------------------------------------------------------

function Knn:estimate(xs, ys, query, k)
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
   -- NOTE: the status result is not really needed for this implemenation
   -- of Knn. However, other kernel-smoothers in this package need it, so
   -- for consistency, its included here.
   
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, k)
   affirm.isTensor1D(ys, 'ys')

   local distances = self:_determineEuclideanDistances(xs, query)
   local _, sortedIndices = torch.sort(distances)
   
   local useFirstIndex = true
   return true, self:_averageKNearest(sortedIndices, 
                                      ys, 
                                      k,
                                      useFirstIndex)
end -- estimate

function Knn:smooth(xs, ys, queryIndex, k, useQueryPoint)
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
   -- true, estimate, cacheHit : an estimate was produced
   --                            estimate is a number
   --                            cacheHit is true iff 
   --                              queryIndex was in the cache so that the
   --                              Euclidean distances were not computed
   -- false, reason            : no estimate was produced
   --                            reason is a string


   local v = makeVerbose(false, 'Knn:smooth')
   v('queryIndex', queryIndex)
   v('useQueryPoint', useQueryPoint)

   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, k)
   affirm.isIntegerPositive(queryIndex, 'queryIndex')
   affirm.isBoolean(useQueryPoint, 'useQueryPoint')

   assert(queryIndex <= xs:size(1),
          'queryIndex cannot exceed number of samples = ' .. xs:size(1))

   assert(k <= self.kmax, 'k exceeds kmax set at construction time')

   local cacheHit = true
   local sortedIndices = self.cacheSortedIndices[queryIndex]
   v('sortedIndices', sortedIndices)
   if sortedIndices == nil then
      cacheHit = false
      allDistances = 
         self:_determineEuclideanDistances(xs, xs[queryIndex])
      v('allDistances', allDistances)
      local _, allSortedIndices = torch.sort(allDistances)
      v('allSortedIndices', allSortedIndices)
      -- simply resizing via allSortedDistances:resize(kmax) does not shrink
      -- the underlying storage
      -- hence this build and copy operation
      -- fit the values into 8 bits
      -- so use ByteTensor, whose values are unsigned 8 bit values
      local effectiveMax = math.min(self.kmax, allDistances:size(1))
      v('effectiveMax', effectiveMax)
      sortedIndices = torch.ByteTensor(effectiveMax)
      for i = 1, effectiveMax do
         sortedIndices[i] = allSortedIndices[i]
      end
      self.cacheSortedIndices[queryIndex] = sortedIndices
      v('new sortedIndices', sortedIndices)
   end
   
   -- make the distance from the query index to itself large, if
   -- we are not using the query point as a neighbor
   if false and not useQueryPoint then
      -- sortedIndices may be from the cache, in which case, we must clone
      -- it, since we don't want to modify the cache
      -- This was very trick to find and debug
      sortedIndices = sortedIndices:clone()  
      sortedIndices[1] = math.huge  -- the queryIndex must be first
      v('sortedIndices after setting huge', sortedIndices)
   end
   
   v('sortedIndices', sortedIndices)
   local ok, value = true, self:_averageKNearest(sortedIndices, 
                                                 ys, 
                                                 k, 
                                                 useQueryPoint)

   return ok, value, cacheHit
end -- smooth

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

function Knn:_averageKNearest(sortedIndices, ys, k, useFirst)
   -- return average of the k nearest samples
   -- ARGS
   -- sortedIndices : 1D Tensor with indices of neighbors, closest to furthest
   -- ys            : 1D Tensor of prices
   -- k             : Integer > 0, how many neighbors to consider
   -- useFirst      : boolean
   --                 if true, start with 1st index in sortedIndices
   --                 if false, start with 2nd index in sortedIndiex

   local v = makeVerbose(false, 'Knn:_averageKNearest')
   v('sortedIndices', sortedIndices)
   assert(k <= sortedIndices:size(1))

   -- sum the y values for the k nearest neighbors
   local sum = 0

   local index = 1
   if not useFirst then index = 2 end

   local count = 0

   while (count < k) do
      sum = sum + ys[sortedIndices[index]]
      count = count + 1
      index = index + 1
   end

   return sum / k
end

function Knn:_determineEuclideanDistances(xs, query)
   -- return 1D tensor such that result[i] = EuclideanDistance(xs[i], query)
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v = makeVerbose(false, 'Knn:_determineEuclideanDistances')

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
      
      return distances
end

-- verify type and values of certain arguments
function Knn:_typeAndValueCheck(xs, ys, k)
   -- type and value check xs
   affirm.isTensor2D(xs, 'xs')
   affirm.isTensor1D(ys, 'ys')
   affirm.isIntegerPositive(k, 'k')
end
