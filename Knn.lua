-- knn.lua
-- k nearest neighbors algorithm

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   -- xs are the inputs, a 2D Tensor
   -- ys are the targets, a 1D Tensor
   local knn = Knn(xs, yy, enableCache)

   -- attributes
   modelXs = knn.xs   -- xs from construction
   modelYs = knn.ys   -- ys from construction

   -- average of k nearest neighbors using Euclidean distance in xs to query
   -- the implementation requires k <= 256
   local ok, estimate, cacheHit = knn:estimate(query, k)
   if not ok then
      error(estimate)  -- in this case, estimate is a string
   end

   -- re-estimate xs[queryIndex] using k nearest neighbor
   -- maybe exclude ys[queryIndex] from this estimate
   local useQueryPoint = false
   local ok, estimate, cacheHit = knn:smooth(queryIndex, k, useQueryPoint)
   if not ok then 
      error(estimate) -- estimate is a string in this case
   end
end -- API overview

--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

local Knn = torch.class('Knn')

-- maintain a cache of distances from each queryIndex to all other points
-- this can speed up cross validation studies using the smooth method
function Knn:__init(xs, ys, enableCache) 
   -- ARGS:
   -- xs            : 2D Tensor
   --                 the i-th input sample is xs[i]
   -- ys            : 1D Tensor or array of numbers
   --                 y[i] is the known value (target) of input sample xs[i]
   --                 number of ys must equal number of rows in xs 
   -- enableCache   : boolean
   --                 if true, create a cache of previous entries, thus
   --                 using more space and less time

   local v, isVerbose = makeVerbose(true, 'Knn:__init')
   verify(v, 
          isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {ys, 'ys', 'isTensor1D'},
           {enableCache, 'enableCache', 'isBoolean'}})

   self._xs = xs
   self._ys = ys
   self._enableCache = enableCache

   self._maxK = 256

   -- cache nearest self._maxK sorted indices to a query
   -- key = tostring(query)
   -- value = 1D IntTensor of size self._kmax containing the indicies
   --         of the self._kmax closest neighbors to the query
   self._cache = {}

   self._kernelSmoother = KernelSmoother()

end -- __init

-----------------------------------------------------------------------------
-- PUBLIC METHODS
-----------------------------------------------------------------------------

function Knn:estimate(query, k)
   -- estimate y for a new query point using the Euclidean distance
   -- ARGS:
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

   local v, isVerbose = makeVerbose(false, 'Knn:estimate')
   verify(v,
          isVerbose,
          {{query, 'query', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})

   assert(k <= self._maxK,
          'this implementation restricts k to <= ' .. tostring(self._maxK))

   function determineSortedIndices()
      -- return sorted indices
      local v = makeVerbose(false, 'determineSortedIndices')
      v('query', query)
         
      local distances = self._kernelSmoother:euclideanDistances(self._xs,
                                                               query)
      local _, allSortedIndices = torch.sort(distances)
      local sortedIndices = self:_resizeAllSortedIndices(allSortedIndices)
      v('sortedIndices', sortedIndices)
      return sortedIndices
   end -- determineSortedIndices

   local cacheHit = nil
   local sortedIndices = nil
   if self._enableCache then
      local queryString = tostring(query)
      sortedIndices = self._cache[queryString]
      if sortedIndices == nil then
         v('computing sortedIndices')
         cacheHit = false
         sortedIndices = determineSortedIndices()
         self._cache[queryString] = sortedIndices
      else
         cacheHit = true
      end
   else
      sortedIndices = determineSortedIndices()
   end
      
   v('sortedIndices', sortedIndices)
         
   return 
      true, 
       self:_averageKNearest(sortedIndices, 
                             k)
end -- estimate

 --[[
function Knn:smooth(queryIndex, k, useQueryPoint)
   -- re-estimate y for an existing xs[queryIndex]
   -- ARGS:
   -- queryIndex    : number >= 1
   --                 xs[math.floor(queryIndex)] is re-estimated
   -- k             : number >= 1 
   --                 math.floor(k) neighbors are considered
   -- useQueryPoint : boolean
   --                if true, use the point at queryIndex as a neighbor
   --                if false, don't
   -- RESULTS:
   -- true, estimate, cacheHit, sortedIndices : an estimate was produced
   --   estimate     : number, the estimate value at the queryIndex
   --   cacheHit     : tri-values, is true iff queryIndex was in the cache
   --                  if nil, the cache was not enabled
   --                  if true, the cache was enabled and the query was found 
   --                     in it
   --                  if false, the cache was enabled and the query was not
   --                     found in it
   --   sortedIndics : 1D Tensor of indices in xs that were neighbors of the
   --                  queryIndex in nearest to furthest order
   --                  the queryIndex itself is among the closest neighbors
   -- false, reason : no estimate was produced
   --   reason        : string explaining the reason

   local v = makeVerbose(false, 'Knn:smooth')

   v('queryIndex', queryIndex)
   v('k', k)
   v('useQueryPoint', useQueryPoint)

   -- type and value check the arguments
   affirm.isIntegerPositive(queryIndex, 'queryIndex')
   affirm.isIntegerPositive(k, 'k')
   affirm.isBoolean(useQueryPoint, 'useQueryPoint')

   assert(queryIndex <= self._xs:size(1),
          'queryIndex cannot exceed number of samples = ' .. self._xs:size(1))

   assert(k <= self._kmax, 'k exceeds kmax set at construction time')

   local cacheHit = true
   local sortedIndices = self._cacheSmooth[queryIndex]
   if sortedIndices == nil or not self._enableCache then
      -- compute sortedIndices from first principles
      v('computing sortedIndices')
      cacheHit = false
      sortedIndices = self:_makeSortedIndices(queryIndex)
      self._cacheSmooth[queryIndex] = sortedIndices
   end
   v('sortedIndices', sortedIndices)
   
   local ok, value = true, self:_averageKNearest(sortedIndices, 
                                                 k, 
                                                 useQueryPoint)

   v('result ok', ok)
   v('result value', value)
   v('result cacheHit', cacheHit)
   return ok, value, cacheHit, sortedIndices
end -- smooth
    --]]
--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

function Knn:_averageKNearest(sortedIndices, k)
   -- return average of the k nearest samples
   -- ARGS
   -- sortedIndices : 1D Tensor with indices of neighbors, closest to furthest
   -- k             : Integer > 0, how many neighbors to consider
   -- useFirst      : boolean
   --                 if true, start with 1st index in sortedIndices
   --                 if false, start with 2nd index in sortedIndiex

   local v, isVerbose = makeVerbose(false, 'Knn:_averageKNearest')
   verify(v,
          isVerbose,
          {{sortedIndices, 'sortedIndices', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})


   v('self', self)

   assert(k <= sortedIndices:size(1))

   -- sum the y values for the k nearest neighbors
   local sum = 0
   for i = 1, k do
      v('index,sortedIndices[index]', index, sortedIndices[index])
      sum = sum + self._ys[sortedIndices[i]]
   end

   return sum / k
end -- _averageKNearest

function Knn:_makeSortedIndices(queryIndex)
   -- return 1D IntTensor of sorted distances from xs[queryIndex] to kmax + 1
   -- nearest neighbors
   local v = makeVerbose(false, 'Knn:_makeSortedIndices')
   affirm.isIntegerPositive(queryIndex, 'queryIndex')

   local allDistances = self:_euclideanDistances(self._xs[queryIndex])
   v('allDistances', allDistances)
   local _, allSortedIndices = torch.sort(allDistances)
   v('allSortedIndices', allSortedIndices)
   sortedIndices = self:_resizeAllSortedIndices(allSortedIndices)
   v('sortedIndices', sortedIndices)
   return sortedIndices
end -- _makeSortedIndices

function Knn:_resizeAllSortedIndices(allSortedIndices)
   -- return new 1D Tensor with first self._maxK elements
   -- simply resizing via allSortedDistances:resize(kmax) does not shrink
   -- the underlying storage
   -- hence this build and copy operation
   local v = makeVerbose(true, 'Knn:_resizeAllSortedIndices')
   v('allSortedIndices', allSortedIndices)
   local sortedIndices = torch.IntTensor(self._maxK):fill(0)
   for i = 1, math.min(allSortedIndices:size(1), self._maxK) do
      sortedIndices[i] = allSortedIndices[i]
   end
   v('sortedIndices', sortedIndices)
   return sortedIndices
end -- _resizeAllSortedIndices
