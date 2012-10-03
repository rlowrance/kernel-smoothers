-- knn.lua
-- k nearest neighbors algorithm

require 'affirm'
require 'makeVerbose'

-- API overview
if false then
   -- xs are the inputs, a 2D Tensor
   -- ys are the targets, a 1D Tensor
   -- kmax is the maximum value of k that will be used
   --      it effects the cache size used by the smooth method
   local knn = Knn(xs, ys, kmax, enableCache)

   -- attributes
   modelXs = knn.xs   -- xs from construction
   modelYs = knn.ys   -- ys from construction

   -- average of k nearest neighbors using Euclidean distance in xs to query
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
function Knn:__init(xs, ys, kmax, enableCache) 
   -- ARGS:
   -- xs            : 2D Tensor
   --                 the i-th input sample is xs[i]
   -- ys            : 1D Tensor or array of numbers
   --                 y[i] is the known value (target) of input sample xs[i]
   --                 number of ys must equal number of rows in xs 
   -- kmax          : integer > 0, maximum k value that we will see
   --                 used in the cache
   --                 must be in interval [1,254]
   -- enableCache   : optional boolean, default true
   --                 for debugging, the cache can be turned off

   local v = makeVerbose(false, 'Knn:__init')

   -- type and value check arguments
   affirm.isTensor2D(xs, 'xs')
   affirm.isTensor1D(ys, 'ys')
   affirm.isIntegerPositive(kmax, 'kmax')

   if enableCache == nil then
      self._enableCache = true
   else
      affirm.isBoolean(enableCache, 'enableCache')
      self._enableCache = enableCache
   end

   self._xs = xs
   self._ys = ys
   self._kmax = kmax

   -- cache nearest sorted indices to a query
   -- used only in the estimate method
   -- key = tostring(query)
   -- value = 1D IntTensor of size kmax + 1 containing the indicies
   --         of the kmax + 1 closest neighbors to the query
   self._cacheEstimate = {}

   -- cache the kmax nearest sorted indices to queryIndex in
   -- used only in the smooth method
   -- key = an index into the xs, the queryIndex in a call to method smooth
   -- value = a 1D IntTensor of size kmax + 1, containing the indices
   --         of the kmax +1 closest neighbors to the key
   self._cacheSmooth = {}

   -- check that the IntTensor value can hold an index for all the xs
   v('kmax', kmax)
   v('xs:size(1)', xs:size(1))
   assert((2^31 - 1) > xs:size(1),
          'xs has too many rows for current implementation')

   -- check that we have enough observations to have kmax neighbors
   assert(kmax + 1 <= xs:size(1),
          string.format('\nkmax (=%d) + 1 exceeds nObs=xs:size(1) (=%d)',
                        kmax, xs:size(1)))
   
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

   local v = makeVerbose(false, 'Knn:estimate')
   v('query', query)
   v('k', k)

   -- type check and value check the arguments
   affirm.isTensor1D(query, 'query')
   affirm.isIntegerPositive(k, 'k')

   function determineSortedIndices()
      -- return sorted indices
      local v = makeVerbose(false, 'determineSortedIndices')
      v('query', query)
      local distances = self:_euclideanDistances(query)
      local _, allSortedIndices = torch.sort(distances)
      local sortedIndices = self:_resizeAllSortedIndices(allSortedIndices)
      v('sortedIndices', sortedIndices)
      return sortedIndices
   end -- determineSortedIndices

   local cacheHit = nil
   local sortedIndices = nil
   if self._enableCache then
      local queryString = tostring(query)
      sortedIndices = self._cacheEstimate[queryString]
      if sortedIndices == nil then
         v('computing sortedIndices')
         cacheHit = false
         sortedIndices = determineSortedIndices()
         self._cacheEstimate[queryString] = sortedIndices
      else
         cacheHit = true
      end
   else
      sortedIndices = determineSortedIndices()
   end
      
   v('sortedIndices', sortedIndices)
         
   local useFirstIndex = true
   return 
      true, 
       self:_averageKNearest(sortedIndices, 
                             k,
                             useFirstIndex),
       cacheHit
end -- estimate

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

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

function Knn:_averageKNearest(sortedIndices, k, useFirst)
   -- return average of the k nearest samples
   -- ARGS
   -- sortedIndices : 1D Tensor with indices of neighbors, closest to furthest
   -- k             : Integer > 0, how many neighbors to consider
   -- useFirst      : boolean
   --                 if true, start with 1st index in sortedIndices
   --                 if false, start with 2nd index in sortedIndiex

   local v = makeVerbose(false, 'Knn:_averageKNearest')
   affirm.isTensor1D(sortedIndices, 'sortedIndices')
   affirm.isIntegerPositive(k, 'k')
   affirm.isBoolean(useFirst, 'useFirst')

   v('self', self)
   v('sortedIndices', sortedIndices)
   v('k', k)
   v('useFirst', useFirst)
   assert(k + 1 <= sortedIndices:size(1))

   -- sum the y values for the k nearest neighbors
   local sum = 0

   local index = 1
   if not useFirst then index = 2 end

   local count = 0

   v('sortedIndices:size(1)', sortedIndices:size(1))
   while (count < k) do
      v('index,sortedIndices[index]', index, sortedIndices[index])
      sum = sum + self._ys[sortedIndices[index]]
      count = count + 1
      index = index + 1
   end

   return sum / k
end -- _averageKNearest

function Knn:_euclideanDistances(query)
   -- return 1D tensor such that result[i] = EuclideanDistance(xs[i], query)
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v, trace = makeVerbose(false, 'Knn:_euclideanDistances')
   local debug = false
   v('self', self)
   v('query', query)
   if debug and trace then 
      for i = 1, query:size(1) do
         print(string.format('query[%d] = %f', i, query[i]))
      end
   end
   affirm.isTensor1D(query, 'query')

   query = query:clone()  -- in case the query is a view of the xs storage
   
   -- create a 2D Tensor where each row is the query
   -- This construction is space efficient relative to replicating query
   -- queries[i] == query for all i in range
   -- Thanks Clement Farabet!
   local queries = 
      torch.Tensor(query:storage(),
                   1,                    -- offset
                   self._xs:size(1), 0,   -- row index offset and stride
                   self._xs:size(2), 1)   -- col index offset and stride
      
   local distances = torch.add(queries, -1 , self._xs) -- queries - xs
   distances:cmul(distances)                          -- (queries - xs)^2
   distances = torch.sum(distances, 2):squeeze() -- \sum (queries - xs)^2
   distances = distances:sqrt()                  -- Euclidean distances
  
   v('distances', distances)
   return distances
end -- _euclideanDistances

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
   -- return new 1D Tensor with first kmax + 1 elements
   -- simply resizing via allSortedDistances:resize(kmax) does not shrink
   -- the underlying storage
   -- hence this build and copy operation
   local v = makeVerbose(false, 'Knn:_resizeAllSortedIndices')
   local sortedIndices = torch.IntTensor(self._kmax + 1):fill(0)
   for i = 1, self._kmax + 1 do
      sortedIndices[i] = allSortedIndices[i]
   end
   v('sortedIndices', sortedIndices)
   return sortedIndices
end -- _resizeAllSortedIndices
