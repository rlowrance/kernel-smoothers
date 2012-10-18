-- Knn.lua
-- k nearest neighbors algorithm in two variants

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   -- xs are the inputs, a 2D Tensor
   -- ys are the targets, a 1D Tensor

   -- smooth existing values (as during cross validation)
   cache = Nncachebuilder.read('path/to/cache file')
   knn = KnnSmoother(allXs, allYs, selected, cache)
   -- re-estimate observation in allXs 
   -- the re-estimated observation must have been selected
   ok, estimate = knn(obsIndex, k)         -- re-estimate allXs[obsIndex]

   -- estimate new query (as after cross validation has found best k)
   knn = KnnEstimator(xs, ys)
   ok, estimate = knn(query, k)            -- estimate a new query
end -- API overview

--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

torch.class('KnnEstimator')
torch.class('KnnSmoother')

function KnnEstimator:__init(xs, ys)
   local v, isVerbose = makeVerbose(false, 'KnnEstimator:__init')
   verify(v, isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {ys, 'ys', 'isTensor1D'}})

   assert(xs:size(1) == ys:size(1))

   self._xs = xs
   self._ys = ys
end -- KnnEstimator:__init

function KnnSmoother:__init(allXs, allYs, selected, cache) 
   -- ARGS:
   -- xs            : 2D Tensor
   --                 the i-th input sample is xs[i]
   -- ys            : 1D Tensor
   --                 y[i] is the known value (target) of input sample xs[i]
   --                 number of ys must equal number of rows in xs 
   -- selected      : 1D tensor of {0,1} values
   --                 the only values used have selected[i] == 1
   -- cache         : table possible create using Nncachebuilder
   --                 cache[obsIndex] = 1D tensor of indices in allXs of
   --                 256 nearest neighbors to allXs[obsIndex]

   local v, isVerbose = makeVerbose(false, 'KnnSmoother:__init')
   verify(v, 
          isVerbose,
          {{allXs, 'allXs', 'isTensor2D'},
           {allYs, 'allYs', 'isTensor1D'},
           {selected, 'selected', 'isTensor1D'},
           {cache, 'cache', 'isTable'}})
   local nObs = allXs:size(1)
   assert(nObs == allYs:size(1))
   assert(nObs == selected:size(1))

   -- check that selected is correctly structured
   for i = 1, selected:size(1) do
      local value = selected[i]
      affirm.isIntegerNonNegative(value, 'value')
      assert(value <= nObs)
      assert(value == 0 or value == 1)
   end
   
   -- check that cache is correctly structured
   for key, value in pairs(cache) do
      affirm.isIntegerPositive(key, 'key')
      affirm.isTensor1D(value, 'value')
   end

   self._allXs = allXs
   self._allYs = allYs
   self._selected = selected
   self._cache = cache

   self._kMax = Nncachebuilder._maxNeighbors()
end -- __init

-----------------------------------------------------------------------------
-- PUBLIC METHODS
-----------------------------------------------------------------------------

function KnnEstimator:estimate(query, k)
   -- estimate y for a new query point using the Euclidean distance
   -- ARGS:
   -- query : 1D Tensor
   -- k     : integer >= 1 
   --         there is no limit on k when estimating
   --         but there is such a limit when smoothing
   -- RESULTS:
   -- true, estimate : estimate is the estimate for the query
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  rsult is a string
   -- NOTE: the status result is not really needed for this implemenation
   -- of Knn. However, other kernel-smoothers in this package need it, so
   -- for consistency, its included here.

   local v, isVerbose = makeVerbose(true, 'KnnEstimator:estimate')
   verify(v,
          isVerbose,
          {{query, 'query', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})

   assert(k <= self._xs:size(1))

   local nearestIndices = KernelSmoother.nearestIndices(self._xs,
                                                        query)

   -- determine average of k nearest
   local sum = 0
   for neighborIndex = 1, k do
      local obsIndex = nearestIndices[neighborIndex]
      sum = sum + self._ys[obsIndex]
   end
   
   local result = sum / k
   return true, result
end -- KnnEstimator:estimate

function KnnSmoother:estimate(obsIndex, k)
   local v, isVerbose = makeVerbose(true, 'KnnSmoother:estimate')
   verify(v, isVerbose,
          {{obsIndex, 'obsIndex', 'isIntegerPositive'},
           {k, 'k', 'isIntegerPositive'}})
   
   assert(self._selected[obsIndex] == 1,
         'query obsIndex was not selected')
   assert(k <= Nncachebuilder:_maxNeighbors())

   local query = self._allXs[obsIndex]:clone()
   local nearestIndices = KernelSmoother.nearestIndices(self._allXs,
                                                        query)
   v('nearestIndices', nearestIndices)
   v('self._selected', self._selected)

   -- determine average of k nearest selected neighbors
   -- NOTE: this can fail if we get unlucky
   -- specifically, if k > (256 - 256/nfolds)
   -- where 256 = Nncachebuild._maxNeighbors()
   -- If so, increase value of Nncachebuilder._maxNeighbors() and rerun
   local sum = 0
   local used = 0
   for i = 1, nearestIndices:size(1) do
      local neighborIndex = nearestIndices[i]
      if self._selected[neighborIndex] == 1 then
         used = used + 1
         v('used neighborIndex', neighborIndex)
         sum = sum + self._allYs[neighborIndex]
         if used == k then 
            break
         end
      end
   end
   assert(used == k,
          'not enough pre-computed neighbor indices in cache' ..
          '\nnearestIndices:size(1) = ' .. tostring(nearestIndices:size(1)))

   local result = sum / k
   return true, result
end -- KnnSmoother:estimate
