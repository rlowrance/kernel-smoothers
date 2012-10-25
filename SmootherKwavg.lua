-- SmootherKwavg.lua
-- estimate value using kernel-weighted average of k nearest neighbors

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   skwavg = SmootherKwavg(allXs, allYs, visible, cache)
   ok, estimate = skwavg:estimate(queryIndex, k)
end -- API overview


--------------------------------------------------------------------------------
-- CONSTRUCTOR
--------------------------------------------------------------------------------

local _, parent = torch.class('SmootherKwavg', 'Smoother')

function SmootherKwavg:__init(allXs, allYs, visible, nncache)
   local v, isVerbose = makeVerbose(false, 'SmootherKwavg:__init')
   verify(v, isVerbose,
          {{allXs, 'allXs', 'isTensor2D'},
           {allYs, 'allYs', 'isTensor1D'},
           {visible, 'visible', 'isTensor1D'},
           {nncache, 'nncache', 'isTable'}})
   assert(torch.typename(nncache) == 'Nncache')
   parent.__init(self, allXs, allYs, visible, nncache)
   v('self', self)
   v('self._nncache', self._nncache)
end -- __init()

--------------------------------------------------------------------------------
-- PUBLIC METHODS
--------------------------------------------------------------------------------

function SmootherKwavg:estimate(obsIndex, k)
   local v, isVerbose = makeVerbose(false, 'SmootherKwavg:estimate')
   verify(v, isVerbose,
          {{obsIndex, 'obsIndex', 'isIntegerPositive'},
           {k, 'k', 'isIntegerPositive'}})
   v('self', self)
   assert(k <= Nncachebuilder:maxNeighbors())

   -- determine distances and lambda
   local nObs = self._visible:size(1)
   local distances = torch.Tensor(nObs):fill(1e100)
   local query = self._allXs[obsIndex]
   local sortedNeighborIndices = self._nncache:getLine(obsIndex)
   assert(sortedNeighborIndices)
   v('sortedNeighborIndices', sortedNeighborIndices)
   local found = 0
   for i = 1, nObs do
      local obsIndex = sortedNeighborIndices[i]
      if self._visible[obsIndex] == 1 then
         local distance= Nn.euclideanDistance(self._allXs[obsIndex], query)
         distances[i] = distance
         v('i,obsIndex,distance', i, obsIndex, distance)
         found = found + 1
         if found == k then
            lambda = distance
            break
         end
      end
   end
   v('labmda', lambda)
   v('distances', distances)

   local weights = Nn.weights(distances, lambda)
   v('weights', weights)

   local ok, estimate = Nn.estimateKwavg(k,
                                         sortedNeighborIndices,
                                         self._visible,
                                         weights,
                                         self._allYs)
   v('ok, estimate', ok, estimate)
   return ok, estimate 
end -- estimate


