-- SmootherAvg.lua
-- estimate value using simple average of k nearest neighbors

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   sa = SmootherAverage(allXs, allYs, visible, cache)
   ok, estimate = sa:estimate(queryIndex, k)
end -- API overview

--------------------------------------------------------------------------------
-- SmootherAvg
--------------------------------------------------------------------------------

local _, parent = torch.class('SmootherAvg', 'Smoother')

function SmootherAvg:__init(allXs, allYs, visible, nncache) 
   local v, isVerbose = makeVerbose(false, 'SmootherAvg:__init')
   parent.__init(self, allXs, allYs, visible, nncache)
   v('self', self)
end -- __init()


function SmootherAvg:estimate(obsIndex, k)
   local v, isVerbose = makeVerbose(false, 'SmootherAvg:estimate')
   verify(v, isVerbose,
          {{obsIndex, 'obsIndex', 'isIntegerPositive'},
           {k, 'k', 'isIntegerPositive'}})
   
   assert(k <= Nncachebuilder:maxNeighbors())

   v('self._nncache', self._nncache)
   local nearestIndices = self._nncache:getLine(obsIndex)
   assert(nearestIndices)
   v('nearestIndices', nearestIndices)
   v('self._visible', self._visible)
   v('self', self)

   local ok, result = Nn.estimateAvg(self._allXs,
                                     self._allYs,
                                     nearestIndices, 
                                     self._visible,
                                     k)
   --halt()
   return ok, result
end -- estimate
