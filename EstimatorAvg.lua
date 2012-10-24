-- EstimatorAvg.lua
-- estimate value using simple average of k nearest neighbors

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   eavg = EstimatorAvg(xs, ys)
   ok, result = eavg:estimate(query, k)
end -- API overview


--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

local _, parent = torch.class('EstimatorAvg', 'Estimator')

function EstimatorAvg:__init(xs, ys)
   local v, isVerbose = makeVerbose(false, 'EstimatorAvg:__init')
   parent.__init(self, xs, ys)
end -- NnEstimatorAvg:__init

--------------------------------------------------------------------------------
-- PUBLIC METHODS
--------------------------------------------------------------------------------

function EstimatorAvg:estimate(query, k)
   -- estimate y for a new query point using the Euclidean distance
   -- ARGS:
   -- query : 1D Tensor
   -- k     : integer > 0, number of neighbors
   -- RESULTS:
   -- true, estimate : estimate is the estimate for the query
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  reason is a string

   local v, isVerbose = makeVerbose(false, 'EstimatorAvg:estimate')
   verify(v, isVerbose,
          {{query, 'query', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})

   local _, nearestIndices = Nn.nearest(self._xs, query)
   v('nearestIndices', nearestIndices)
   local nObs = self._ys:size(1)

   local visible = torch.Tensor(nObs):fill(1)  -- all observations are visible

   local ok, result = Nn.estimateAvg(self._xs,
                                     self._ys,
                                     nearestIndices,
                                     visible,
                                     k)
   return ok, result
end -- estimate()

