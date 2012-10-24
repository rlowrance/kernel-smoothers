-- EstimatorKwavg.lua
-- estimate value using kernel-weighted average of k nearest neighbors

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   ekwavg = EstimatorKwAvg(xs, ys)

   -- when estimating a brand new query and hence not using the cache
   ok, estimate = ekwavg:estimate(query, k)

   -- when smoothing an existing query and hence using the cached indices:
   ok, estimate = ekwavg:estimate(query, k, visible, sortedNeighborIndices)
end -- API overview


--------------------------------------------------------------------------------
-- CONSTRUCTOR
--------------------------------------------------------------------------------

local _, parent = torch.class('EstimatorKwavg', 'Estimator')

function EstimatorKwavg:__init(xs, ys)
   local v, isVerbose = makeVerbose(true, 'EstimatorKwavg:__init')
   parent.__init(self, xs, ys)
end -- EstimatorKwavg:__init()

--------------------------------------------------------------------------------
-- PUBLIC METHODS
--------------------------------------------------------------------------------

function EstimatorKwavg:estimate(query, k)
   -- estimate y for a new query point using the Euclidean distance
   -- ARGS:
   -- query         : 1D Tensor
   -- k             : integer > 0, number of neighbors
   -- RESULTS:
   -- true, estimate : estimate is the estimate for the query
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  reason is a string explaining why

   local v, isVerbose = makeVerbose(false, 'EstimatorKwavg:estimate')
   verify(v, isVerbose,
          {{query, 'query', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})


   local sortedDistances, sortedNeighborIndices = Nn.nearest(self._xs,
                                                             query)
   v('sortedDistances', sortedDistances)
   v('sortedNeighborIndices', sortedNeighborIndices)
   
   local lambda = sortedDistances[k]
   local weights = Nn.weights(sortedDistances, lambda)
   v('lambda', lambda)
   v('weights', weights)

   local visible = torch.Tensor(self._ys:size(1)):fill(1)
   local ok, estimate = Nn.estimateKwavg(k,
                                         sortedNeighborIndices,
                                         visible,
                                         weights,
                                         self._ys)
   v('ok,estimate', ok, estimate)
   return ok, estimate
end -- estimate()

--------------------------------------------------------------------------------
-- PRIVATE METHODS (NONE)
--------------------------------------------------------------------------------
