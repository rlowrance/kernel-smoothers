-- EstimatorLlr.lua
-- estimate value using local linear regression of k nearest neighbors

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   llr = EstimatorLlr(xs, ys)
   
   -- estimate using k nearest neighbors
   ok, estimate = llr:estimate(query, k)
end

--------------------------------------------------------------------------------
-- CONSTRUCTOR
--------------------------------------------------------------------------------

local _, parent = torch.class('EstimatorLlr', 'Estimator')

function EstimatorLlr:__init(xs, ys, kernelName)
   local v, isVerbose = makeVerbose(false, 'EstimatorLlr:__init')
   verify(v, isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {ys, 'ys', 'isTensor1D'},
           {kernelName, 'kernelName', 'isString'}})
   assert(kernelName == 'epanechnikov quadratic',
          'only kernel supported is epanechnikov quadratic')
   parent.__init(self, xs, ys)
end -- __init()

--------------------------------------------------------------------------------
-- PUBLIC METHODS
--------------------------------------------------------------------------------

function EstimatorLlr:estimate(query, params)
   -- estimate y for a new query point using the Euclidean distance
   -- ARGS:
   -- query          : 1D Tensor
   -- params         : table
   --                  params.k : integer > 0, number of neighbors
   --                  params.regularizer : number >= 0
   -- RESULTS:
   -- true, estimate : estimate is the estimate for the query
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  reason is a string explaining why

   local v, isVerbose = makeVerbose(false, 'EstimatorLlr:estimate')
   if isVerbose then print('*******************************************') end
   verify(v, isVerbose,
          {{query, 'query', 'isTensor1D'},
           {params, 'params', 'isTable'}})

   affirm.isIntegerPositive(params.k, 'k')
   affirm.isNumberNonNegative(params.regularizer, 'regularizer')
   local k = params.k

   v('self', self)

   local nObs = self._ys:size(1)
   assert(k <= nObs,
          string.format('k (=%s) exceeds number of observations (=%d)',
                        tostring(k), nObs))
   local sortedDistances, sortedNeighborIndices = Nn.nearest(self._xs,
                                                             query)
   v('sortedDistances', sortedDistances)
   v('sortedNeighborIndices', sortedNeighborIndices)
   
   local lambda = sortedDistances[k]
   local weights = Nn.weights(sortedDistances, lambda)
   v('lambda', lambda)
   v('weights', weights)

   local visible = torch.Tensor(nObs):fill(1)
   local ok, estimate = Nn.estimateLlr(k,
                                       params.regularizer,
                                       sortedNeighborIndices,
                                       visible,
                                       weights,
                                       query,
                                       self._xs,
                                       self._ys)
   v('ok,estimate', ok, estimate)
   return ok, estimate
end -- estimate()

--------------------------------------------------------------------------------
-- PRIVATE METHODS (NONE)
--------------------------------------------------------------------------------
