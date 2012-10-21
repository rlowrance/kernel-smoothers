-- Nn.lua
-- k nearest neighbors algorithm in two variants: Estimator, Smoother
-- with 3 methods: simpleAverage, weightedAverage, localLinearRegression
-- the weighted average and llr methods use a k-nearest neighborhood, not
-- a metric neighborhood (see Hastie-01,  p 167)

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   -- xs are the inputs, a 2D Tensor
   -- ys are the targets, a 1D Tensor

   -- smooth existing values (as during cross validation)
   cache = Nncachebuilder.read(filePathPrefix)

   -- simple average
   knn = NnAvgEstimator(xs, ys)
   ok, estimate = knn:estimate(obsIndex, k)
   
   knn = NnAvgSmoother(allXs, allYs, selected, cache)
   ok, estimate = knn:estimate(obsIndex, k) 

   -- Smoother: weighted average of k nearest neighbors
   knn = NnSmootherKwavg(allXs, allYs, selected, cache, 
                          'epanechnikov quadratic')
   ok, estimate = knn:estimate(obsIndex, k) 

   -- Smoother: regularized local linear regression
   knn = NnSmootherLlr(allXs, allYs, selected, cache, 
                        'epanenchnikov quadratic')
   ok, estimate = knn:estimate(obsIndex, {k, regularizer})

   -- Estimator: simple average of k nearest neighbors

   -- Estimator: weighted average of k nearest neighbors
   knn = NnEstimatorKwavg(allXs, allYs, selected, cache, 
                           'epanechnikov quadratic')
   ok, estimate = knn:estimate(obsIndex, k)

   -- Estimator: local linear regression
   knn = NnEstimatorLlr(allXs, allYs, selected, cache, 
                         'epanechnikov quadratic')
   ok, estimate = knn:estimate(obsIndex, {k, regularizer})
end -- API overview

--------------------------------------------------------------------------------
-- Nn : Auxillary methods (serves same purpose as prior KernelSmoother class)
--------------------------------------------------------------------------------

Nn = {}

function Nn.estimateAvg(k, nearestIndices, selected, allYs)
   -- determine average of k nearest selected neighbors
   -- NOTE: this can fail if we get unlucky
   -- specifically, if k > (256 - 256/nfolds)
   -- where 256 = Nncachebuild._maxNeighbors()
   -- If so, increase value of Nncachebuilder._maxNeighbors() and rerun
   local v, isVerbose = makeVerbose(true, 'Nn.estimateAvg')
   verify(v, isVerbose,
          {{k, 'k', 'isIntegerPositive'},
           {nearestIndices, 'nearestIndices', 'isTensor1D'},
           {selected, 'selected', 'isTensor1D'},
           {allYs, 'allYs', 'isTensor1D'}})

   local sum = 0
   local used = 0
   for i = 1, nearestIndices:size(1) do
      local neighborIndex = nearestIndices[i]
      v('neighborIndex', neighborIndex)
      if selected[neighborIndex] == 1 then
         used = used + 1
         v('used neighborIndex', neighborIndex)
         sum = sum + allYs[neighborIndex]
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
end -- Nn.estimateAvg

function Nn.euclideanDistances(xs, query)
   -- return 1D tensor such that result[i] = EuclideanDistance(xs[i], query)
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v, isVerbose = makeVerbose(false, 'KernelSmoother:euclideanDistances')
   verify(v,
          isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {query, 'query', 'isTensor1D'}})
            
   assert(xs:size(2) == query:size(1),
          'number of columns in xs must equal size of query')

   -- create a 2D Tensor where each row is the query
   -- This construction is space efficient relative to replicating query
   -- queries[i] == query for all i in range
   -- Thanks Clement Farabet!
   local queries = 
      torch.Tensor(query:clone():storage(),-- clone in case query is a row of xs
                   1,                    -- offset
                   xs:size(1), 0,   -- row index offset and stride
                   xs:size(2), 1)   -- col index offset and stride
      
   local distances = torch.add(queries, -1 , xs) -- queries - xs
   distances:cmul(distances)                          -- (queries - xs)^2
   distances = torch.sum(distances, 2):squeeze() -- \sum (queries - xs)^2
   distances = distances:sqrt()                  -- Euclidean distances
  
   v('distances', distances)
   return distances
end -- Nn.euclideanDistances

function Nn.kernels(sortedDistances, lambda)
   -- return values of Epanenchnov kernel using euclidean distance
   local v, isVerbose = makeVerbose(true, 'KernelSmoother.kernels')
   verify(v, isVerbose,
          {{sortedDistances, 'sortedDistances', 'isTensor1D'},
           {lambda, 'lambda', 'isNumberPositive'}})
   local nObs = sortedDistances:size(1)
   local t = sortedDistances / lambda
   local one = torch.Tensor(nObs):fill(1)
   local dt = torch.mul(one - torch.cmul(t, t), 0.75)
   local le = torch.le(torch.abs(t), one):type('torch.DoubleTensor')
   local kernels = torch.cmul(le, dt)
   v('kernels', kernels)
   return kernels
end -- Nn.kernels

function Nn.nearest(xs, query)
   -- find nearest observations to a query
   -- RETURN
   -- sortedDistances : 1D Tensor 
   --                   distances of each xs from query
   -- sortedIndices   : 1D Tensor 
   --                   indices that sort the distances
   local v, isVerbose = makeVerbose(false, 'Nn.nearest')
   verify(v, isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {query, 'query', 'isTensor1D'}})
   local distances = Nn.euclideanDistances(xs, query)
   local sortedDistances, sortedIndices = torch.sort(distances)
   return sortedDistances, sortedIndices
end -- Nn.nearest

--------------------------------------------------------------------------------
-- NnEstimator: parent class of all NnEstimatorXXX classes
--------------------------------------------------------------------------------

torch.class('NnEstimator')

function NnEstimator:__init(xs, ys)
   local v, isVerbose = makeVerbose(false, 'Nn:__init')
   verify(v, isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {ys, 'ys', 'isTensor1D'}})

   assert(xs:size(1) == ys:size(1))

   self._xs = xs
   self._ys = ys
end -- NnEstimator:__init()

--------------------------------------------------------------------------------
-- NnSmoother parent class of all NnSmootherXXX classes
--------------------------------------------------------------------------------

torch.class('NnSmoother')

function NnSmoother:__init(allXs, allYs, selected, cache) 
   -- ARGS:
   -- xs            : 2D Tensor
   --                 the i-th input sample is xs[i]
   -- ys            : 1D Tensor
   --                 y[i] is the known value (target) of input sample xs[i]
   --                 number of ys must equal number of rows in xs 
   -- selected      : 1D tensor of {0,1} values
   --                 the only values used have selected[i] == 1
   -- cache         : Nncache object or nil
   --                 cache[obsIndex] = 1D tensor of indices in allXs of
   --                 256 nearest neighbors to allXs[obsIndex]

   local v, isVerbose = makeVerbose(false, 'NnSmoother:__init')
   verify(v, 
          isVerbose,
          {{allXs, 'allXs', 'isTensor2D'},
           {allYs, 'allYs', 'isTensor1D'},
           {selected, 'selected', 'isTensor1D'}})
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
   
   -- check that cache, if supplied, is an Nncache
   if cache then
      assert(torch.typename(cache) == 'Nncache',
             'bad torch.typename(cache) = ' .. tostring(torch.typename(cache)))
   end

   self._allXs = allXs
   self._allYs = allYs
   self._selected = selected
   self._cache = cache

   self._kMax = Nncachebuilder.maxNeighbors()
end -- NnSmoother:__init()

--------------------------------------------------------------------------------
-- NnAvgEstimator
--------------------------------------------------------------------------------

local _, parent = torch.class('NnAvgEstimator', 'NnEstimator')

function NnAvgEstimator:__init(xs, ys)
   local v, isVerbose = makeVerbose(true, 'NnAvgEstimate:__init')
   verify(v, isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {ys, 'ys', 'isTensor1D'}})
   assert(xs:size(1) == ys:size(1))
   self._xs = xs
   self._ys = ys
   self._selected = torch.Tensor(xs:size(1)):fill(1)
end -- NnEstimatorAvg:__init

function NnAvgEstimator:estimate(query, k)
   -- estimate y for a new query point using the Euclidean distance
   -- ARGS:
   -- query : 1D Tensor
   -- k     : integer > 0, number of neighbors
   -- RESULTS:
   -- true, estimate : estimate is the estimate for the query
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  reason is a string

   local v, isVerbose = makeVerbose(true, 'NnAvgEstimator:estimate')
   verify(v, isVerbose,
          {{query, 'query', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})

   local _, nearestIndices = Nn.nearest(self._xs, query)
   local ok, result = Nn.estimateAvg(k,
                                     nearestIndices,
                                     self._selected,
                                     self._ys)
   return ok, result
end -- NnEstimator:estimate()

if false then
   -- was part of NnAvgEstimate:estimate()
   local v, isVerbose = makeVerbose(false, 'NnEstimatorAvg:estimate')
   verify(v,
          isVerbose,
          {{query, 'query', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})

   local sortedDistances, sortedIndices = 
      Nn.nearest(self._xs, query)

   local sum = 0
   for neighborIndex = 1, k do
      local obsIndex = sortedIndices[neighborIndex]
      sum = sum + self._ys[obsIndex]
   end
   
   local estimate = sum / k
   return true, estimate
end -- NnEstimator:estimate

--------------------------------------------------------------------------------
-- NnAvgSmooth
--------------------------------------------------------------------------------

local _, parent = torch.class('NnAvgSmoother', 'NnSmoother')

function NnAvgSmoother:__init(allXs, allYs, selected, cache) 
   parent.__init(self, allXs, allYs, selected, cache)
end -- NnSmootherAvg:__init()


function NnAvgSmoother:estimate(obsIndex, k)
   local v, isVerbose = makeVerbose(false, 'NnSmootherAvg:estimate')
   verify(v, isVerbose,
          {{obsIndex, 'obsIndex', 'isIntegerPositive'},
           {k, 'k', 'isIntegerPositive'}})
   
   assert(k <= Nncachebuilder:maxNeighbors())

   local nearestIndices = self._cache:getLine(obsIndex)
   assert(nearestIndices)
   v('nearestIndices', nearestIndices)
   v('self._selected', self._selected)

   local ok, result = Nn.estimateAvg(k, 
                                     nearestIndices, 
                                     self._selected, 
                                     self._allYs)
   return ok, result
end -- NnSmootherAvg:estimate


--------------------------------------------------------------------------------
-- NnEstimatorKwavg
--------------------------------------------------------------------------------

local _, parent = torch.class('NnEstimatorKwavg', 'NnEstimator')

function NnEstimatorKwavg:__init(xs, ys)
   parent.__init(self, xs, ys)
end -- NnEstimatorKwavg:__init()


function NnEstimatorKwavg:estimate(query, k)
   local v, isVerbose = makeVerbose(true, 'NnEstimatorKwavg:estimate')
   verify(v, isVerbose,
          {{query, 'query', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})

   local sortedDistances, sortedIndices =
      Nn.nearest(self._xs, query)

   local nObs = sortedDistances:size(1)
   assert(k <= nObs)

   local lambda = sortedDistances[k]
   if lambda == 0 then
      return false, 'kth nearest observation has 0 distance'
   end

   local kernels = Nn.kernels(sortedDistances, lambda)
   v('kernels', kernels)

   local sumYs = 0
   local sumKernels = 0
   for neighborIndex = 1, k do
      local obsIndex = sortedIndices[neighborIndex]
      local kernel = kernels[neighborIndex]
      sumYs = sumYs + self._ys[obsIndex] * kernel
      sumKernels = sumKernels + kernel
   end

   assert(sumKernels ~= 0)

   local estimate = sumYs / sumKernels

   return true, estimate
end -- NnEstimaorKwavg:estimate()

--------------------------------------------------------------------------------
-- NnEstimatorLlr
--------------------------------------------------------------------------------

local _, parent = torch.class('NnEstimatorLlr', 'NnEstimator')

function NnEstimatorLlr:__init(xs, ys)
   parent._init(self, xs, ys)
end -- NnEstimatorLlr:__init()

function NnEstimatorLlr:estimate(query, params)
   local v, isVerbose = makeVerbose(true, 'NnEstimatorLLr:estimate')
   verify(v, isVerbose,
          {{query, 'query', 'isTensor1D'},
           {params, 'params', 'isTable'}})

   local k = params.k
   local regularizer = params.regularizer

   print('STUB: NnEstimatorLlr:estimate')
   return false, 'not implemented'
end -- NnEstimatorLlr:estimate()


--------------------------------------------------------------------------------
-- NnSmootherKwavg
--------------------------------------------------------------------------------

local _, parent = torch.class('NnSmootherKwavg', 'NnSmoother')

function NnSmootherKwavg:__init(allXs, allYs, selected, cache)
   parent.__init(self, allXs, allYs, selected, cache)
end -- NnSmootherKwavg:__init()

function NnSmootherKwavg:estimate(obsIndex, k)
   local v, isVerbose = makeVerbose(false, 'NnSmootherKwavg:estimate')
   verify(v, isVerbose,
          {{obsIndex, 'obsIndex', 'isIntegerPositive'},
           {k, 'k', 'isIntegerPositive'}})
   
   assert(k <= Nncachebuilder:maxNeighbors())

   local nearestIndices = self._cache[obsIndex]
   assert(nearestIndices)
   v('nearestIndices', nearestIndices)
   v('self._selected', self._selected)
   
   local sortedDistances, sortedIndices = 
      Nn.nearest(self._allXs, 
                       self._allXs[obsIndex]:clone())
   v('sortedDistances', sortedDistances)
   v('sortedIndices', sortedIndices)

   local function permuteTensorElements(tensor, permutation)
      -- put self._selected in same order as sortedIndices
      -- ARGS:
      -- tensor : 1D Tensor
      -- permutation : 1D Tensor containing each index 
      -- RETURNS
      -- permuted: 1D Tensor such that permutate[i] = tensor[permutation[i]]
      -- NOTE: this operation should be build into torch.Tensor
      -- TODO: move to external
      local n = tensor:size(1)
      assert(permutation:size(1) == n)
      local permuted = torch.Tensor(n)
      for i = 1, n do
         permuted[i] = tensor[sortedIndices[i]]
      end
      return permuted
   end -- permuteTensorElements

   local selected = permuteTensorElements(self._selected, sortedIndices)
   local selectedDistances = torch.cmul(sortedDistances, selected)
   -- set lambda to distance of the k-th nearest neighbor
   local lambda
   local count = 0
   for i = 1, sortedDistances:size(1) do
      if selected[i] == 1 then
         count = count + 1
         if count == k then
            lambda = sortedDistances[i]
            break
         end
      end
   end
   v('lambda', lambda)
   if lambda == 0 then
      -- determine kernel-weighted average of k nearest selected neighbors
      -- NOTE: this can fail if we get unlucky
      -- specifically, if k > (256 - 256/nfolds)
      -- where 256 = Nncachebuild._maxNeighbors()
      -- If so, increase value of Nncachebuilder._maxNeighbors() and rerun
      
      return false, 'kth nearest observation has distance 0'
   end
   local t = torch.cmul(selected, selectedDistances / lambda)
   v('t', t)
   local ones = torch.Tensor(t:size(1)):fill(1)
   local w = torch.cmul(selected,
                        torch.cmul(torch.le(t, ones):type('torch.DoubleTensor'),
                                   ones - torch.cmul(t, t)) * 0.75)
   v('torch.le(t, ones):type(...)', 
     torch.le(t, ones):type('torch.DoubleTensor'))
   v('ones - torch.cmul(t, t)', ones - torch.cmul(t, t))
   v('w', w) 
   local permutedYs = permuteTensorElements(self._allYs, sortedIndices)
   local sumWYs = torch.sum(torch.cmul(w, permutedYs))
   local estimate = sumWYs / torch.sum(w)
   if true then
      v('selected in sorted order', selected)
      v('selectedDistances', selectedDistances)
      v('cmul', torch.cmul(sortedDistances, selected))
      v('lambda', lambda)
      v('t', t)
      v('ones', ones)
      v('ones - cmul(t,t)', ones - torch.cmul(t,t))
      v('w', w)
      v('self._allYs', self._allYs)
      v('sumWYs', sumWYs)
      v('sum w', torch.sum(w))
      v('estimate', estimate)
      --halt()
   end
   
   return true, estimate
end -- NnSmootherKwavg:estimate()