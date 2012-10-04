-- Kwavg.lua
-- kernel weighted average using the Epanechnikov Quadratic Kernel
-- ref: Hastie-01 p.166 and following

-- example
if false then
   local kwavg = Kwavg(xs, ys, 'epanechnikov quadratic')

   local estimate = Knn:estimate(query, lambda)

   local useQueryPoint = false
   local smoothedEstimate = Kwavg:smooth(queryIndex, lambda,
                                         useQueryPoint)
end

require 'affirm'
require 'KernelSmoother'
require 'makeVerbose'




--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

-- create class object
torch.class('Kwavg')

function Kwavg:__init(xs, ys, kernelName)
   -- ARGS:
   -- xs         : 2D Tensor
   --              the i-th input sample is xs[i]
   -- ys         : 1D Tensor or array of numbers
   --              y[i] is the known value (target) of input sample xs[i]
   --              number of ys must equal number of rows in xs
   -- kernelName : string
   affirm.isTensor2D(xs, 'xs')
   affirm.isTensor1D(ys, 'ys')
   affirm.isString(kernelName, 'kernelName')

   assert(kernelName == 'epanechnikov quadratic',
          'for now, only kernel supported is epanechnikov quadratic')
   self._xs = xs
   self._ys = ys
   self._kernelName = kernelName
   self._kernelsmoother = KernelSmoother()
end

--------------------------------------------------------------------------------
-- PUBLIC METHODS
--------------------------------------------------------------------------------

function Kwavg:estimate(query, lambda)
   -- estimate y for a new query point using the specified kernel
   -- ARGS:
   -- lambda : number > 0, xs outside of this radius are given 0 weights
   -- RESULTS:
   -- true, estimate : estimate is the estimate for the query
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  reason is a string
   local v = makeVerbose(false, 'Kwavg:estimate')

   v('self', self)
   v('query', query)
   v('lambda', lambda)

   affirm.isTensor1D(query, 'query')
   affirm.isNumberPositive(lambda, 'lambda')

   local weights = self:_determineWeights(query, lambda)
   local ok, estimate = self:_weightedAverage(weights)
   
   v('weights', weights)
   v('ok', ok)
   v('estimate', estimate)

   return ok, estimate
end -- estimate

function Kwavg:smooth(queryIndex, lambda, useQueryPoint)
   -- re-estimate y for an existing xs[queryIndex]
   -- ARGS:
   -- queryIndex    : number >= 1
   --                 xs[math.floor(queryIndex)] is re-estimated
   -- lambda        : number > 0 
   -- useQueryPoint : boolean
   --                 if true, use the point at queryIndex as a neighbor
   --                 if false, don't
   -- RESULTS:
   -- true, estimate : estimate is the estimate at the query index
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  reason is a string
   local v = makeVerbose(false, 'Kwavg:smooth')

   v('queryIndex', queryIndex)
   v('lambda', lambda)
   v('useQueryPoint', useQueryPoint)

   affirm.isIntegerPositive(queryIndex, 'queryIndex')
   affirm.isNumberPositive(lambda, 'lambda')
   affirm.isBoolean(useQueryPoint, 'useQueryPoint')

   assert(queryIndex <= self._xs:size(1),
          'queryIndex cannot exceed number of samples = ' .. 
             tostring(self._xs:size(1)))

   local weights = 
      self:_determineWeights(self._xs[queryIndex], lambda)

   if not useQueryPoint then
      weights[queryIndex] = 0
   end

   local ok, value = self:_weightedAverage(weights)

   v('ok', ok)
   v('value', value)

   return ok, value
end -- smooth

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

function Kwavg:_determineWeights(query, lambda) 
   -- return 1D tensor such that result[i] = Kernel_lambda(query, xs[i])
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v = makeVerbose(false, 'Kwavg:_determineWeights')
   
   v('query', query)
   v('lambda', lambda)

   affirm.isTensor1D(query, 'query')
   affirm.isNumberPositive(lambda, 'lambda')

   local distances = self._kernelsmoother:euclideanDistances(self._xs, query)
   local t = distances / lambda
   
   local one = torch.Tensor(self._xs:size(1)):fill(1)
   local dt = torch.mul(one - torch.cmul(t, t), 0.75)
   local le = torch.le(torch.abs(t), one):type('torch.DoubleTensor')
   local weights = torch.cmul(le, dt)
   
   v('t', t)
   v('dt', dt)
   v('le', le)
   v('weights', weights)
   
   return weights
end -- _determineWeights

function Kwavg:_weightedAverage(weights)
   -- attempt to determine weighted average of the weights and y values
   -- RETURNS
   -- true, weightedAverage : if the weighted average can be determined
   --                         weightedAverage is a number
   -- false, reason         : if the weighted average cannot be determined
   --                         reason is a string          
   local v = makeVerbose(false, 'Kwavg:_weightedAverage')

   v('weights', weights)

   affirm.isTensor1D(weights, 'weights')

   assert(weights:size(1) == self._ys:size(1),
          'weights not same size as ys')

   local sumWeights = torch.sum(weights)

   if sumWeights == 0 then
      local reason = 'all weights used were 0'
      return false, reason
   end

   local numerator = torch.sum(torch.cmul(weights, self._ys))
   local result = numerator / sumWeights
   
   v('numerator', numerator)
   v('sumWeights', sumWeights)
   v('result', result)

   assert(result == result, 'result is NaN')

   return true, result
end -- _weightedAverage
