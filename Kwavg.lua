-- Kwavg.lua
-- kernel weighted average using the Epanechnikov Quadratic Kernel
-- ref: Hastie-01 p.166 and following

-- example
if false then
   local kwavg = Kwavg(xs, ys, 'epanechnikov quadratic')

   local ok, estimate = kwavg:estimate(query, lambda)
   if not ok then
      -- estimate is a string explaining why no estimate was provided
   end

   -- smooth is DEPRECATED
   local useQueryPoint = false
   local smoothedEstimate = Kwavg:smooth(queryIndex, lambda,
                                         useQueryPoint)
end

require 'affirm'
require 'KernelSmoother'
require 'makeVerbose'
require 'verify'



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
   local v, verbose = makeVerbose(false, 'Kwavg:__init')

   verify(v,
          verbose,
          {{xs, 'xs', 'isTensor2D'},
           {ys, 'ys', 'isTensor1D'},
           {kernelName, 'kernelName', 'isString'}
          })

   assert(kernelName == 'epanechnikov quadratic',
          'for now, only kernel supported is epanechnikov quadratic')
   self._xs = xs
   self._ys = ys
   self._kernelName = kernelName
   self._kernelSmoother = KernelSmoother()
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
   local v, verbose = makeVerbose(false, 'Kwavg:estimate')

   verify(v,
          verbose,
          {{query, 'query', 'isTensor1D'},
           {lambda, 'lambda', 'isNumberPositive'}
          })

   local weights = self._kernelSmoother:weights(self._xs, 
                                                query, 
                                                lambda)
   local ok, estimate = self._kernelSmoother:weightedAverage(self._ys, 
                                                             weights)
   
   v('weights', weights)
   v('ok', ok)
   v('estimate', estimate)

   return ok, estimate
end -- estimate

-- DEPRECATE smooth method
--[[
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
   --]]

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

-- NONE