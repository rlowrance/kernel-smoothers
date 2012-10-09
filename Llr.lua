-- Llr.lua
-- local linear regression
-- ref: Hastie-01 p.166 and following

-- example
if false then
   local kwavg = Llr(xs, ys, kernelName)

   local ok, estimate = LLr:estimate(query, lambda)
   if not ok then
      -- estimate is a string explaining what went wrong
   end
end

require 'affirm'
require 'KernelSmoother'
require 'makeVerbose'
require 'verify'




--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

-- create class object
torch.class('Llr')

function Llr:__init(xs, ys, kernelName)
   -- ARGS:
   -- xs         : 2D Tensor
   --              the i-th input sample is xs[i]
   -- ys         : 1D Tensor or array of numbers
   --              y[i] is the known value (target) of input sample xs[i]
   --              number of ys must equal number of rows in xs
   -- kernelName : string
   local v, verbose = makeVerbose(true, 'Llr:__init')

   verify(v,
          verbose,
          {{xs, 'xs', 'isTensor2D'},
           {ys, 'ys', 'isTensor1D'},
           {kernelName, 'kernelName', 'isString'}
          })

   assert(xs:size(1) == ys:size(1),
          'not same number of observations in xs and ys')

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

function Llr:estimate(query, lambda)
   -- estimate y for a new query point using the specified kernel
   -- ARGS:
   -- lambda : number > 0, xs outside of this radius are given 0 weights
   -- RESULTS:
   -- true, estimate : estimate is the estimate for the query
   --                  estimate is a number
   -- false, reason  : no estimate was produced
   --                  reason is a string
   local v, verbose = makeVerbose(true, 'Llr:estimate')

   verify(v,
          verbose,
          {{query, 'query', 'isTensor1D'},
           {lambda, 'lambda', 'isNumberPositive'}
          })

   v('self', self)

   assert(query:size(1) == self._xs:size(2),
          'query not size compatible with xs')

   local weights = self._kernelSmoother:weights(self._xs,
                                                query, 
                                                lambda)
   v('weights', weights)

   local function augment(v)
      -- return 1D tensor (1, v)
      local result = torch.Tensor(v:size(1) + 1)
      result[1] = 1
      for d = 1, v:size(1) do
         result[d+1] = v[d]
      end
      return result
   end -- augment

   local function imbed(v)
      -- return 2D tensor of size 1 x v:size(1)
      local result = torch.Tensor(1, v:size(1))
      for d = 1, v:size(1) do
         result[1][d] = v[d]
      end
      return result
   end -- imbed
   
   local nObs = self._xs:size(1)
   local nDims = self._xs:size(2)
   local dp1 = nDims + 1
   local B = torch.Tensor(nObs, dp1)
   for i = 1, nObs do
      B[i] = augment(self._xs[i])
   end
   v('B', B)

   local BT = B:t()
   
   local W = torch.Tensor(nObs, nObs):zero()
   for i = 1, nObs do
      W[i][i] = weights[i]
   end
   v('W', W)

   -- BTWB = B^t W B
   local BTWB = BT * W * B  -- order does not matter for efficiency
   v('BTWB', BTWB)

   -- catch error
   local ok, BTWBInv = pcall(torch.inverse, BTWB)
   if not ok then
      print('Llr:estimate: error in call to torch.inverse')
      print('error message = ' .. BTWBInv)
      return false, BTWBInv
   end
   BTWBInv = torch.inverse(BTWB)
   v('BTWBInv', BTWBInv)

   if debug then
      -- break long multiplication into components for debugging
      v('imbed(augment(query))', imbed(augment(query)))
      local result1 = imbed(augment(query)) * BTWBInv 
      local result2 = BT * W 
      local result3 = result1 * result2
      local result = result3 * self._ys
      v('debug result', result)
   end
   local result = imbed(augment(query)) * BTWBInv * BT * W * self._ys
   -- result is a 1D tensor
   v('result', result)

   return true, result[1]
end -- estimate

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

-- NONE
