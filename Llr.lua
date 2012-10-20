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
   local v, verbose = makeVerbose(false, 'Llr:__init')

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

   v('self._xs:size()', self._xs:size())
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
   local v, isVerbose = makeVerbose(true, 'Llr:estimate')
   local debug = 1  -- torch.inverse(BTWB): U is always singular

   if debug ~= 0 then
      print('DEBUGGING Llr:estimate')
   end

   verify(v,
          isVerbose,
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
   if allZeroes(weights) then
      return 
         false, 
         'weights all zeroes (probably no neighbors; if so, increase lambda)'
   end

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

   local function printZeroRowsCols(t, name)
      print('rows and cols that are all zeroes', name)
      for d1 = 1, t:size(1) do
         if allZeroes(t[d1]) then
            print(string.format('%s[%d] = 0', name, d1))
         end
      end
      for d2 = 1, t:size(2) do
         if allZeroes(t[d2]) then
            print(string.format('%s[:][%d] = 0', name, d2))
         end
      end  
      print('end of rows and cols that are all zeros')
   end -- printZeroRowsCols

   if debug == 1 then
      printZeroRowsCols(self._xs, 'self._xs')
   end

   local nObs = self._xs:size(1)
   local nDims = self._xs:size(2)
   local dp1 = nDims + 1
   local B = torch.Tensor(nObs, dp1)
   for i = 1, nObs do
      B[i] = augment(self._xs[i])
      if debug == 1 and allZeroes(self._xs[i]) then
         print(string.format('self._xs[%d] = 0', i))
      end
   end
   
   --v('B', B)

   if debug == 1 and false then
      -- print row 8 of B
      print('row 8 of B')
      for d2 = 1, B:size(2) do
         print(string.format('B[8][%d]=%f', d2, B[8][d2]))
      end
   end

   local BT = B:t()
   
   v('nObs', nObs)
   local W = DiagonalMatrix(weights)

   -- BTWB = B^t W B
   local BTWB = BT * (W:mul(B))
   --v('BTWB', BTWB)
   v('BTWB:size()', BTWB:size())
   if debug == 1 then
      printZeroRowsCols(BTWB, 'BTWB')
   end
   if debug == 1 and allZeroes(BTWB) then
      v('BTWB is all zeroes')
      halt()
   end
   if debug == 1 and false then
      -- replace near zeroes with random values
      -- this removes the singularity
      local small = 1e-4
      for d1 = 1, BTWB:size(1) do
         for d2 = 1, BTWB:size(2) do
            if math.abs(BTWB[d1][d2]) < small then
               BTWB[d1][d2] = torch.random(0, 1) -- sample from N(0,1)
               print('replaced d1,d2', d1, d2)
            end
         end
      end
   end

   -- catch error in attempting to invert BTWB
   if debug == 1 and false then
      -- no problem with inverse of random matrix
      BTWB = torch.rand(BTWB:size(1), BTWB:size(2))
   end
   local ok, BTWBInv = pcall(torch.inverse, BTWB)
   --v('BTWBInv', BTWBInv)
   if not ok then
      -- if the error message is "getrf: U(i,i) is 0, U is singular"
      -- then LU factorization succeeded but U is exactly 0, so that
      --      division by zero will occur if U is used to solve a
      --      system of equations
      -- ref: http://dlib.net/dlib/matrix/lapack/getrf.h.html
      if debug == 1 then
         print('Llr:estimate: error in call to torch.inverse')
         print('error message = ' .. BTWBInv)
         error(BTWBInv)
      end
      halt()
      return false, BTWBInv
   end

   if false then
      -- break long multiplication into components for debugging
      v('imbed(augment(query))', imbed(augment(query)))
      local result1 = imbed(augment(query)) * BTWBInv 
      local result2 = BT * W 
      local result3 = result1 * result2
      local result = result3 * self._ys
      v('debug result', result)
   end
   local result = imbed(augment(query)) * BTWBInv * BT * W:mul(self._ys)
   -- result is a 1D tensor
   v('result', result)

   return true, result[1]
end -- estimate

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

-- NONE
