-- Kwavg.lua
-- kernel weighted average using the Epanechnikov Quadratic Kernel
-- ref: Hastie-01 p.166 and following

-- example
if false then
   local kwavg = Kwavg('epanechnikov quadratic')

   local estimate = Knn:estimate(xs, ys, query, lambda)

   local useQueryPoint = false
   local smoothedEstimate = Kwavg:smooth(xs, ys, queryIndex, lambda,
                                         useQueryPoint)
end

-- create class object
local Kwavg = torch.class('Kwavg')

--------------------------------------------------------------------------------
-- initializer
--------------------------------------------------------------------------------

-- validate kernel name
function Kwavg:__init(kernelName)
   assert(kernelName, 'kernel name is missing')
   assert(kernelName == 'epanechnikov quadratic',
          'for now, only kernel support is epanechnikov quadratic')
   self._kernelName = kernelName
end

--------------------------------------------------------------------------------
-- estimate
--------------------------------------------------------------------------------

-- estimate y for a new query point using the specified kernel
-- ARGS:
-- xs     : 2D Tensor
--          the i-th input sample is xs[i]
-- ys     : 1D Tensor or array of numbers
--          y[i] is the known value (target) of input sample xs[i]
--          number of ys must equal number of rows in xs
-- query  : 1D Tensor
-- lambda : number > 0, xs outside of this radius are given 0 weights
-- RESULTS:
-- true, estimate : estimate is the estimate for the query
--                  estimate is a number
-- false, reason  : no estimate was produced
--                  rsult is a string
function Kwavg:estimate(xs, ys, query, lambda)
   local trace = false
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, lambda)

   assert(query)
   assert(query:dim() == 1,
          'query must be 1D Tensor')

   if trace then
      print('Kwavg:estimate')
      print(' xs:size()', xs:size())
      print(' ys:size()', ys:size())
      print(' query:size()', query:size())
      print(' lambda', lambda)
   end
   
   local weights = self:_determineWeights(xs, query, lambda)
   
   return self:_weightedAverage(weights, ys)

end

-----------------------------------------------------------------------------
-- smooth
-----------------------------------------------------------------------------

-- re-estimate y for an existing xs[queryIndex]
-- ARGS:
-- xs            : 2D Tensor
--                 the i-th input sample is xs[i]
-- ys            : 1D Tensor or array of numbers
--                 y[i] is the known value (target) of input sample xs[i]
--                 number of ys must equal number of rows in xs 
-- queryIndex    : number >= 1
--                 xs[math.floor(queryIndex)] is re-estimated
-- k             : number >= 1 
--                 math.floor(k) neighbors are considered
-- useQueryPoint : boolean
--                 if true, use the point at queryIndex as a neighbor
--                 if false, don't
-- errorIfZeroSumWeights : boolean, optional, default false
--                         if false, call error if the weights sum to zero
--                         if true, return NaN as the estimates in this case    
--
-- RESULTS:
-- true, estimate : estimate is the estimate at the query index
--                  estimate is a number
-- false, reason  : no estimate was produced
--                  reason is a string
function Kwavg:smooth(xs, ys, queryIndex, lambda, useQueryPoint)
   local trace = false
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, lambda)

   assert(type(queryIndex) == 'number',
          'queryIndex must be a number')
   assert(queryIndex >= 1,
          'queryIndex must be at least 1')
   assert(queryIndex <= xs:size(1),
          'queryIndex cannot exceed number of samples = ' .. xs:size(1))

   if useQueryPoint == nil then
      error('useQueryPoint not supplied')
   end
   assert(type(useQueryPoint) == 'boolean')

   local weights = 
      self:_determineWeights(xs, xs[math.floor(queryIndex)], lambda)

   if false and trace then
      -- check if all weights are close to zero
      local tolerance = 1e-6
      print('Kwavg:smooth weights >', tolerance)
      print(' number of weights', weights:size(1))
      local countPrinted = 0
      for i = 1, weights:size(1) do
         if weights[i] > tolerance then
            print(' i, weight', i, weights[i])
            if i == queryIndex then
               print(' above i value is for the queryIndex')
            end
            countPrinted = countPrinted + 1
         end
      end
      if countPrinted == 0 then
         error(' all weights close to zero')
      end
   end

   -- if not using the point at queryIndex, make its weight 0
   if not useQueryPoint then
      weights[queryIndex] = 0
   end
   
   local ok, value = self:_weightedAverage(weights, ys)
   if trace then
      print('Kwavg:smooth ok, value', ok, value)
   end
   
   return ok, value
end

--------------------------------------------------------------------------------
-- PRIVATE METHODS
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- _determineWeights
--------------------------------------------------------------------------------

-- return 1D tensor such that result[i] = Kernel_lambda(query, xs[i])
-- We require use of Euclidean distance so that this code will work.
-- It computes all the distances from the query point at once
-- using Clement Farabet's idea to speed up the computation.
function Kwavg:_determineWeights(xs, query, lambda)
   local trace = false
   assert(xs)
   assert(query)
   assert(lambda)
   assert(lambda > 0)
   assert(self._kernelName == 'epanechnikov quadratic')

   query = query:clone()  -- in case the query is a view of the xs storage
   
   -- create a 2D Tensor where each row is the query
   -- This construction is space efficient relative to replicating query
   -- queries[i] == query for all i in range
   -- Thanks Clement!
   local queries = 
      torch.Tensor(query:storage(),
                   1,               -- offset
                   xs:size(1), 0,   -- row index offset and stride
                   xs:size(2), 1)   -- col index offset and stride
      
      local distances = torch.add(queries, -1 , xs) -- queries - xs
      distances:cmul(distances)                     -- (queries - xs)^2
      distances = torch.sum(distances, 2):squeeze() -- \sum (queries - xs)^2
      distances = distances:sqrt()                  -- Euclidean distances
      
      if trace then
         print('Kwavg:_determineWeights')
         print('query\n', query)    
         local sizeLimit = 10
         --self:_print2DTensor(xs, 'xs')
         self:_print1DTensor(distances, 'distances')
         -- check that distances are not all zero
         local tolerance = 1e-15
         local countAtLeastTolerance = 0
         for index = 1, distances:size(1) do
            if distances[index] > tolerance then
               countAtLeastTolerance = countAtLeastTolerance + 1
            end
         end
         print(string.format('number of distance at least %f is %d',
                             tolerance, countAtLeastTolerance))
         print('lambda', lambda)
      end

      local t = distances / lambda
      
      local one = torch.Tensor(xs:size(1)):fill(1)
      local dt = torch.mul(one - torch.cmul(t, t), 0.75)
      local le = torch.le(torch.abs(t), one):type('torch.DoubleTensor')
      local weights = torch.cmul(le, dt)
      if trace then 
         self:_print1DTensor(t, 't')
         self:_print1DTensor(dt, 'dt')
         self:_print1DTensor(le, 'le')
         self:_print1DTensor(weights, 'weights')
      end

      return weights
end

--------------------------------------------------------------------------------
-- _weightedAverage
--------------------------------------------------------------------------------

-- attempt to weighted average of the weights and y values
-- RETURNS
-- true, weightedAverage : if the weighted average can be determined
--                         weightedAverage is a number
-- false, reason         : if the weighted average cannot be determined
--                         reason is a string          
function Kwavg:_weightedAverage(weights, ys)
   local trace = false
   assert(weights)
   assert(ys)

   local sumWeights = torch.sum(weights)
   --assert(weights[3] > 0.5, 'weights[3] = ' .. weights[3])
   if sumWeights == 0 then
      local reason = 'no observations within lambda of query observation'
      if trace then
         print('Kwavg:_weightedAverage returning false,', reason)
      end
      return false, reason
   end

   if trace then
      print('Kwavg:_weightedAverage')
      print(' weights:size()', weights:size())
      print(' ys:size()', ys:size())
   end

   local numerator = torch.sum(torch.cmul(weights, ys))
   local result = numerator / sumWeights
   assert(result == result, 'result is NaN')
   if trace then 
      print('Kwavg:_weightedAverage')
      --print('weights\n', weights)
      --print('ys\n', ys)
      print(string.format(' sumWeights = %0.15f', sumWeights))
      print(string.format(' numerator  = %0.15f', numerator))
      print(string.format(' result     = %0.15f', result))
   end
   return true, result
end

--------------------------------------------------------------------------------
-- _print1DTensor
--------------------------------------------------------------------------------

-- print entire Tensor or just first entries if its large
function Kwavg:_print1DTensor(t, name)
   assert(t)
   assert(name)

   print(string.format('tensor %s of type %s', name, torch.typename(t)))
   local sizeLimit = 10
   if t:size(1) <= sizeLimit then
      print(t)
   else
      print(string.format('first %d entries are:', sizeLimit))
      for i = 1, sizeLimit do
         print(string.format('%s[%d] = %f', name, i, t[i]))
      end
   end
end

--------------------------------------------------------------------------------
-- _print2DTensor
--------------------------------------------------------------------------------

-- print entire Tensor or just first entries if its large
function Kwavg:_print2DTensor(t, name)
   assert(t)
   assert(name)

   print(string.format('tensor %s of type %s', name, torch.typename(t)))
   local sizeLimit = 10
   if t:size(1) <= sizeLimit then
      print(t)
   else
      print(string.format('first %d entries are:', sizeLimit))
      for i = 1, sizeLimit do
         print(string.format('%s[%d] = %f', name, i, t[i]))
      end
   end
end

         
--------------------------------------------------------------------------------
-- _typeAndValueCheck
--------------------------------------------------------------------------------

-- verify type and values of arguments
function Kwavg:_typeAndValueCheck(xs, ys, lambda)
   -- type and value check xs
   assert(xs,
          'xs must be supplied')
   assert(string.match(torch.typename(xs), 'torch%..*Tensor'),
          'xs must be a Tensor')
   assert(xs:dim() == 2,
          'xs must be 2D Tensor')

   -- type and value check ys
   assert(ys,
          'ys must be supplied')
   if type(ys) == 'table' then
      assert(#ys == xs:size(1), 
             'number of ys must equal number of xs')
   elseif string.match(torch.typename(ys), 'torch%..*Tensor') then
      assert(ys:dim() == 1, 'ys must be a 1D Tensor')
      assert(ys:size(1) == xs:size(1),
             'number of ys must equal number of xs')
   else
      assert(false, 'ys must be an array or a 1D Tensor')
   end
   assert(ys[1],
          'ys must be subscriptable by one value')

   -- type and value check lambda
   assert(type(lambda) == 'number',
          'lambda must be a number')
   assert(lambda > 0,
          'lambda must be positive')

end
