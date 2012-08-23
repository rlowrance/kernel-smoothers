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
-- errorIfZeroSumWeights : boolean, optional, default false
--                         if false, call error if the weights sum to zero
--                         if true, return NaN as the estimates in this case    
--
-- RESULTS:
-- estimate : number
--            weighted average y value using the construction-time 
--            specified kernel function
function Kwavg:estimate(xs, ys, query, lambda, errorIfZeroSumWeights)
   -- type check and value check the arguments
   self:_typeAndValueCheck(xs, ys, lambda)

   assert(query:dim() == 1,
          'query must be 1D Tensor')

   local errorIfZeroSumWeights
   if errorIfZeroSumWeights == nil then errorIfZeroSumWeights = true end
   assert(type(errorIfZeroSumWeights) == 'boolean')

   local weights = self:_determineWeights(xs, query, lambda)
   
   return self:_weightedAverage(weights, ys, errorIfZeroSumWeights)

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
-- estimate : number
--            average y value of the k nearest neighbors in the xs
--            from the query using the Euclidean distance 
function Kwavg:smooth(xs, ys, queryIndex, lambda, 
                      useQueryPoint, errorIfZeroSumWeights)
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

   local errorIfZeroSumWeights
   if errorIfZeroSumWeights == nil then errorIfZeroSumWeights = true end
   assert(type(errorIfZeroSumWeights) == 'boolean')

   local weights = 
      self:_determineWeights(xs, xs[math.floor(queryIndex)], lambda)

   if trace then
      print('smooth: weights before adjustments\n', weights)
   end

   -- if not using the point at queryIndex, make its weight 0
   if not useQueryPoint then
      weights[queryIndex] = 0
   end
   
   return self:_weightedAverage(weights, ys, errorIfZeroSumWeights)
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
         print('xs\n', xs)
         print('query\n', query)
         print('distances\n', distances)
      end

      local t = distances / lambda
      if trace then print('t\n', t) end
      
      local one = torch.Tensor(xs:size(1)):fill(1)
      local dt = torch.mul(one - torch.cmul(t, t), 0.75)
      local le = torch.le(torch.abs(t), one):type('torch.DoubleTensor')
      local weights = torch.cmul(le, dt)
      if trace then print('weights\n', weights) end
      --if lambda == 1 then halt() end
      return weights
end

--------------------------------------------------------------------------------
-- _weightedAverage
--------------------------------------------------------------------------------

-- return weighted average of the weights and y values
function Kwavg:_weightedAverage(weights, ys, errorIfZeroSumWeights)
   local trace = false
   assert(weights)
   assert(ys)
   assert(errorIfZeroSumWeights)

   local sumWeights = torch.sum(weights)
   if sumWeights == 0 and ignoreZeroSumWeights == false then
      assert(sumWeights > 0, 
             'sum of weights are not positive; is ' .. sumWeights)
   end
   if trace then 
      print('Kwavg:_weightedAverage')
      print('weights\n', weights)
      print('ys\n', ys)
      print(string.format('numerator=%f', torch.sum(torch.cmul(weights, ys))))
      print(string.format('denominator=%f', sumWeights))
   end
   return torch.sum(torch.cmul(weights, ys)) / sumWeights
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
