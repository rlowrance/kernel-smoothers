-- KernelSmoother.lua
-- auxillary class for all kernel smoothers
-- used to hold utility functions
-- could have been a parent class

-- API overview
if false then
   ks = KernelSmoother()
   
   -- distances from query to each x
   distance = ks:euclideanDistances(xs, query) 

   -- weights using Epanechnikov quadratic kernel from query with radius lambda
   weights = ks:weights(xs, query, lambda)

   -- weighted average ys 
   weightedAverage = ks:weightedAverage(weights)
end

--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

torch.class('KernelSmoother')

function KernelSmoother:__init()
end -- __init

--------------------------------------------------------------------------------
-- PUBLIC METHODS
--------------------------------------------------------------------------------


function KernelSmoother:euclideanDistances(xs, query)
   -- return 1D tensor such that result[i] = EuclideanDistance(xs[i], query)
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v, isVerbose = makeVerbose(false, 'Knn:_euclideanDistances')
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
end -- euclideanDistances

function KernelSmoother:weightedAverage(ys, weights)
   -- maybe return weighted average of ys
   -- RETURN
   --    true, weightedAverage
   --    false, reason
   local v, isVerbose = makeVerbose(false, 'KernelSmoother:weightedAverage')
   verify(v,
          isVerbose,
          {{ys, 'ys', 'isTensor1D'},
           {weights, 'weights', 'isTensor1D'}})


   assert(weights:size(1) == ys:size(1),
          'weights not same size as ys')

   local sumWeights = torch.sum(weights)

   if sumWeights == 0 then
      local reason = 'all weights used were 0'
      return false, reason
   end

   local numerator = torch.sum(torch.cmul(weights, ys))
   local result = numerator / sumWeights
   
   v('numerator', numerator)
   v('sumWeights', sumWeights)
   v('result', result)

   assert(result == result, 'result is NaN') -- should never happen

   return true, result
end -- weightedAverage

function KernelSmoother:weights(xs, query, lambda) 
   -- return 1D tensor such that result[i] = Kernel_lambda(query, xs[i])
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v, isVerbose = makeVerbose(false, 'Kwavg:_determineWeights')
   verify(v,
          isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {query, 'query', 'isTensor1D'},
           {lambda, 'lambda', 'isNumberPositive'}})
   
   local distances = self:euclideanDistances(xs, query)
   local t = distances / lambda
   
   local one = torch.Tensor(xs:size(1)):fill(1)
   local dt = torch.mul(one - torch.cmul(t, t), 0.75)
   local le = torch.le(torch.abs(t), one):type('torch.DoubleTensor')
   local weights = torch.cmul(le, dt)
   
   v('distances', distances)
   v('t', t)
   v('dt', dt)
   v('le', le)
   v('weights', weights)
   
   return weights
end -- weights
