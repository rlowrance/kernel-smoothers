-- KernelSmoother.lua
-- auxillary functions for all kernel smoothers

-- API overview
if false then
   -- distances from query to each x
   distance = KernelSmoother.euclideanDistances(xs, query) 

   -- indices of all nearest points to query
   indices = KernelSmoother.nearestIndices(xs, query)

   -- weights using Epanechnikov quadratic kernel from query with radius lambda
   weights = KernelSmoother.weights(xs, query, lambda)

   -- weighted average ys 
   weightedAverage = KernelSmoother.weightedAverage(weights)
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

function KernelSmoother.euclideanDistance(x1, x2)
   local v, isVerbose = makeVerbose(true, 'KernelSmoother:euclideanDistance')
   verify(v, isVerbose,
          {{x1, 'x1', 'isTensor1D'},
           {x2, 'x2', 'isTensor1D'}})
   assert(x1:size(1) == x2:size(1))

   local ds = torch.add(x1, -1, x2)  -- x1 - x2
   ds:cmul(ds)
   local d = math.sqrt(torch.sum(ds))
   return d
end -- euclideanDistance
   

function KernelSmoother.euclideanDistances(xs, query)
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
end -- euclideanDistances

function KernelSmoother.nearest(xs, query)
   -- determine distance from query to each row of xs
   -- RETURN
   -- sortedDistances : 1D Tensor 
   -- sortedIndices   : 1D Tensor 
   local v, isVerbose = makeVerbose(false, 'KernelSmoother:nearest')
   verify(v, isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {query, 'query', 'isTensor1D'}})
   local distances = KernelSmoother.euclideanDistances(xs, query)
   local sortedDistances, sortedIndices = torch.sort(distances)
   return sortedDistances, sortedIndices
end -- nearest

function KernelSmoother.weightedAverage(ys, weights)
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

function KernelSmoother.kernels(sortedDistances, lambda)
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
end -- kernels

function KernelSmoother.kernelOLD(xs, query, lambda) 
   -- return 1D tensor such that result[i] = Kernel_lambda(query, xs[i])
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v, isVerbose = makeVerbose(false, 'KernelSmoother:kernel')

   verify(v,
          isVerbose,
          {{xs, 'xs', 'isTensor2D'},
           {query, 'query', 'isTensor1D'},
           {lambda, 'lambda', 'isNumberPositive'}})
   
   local distances = KernelSmoother.euclideanDistances(xs, query)
   v('distances', distances)

   local t = distances / lambda
   if debug == 1 and allZeroes(t) then
      error('t is all zeroes')
   end

   local one = torch.Tensor(xs:size(1)):fill(1)
   local dt = torch.mul(one - torch.cmul(t, t), 0.75)
   local le = torch.le(torch.abs(t), one):type('torch.DoubleTensor')
   local weights = torch.cmul(le, dt)
   
   v('t', t)
   v('dt', dt)
   v('le', le)
   v('weights', weights)

   return weights
end -- weights
