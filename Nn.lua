-- Nn.lua
-- common functions for the Nearest Neighbor package

require 'affirm'
require 'makeVerbose'
require 'verify'

-- API overview
if false then
   -- simple average
   ok, estimate = Nn.estimateAvg(xs, ys, nearestIndices, visible, weights, k)

   -- kernel-weighted average
   ok, estimate = Nn.estimateKwavg(xs, ys, nearestIndices, visible, weights, k)

   -- local linear regression
   ok,estimate = Nn.estimateLlr(xs, ys, nearestIndices, visible, weights, k)
   
   -- euclidean distance
   distances = Nn.euclideanDistances(xs, query)

   -- nearest neighbor distances and indices
   sortedDistances, sortedIndices = Nn.nearest(xs, query)

   -- weights from the Epanenchnikov kernel
   -- where lambda is the distance to the k-th nearest neighbor
   weights = Nn.weights(sortedDistances, lambda)
end

Nn = {}

function Nn.estimateAvg(xs, ys, nearestIndices, visible, k)
   -- return true, average of k nearest visible neighbors
   -- ignore the weights
   local v, isVerbose = makeVerbose(false, 'Nn.estimateAvg')
   verify(v, isVerbose,
          {{xs, 'xs', 'isAny'},
           {ys, 'ys', 'isTensor1D'},
           {nearestIndices, 'nearestIndices', 'isTensor1D'},
           {visible, 'visible', 'isTensor1D'},
           {k, 'k', 'isIntegerPositive'}})

   local sum = 0
   local found = 0
   for nearestIndex = 1, nearestIndices:size(1) do
      local obsIndex = nearestIndices[nearestIndex]
      if visible[obsIndex] == 1 then
         found = found + 1
         sum = sum + ys[obsIndex]
         v('obsIndex, y', obsIndex, ys[obsIndex])
         if found == k then 
            break
         end
      end
   end

   if found < k then
      return false, 'not able to find k neighbors'
   else
      local result = sum / k
      v('result', result)
      return true, result
   end
end -- Nn.estimateAvg

function Nn.estimateKwavg(k, sortedNeighborIndices, visible, weights, allYs)
   -- ARGS
   -- xs             : 2D Tensor of inputs; obs[i] = xs[i]
   -- ys             : 1D Tensor of target values
   -- nearestIndices : table
   --                  nearestIndices[1] = obsIndex of closest neighbor
   --                     xs[nearestIndices[1]] = features of closest neighbor
   --                   nearestIndices[2] = obsIndex of 2nd closest neighbor
   -- visible        : table with elements in {0,1}
   --                  visible[1] == 1 iff using obsIndex 1
   -- k              : integer > 0, number of neighbors to use
   -- RETURNS
   -- ok             : true or false
   -- estimate       : number or string
   
   local v, isVerbose = makeVerbose(false, 'Nn.estimateKwavg')
   verify(v, isVerbose,
          {{k, 'k', 'isIntegerPositive'},
           {sortedNeighborIndices, 'sortedNeighborIndices', 'isTensor1D'},
           {visible, 'visible', 'isTensor1D'},
           {weights, 'weights', 'isTensor1D'},
           {allYs, 'allYs', 'isTensor1D'}})

   local sumWeightedYs = 0
   local sumWeights = 0
   local found = 0
   for i = 1, visible:size(1) do
      local obsIndex = sortedNeighborIndices[i]
      if visible[obsIndex] == 1 then
         local weight = weights[i]
         local y = allYs[obsIndex]
         v('i,obsIndex,weight,y', i, obsIndex, weight, y)
         sumWeights = sumWeights + weight
         sumWeightedYs = sumWeightedYs + weight * y
         found = found + 1
         if found == k then
            break
         end
      end
   end
   v('sumWeights', sumWeights)
   v('sumWeightedYs', sumWeightedYs)

   if sumWeights == 0 then
      return false, 'all weights were zero'
   elseif found < k then
      return false, string.format('only %d obs in neighborhood; k = %d',
                                  found, k)
   else
      local estimate = sumWeightedYs / sumWeights
      v('estimate', estimate)
      return true, estimate
   end
end -- Nn.estimateKwavg


function Nn.euclideanDistance(x, query)
   -- return scalar Euclidean distance
   local v, isVerbose = makeVerbose(false, 'Nn:euclideanDistance')
   verify(v, isVerbose,
          {{x, 'x', 'isTensor1D'},
           {query, 'query', 'isTensor1D'}})
   assert(x:size(1) == query:size(1))
   local ds = x - query
   local distance = math.sqrt(torch.sum(torch.cmul(ds, ds)))
   v('distance', distance)
   return distance
end -- euclideanDistance

function Nn.euclideanDistances(xs, query)
   -- return 1D tensor such that result[i] = EuclideanDistance(xs[i], query)
   -- We require use of Euclidean distance so that this code will work.
   -- It computes all the distances from the query point at once
   -- using Clement Farabet's idea to speed up the computation.

   local v, isVerbose = makeVerbose(false, 'Nn:euclideanDistances')
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
   v('distances', distances)
   local sortedDistances, sortedIndices = torch.sort(distances)
   v('sortedDistances', sortedDistances)
   v('sortedIndices', sortedIndices)
   return sortedDistances, sortedIndices
end -- Nn.nearest

function Nn.weights(sortedDistances, lambda)
   -- return values of Epanenchnov kernel using euclidean distance
   local v, isVerbose = makeVerbose(false, 'KernelSmoother.weights')
   verify(v, isVerbose,
          {{sortedDistances, 'sortedDistances', 'isTensor1D'},
           {lambda, 'lambda', 'isNumberPositive'}})
   local nObs = sortedDistances:size(1)

   local t = sortedDistances / lambda
   v('t', t)

   local one = torch.Tensor(nObs):fill(1)
   local indicator = torch.le(torch.abs(t), one):type('torch.DoubleTensor')
   v('indicator', indicator)

   local dt = torch.mul(one - torch.cmul(t, t), 0.75)
   v('dt', dt)

   -- in torch, inf * 0 --> nan (not zero)
   local weights = torch.cmul(dt, indicator)
   v('weights', weights)

   return weights
end -- Nn.weights
