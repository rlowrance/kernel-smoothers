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
   -- k                     : integer > 0, number of neighbors to use
   -- sortedNeighborIndices : 1D Tensor
   --                         use first k neighbors that are also visible
   -- visible               : 1D Tensor
   --                         visible[obsIndex] == 1 ==> use this observation
   --                         as a neighbor
   -- weights               : 1D Tensor
   -- allYs                 : 1D Tensor
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

function Nn.estimateLlr(k, regularizer,
                        sortedNeighborIndices, visible, weights, 
                        query, allXs, allYs)
   -- ARGS
   -- k                     : integer > 0, number of neighbors to use
   -- regularizer           : number >= 0, added to each weight
   -- sortedNeighborIndices : 1D Tensor
   --                         use first k neighbors that are also visible
   -- visible               : 1D Tensor
   --                         visible[obsIndex] == 1 ==> use this observation
   --                         as a neighbor
   -- weights               : 1D Tensor
   -- allXs                 : 2D Tensor
   -- allYs                 : 1D Tensor
   -- RETURNS
   -- ok             : true or false
   -- estimate       : number or string

   local debug = 0
   --local debug = 1  -- determine why inverse fails
   local v, isVerbose = makeVerbose(false, 'Nn.estimateLlr')
   verify(v, isVerbose,
          {{k, 'k', 'isIntegerPositive'},
           {regularizer, 'regularizer', 'isNumberNonNegative'},
           {sortedNeighborIndices, 'sortedNeighborIndices', 'isTensor1D'},
           {visible, 'visible', 'isTensor1D'},
           {weights, 'weights', 'isTensor1D'},
           {query, 'query', 'isTensor1D'},
           {allXs, 'allXs', 'isTensor2D'},
           {allYs, 'allYs', 'isTensor1D'}})

   assert(allYs:size(1) == allXs:size(1),
          'allXs and allYs must have same number of observations')
   
   local nDims = allXs:size(2)

   assert(k > nDims,
          string.format('undetermined since k(=%d) <= nDims (=%d)',
                        k, nDims))
   -- FIX ME: create using k nearest visible neighbors
   -- create regression matrix B
   -- by prepending a 1 in the first position
   local B = torch.Tensor(k, nDims + 1)
   local selectedYs = torch.Tensor(k)
   local wVector = torch.Tensor(k)
   local found = 0
   for i = 1, allXs:size(1) do
      local obsIndex = sortedNeighborIndices[i]
      if visible[obsIndex] == 1 then
         found = found + 1
         v('i,obsIndex,found', i, obsIndex, found)
         B[found][1] = 1
         for d = 1, nDims do
            B[found][d+1] = allXs[obsIndex][d]
         end
         selectedYs[found] = allYs[obsIndex]
         wVector[found] = weights[i] + regularizer
         if found == k then
            break
         end
      end
   end
   v('nObs, nDims', nObs, nDims)
   v('B (regression matrix)', B)
   v('selectedYs', selectedYs)
   v('wVector', wVector)

   -- also prepend a 1 in the first position of the query
   -- to make the final multiplication work, the prepended query needs to be 2D
   local extendedQuery = torch.Tensor(1, nDims + 1)
   extendedQuery[1][1] = 1
   for d = 1, nDims do
      extendedQuery[1][d + 1] = query[d]
   end
   v('extendedQuery', extendedQuery)

   local BT = B:t()        -- transpose B
   local W = DiagonalMatrix(wVector)
   v('BT', BT)
   v('W', W) 
   
   -- BTWB = B^T W B
   local BTWB = BT * (W:mul(B))
   v('BTWB', BTWB)

   -- invert BTWB, catching error
   local ok, BTWBInv = pcall(torch.inverse, BTWB)
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
      return false, BTWBInv  -- return the error message
   end

   local betas =  BTWBInv * BT * W:mul(selectedYs)
   local estimate1 = extendedQuery * BTWBInv * BT * W:mul(selectedYs)
   v('estimate1', estimate1)
   local estimate = extendedQuery * betas
   v('extendedQuery', extendedQuery)
   v('beta', BTWBInv * BT * W:mul(selectedYs))
   v('estimate', estimate[1])


   affirm.isTensor1D(estimate, 'estimate')
   assert(1 == estimate:size(1))
   return true, estimate[1]
end -- Nn.estimateLlr


function Nn.euclideanDistance(x, query)
   -- return scalar Euclidean distance
   local debug = 0
   --debug = 1  -- zero value for lambda
   local v, isVerbose = makeVerbose(false, 'Nn:euclideanDistance')
   verify(v, isVerbose,
          {{x, 'x', 'isTensor1D'},
           {query, 'query', 'isTensor1D'}})
   assert(x:size(1) == query:size(1))
   local ds = x - query
   if debug == 1 then
      for i = 1, x:size(1) do
         print(string.format('x[%d] %f query[%d] %f ds[%d] %f',
                             i, x[i], i, query[i], i, ds[i]))
      end
   end
   v('ds', ds)
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
