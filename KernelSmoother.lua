-- KernelSmoother.lua
-- auxillary class for all kernel smoothers
-- used to hold utility functions
-- could have been a parent class

-- API overview
if false then
   ks = KernelSmoother()
   
   distance = ks:euclideanDistances(xs, query) -- distances from query to each x
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

   local v, trace = makeVerbose(false, 'Knn:_euclideanDistances')

   v('self', self)
   v('xs', xs)
   v('query', query)

   affirm.isTensor2D(xs, 'xs')
   affirm.isTensor1D(query, 'query')

   assert(xs:size(2) == query:size(1),
          'number of columns in xs must equal size of query')

   query = query:clone()  -- in case the query is a view of the xs storage
   
   -- create a 2D Tensor where each row is the query
   -- This construction is space efficient relative to replicating query
   -- queries[i] == query for all i in range
   -- Thanks Clement Farabet!
   local queries = 
      torch.Tensor(query:storage(),
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

