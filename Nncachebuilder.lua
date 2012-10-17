-- Nncachebuilder.lua
-- cache for nearsest 256 neighbors

-- API overview
if false then
   nnc = Nncachebuilder(allXs, nShards)
   for n = 1, nShards do
      -- serialize cache object (a table) to file <prefix>nncache-shard-n.txt
      nnc:createShard(n, 'filePathPrefix')
   end
   -- create serialized cache object int file <prefix>nncache-merged.txt
   nnc:mergeShards('filePathPrefix')

   -- read the serialized merged cache from file system
   cache = Nncachebuilder.read('filePathPrefix') 
   -- now cache[27] is a 1D tensor of the sorted indices closest to obs # 27
   -- in the original xs
   
   -- use the cache to smooth values
   selected = setSelected() -- to selected observations in Xs and Ys
   knnSmoother = KnnSmoother(allXs, allYs, selected, cache) -- use original Xs!
   estimate = knnSmoother:estimate(queryIndex, k)
end

--------------------------------------------------------------------------------
-- CONSTRUCTION
--------------------------------------------------------------------------------

torch.class('Nncachebuilder')

function Nncachebuilder:__init(tensor2D, nShards)
   local v, isVerbose = makeVerbose(false, 'Nncachebuilder:__init')
   verify(v, isVerbose,
          {{tensor2D, 'tensor2D', 'isTensor2D'},
           {nShards, 'nShards', 'isIntegerPositive'}})
   -- an index must fit into an integer
   assert(tensor2D:size(1) <= 2147483647,  -- about 2 billion
          'more than 2^31 - 1 rows in the tensor')
   self._tensor2D = tensor2D
   self._nShards = nShards
   self._cache = nil  -- _cache[index]=<1D tensor of obsIndices in tensor2D>
end -- __init

--------------------------------------------------------------------------------
-- PUBLIC CLASS METHODS
--------------------------------------------------------------------------------

function Nncachebuilder.read(filePathPrefix)
   -- return an Nncachebuilder that was serialized to a file
   local v, isVerbose = makeVerbose(false, 'Nncachebuilder.read')
   verify(v, isVerbose,
          {{filePathPrefix, 'filePathPrefix', 'isString'}})
   local inPath = filePathPrefix .. Nncachebuilder._mergedFileSuffix()
   v('inPath', inPath)
   local nncache = torch.load(inPath,
                              Nncachebuilder._format())
   affirm.isTable(nncache, 'nncache')
   -- can't figure out how to test more than the type
   -- It may be that there are fewer than 256 rows, as the original allXs
   -- may only have a few rows.
   v('nncache', nncache)
   return nncache
end -- read

--------------------------------------------------------------------------------
-- PRIVATE CLASS METHODS
--------------------------------------------------------------------------------

function Nncachebuilder._format()
   -- format used to serialize the cache
   return 'ascii' -- 'binary' is faster
end -- _format

function Nncachebuilder._maxNeighbors()
   -- number of neighbor indices stored; size of cache[index]
   return 256
end -- _maxNeighbors

function Nncachebuilder._mergedFileSuffix()
   -- end part of file name
   return 'nncache-merged.txt'
end -- _mergedFileSuffix

function Nncachebuilder._shardFileSuffix(n)
   -- end part of file name
   return string.format('nncache-shard-%d.txt', n)
end -- _shardFileSuffix

--------------------------------------------------------------------------------
-- PUBLIC INSTANCE METHODS
--------------------------------------------------------------------------------

function Nncachebuilder:createShard(shardNumber, filePathPrefix)
   -- return file path where cache data were written
   local v, isVerbose = makeVerbose(false, 'createShard')
   verify(v, isVerbose,
          {{shardNumber, 'shardNumber', 'isIntegerPositive'},
           {filePathPrefix, 'filePathPrefix', 'isString'}})
   v('self', self)
   assert(shardNumber <= self._nShards)

   local cache = {}
   local count = 0
   local shard = 0
   for obsIndex = 1, self._tensor2D:size(1) do
      shard = shard + 1
      if shard == shardNumber then
         cache[obsIndex] = self:_nearestIndices(obsIndex)
         count = count + 1
      end
      if shard == self._nShards then
         shard = 0
      end
   end
   v('count', count)

   -- write by serializing
   local filePath = 
      filePathPrefix .. Nncachebuilder._shardFileSuffix(shardNumber)
   v('filePath', filePath)
   torch.save(filePath, cache, Nncachebuilder._format())
   return filePath
end -- createShard


function Nncachebuilder:mergeShards(filePathPrefix)
   -- return file path where merged cache data were written
   local v, isVerbose = makeVerbose(false, 'mergeShards')
   verify(v, isVerbose,
          {{filePathPrefix, 'filePathPrefix', 'isString'}})
   local cache = {}
   for n = 1, self._nShards do
      local path = filePathPrefix .. Nncachebuilder._shardFileSuffix(n)
      print('reading shard cache file ', path)
      local shard = torch.load(path, Nncachebuilder._format())
      check.isTable(shard, 'shard')
      local count = 0
      for key, value in pairs(shard) do
         count = count + 1
         assert(cache[key] == nil)  -- do duplicates across shards
         cache[key] = value
      end
      print('number records inserted from shard', count)
   end

   local mergedFilePath = filePathPrefix .. Nncachebuilder._mergedFileSuffix()
   print('writing merged cache file', mergedFilePath)
   torch.save(mergedFilePath, cache, Nncachebuilder._format())
   return mergedFilePath
end -- mergeShards

--------------------------------------------------------------------------------
-- PRIVATE INSTANCE METHODS
--------------------------------------------------------------------------------

function Nncachebuilder:_nearestIndices(obsIndex)
   local v, isVerbose = makeVerbose(false, 'Nncachebuilder:_nearest')
   verify(v, isVerbose,
          {{obsIndex, 'obsIndex', 'isIntegerPositive'}})
   --v('self._t', self._t)

   -- 1. determine distances from obsIndex to all other indices
   local query = self._tensor2D[obsIndex]:clone()

   -- create a 2D Tensor where each row is the query
   -- This construction is space efficient relative to replicating query
   -- queries[i] == query for all i in range
   -- Thanks Clement Farabet!
   local queries = 
      torch.Tensor(query:storage(),             -- reuse storage
                   1,                           -- offset
                   self._tensor2D:size(1), 0,   -- row index offset and stride
                   self._tensor2D:size(2), 1)   -- col index offset and stride
      
   local distances = torch.add(queries, -1 , self._tensor2D) -- queries - xs
   distances:cmul(distances)                                 -- (queries - xs)^2
   distances = torch.sum(distances, 2):squeeze() -- \sum (queries - xs)^2
   distances = distances:sqrt()                  -- Euclidean distances
  
   v('distances', distances)
   
   -- 2. Determine indices to sort the distances
   local _, sortedIndices = torch.sort(distances)
   v('sortedIndices', sortedIndices)
   
   -- 3. Convert to integer tensor to save space
   -- skip for now
   local result
   if false then
      -- skip for now
      result = torch.IntTensor(Nncachebuilder._maxNeighbors())
      for i = 1, Nncachebuilder._maxNeighbors() do
         result[i] = sortedIndices[i]
      end
   else
      -- for now, just string to maxNeighbors size
      result = torch.Tensor(Nncachebuilder._maxNeighbors())
      for i = 1, Nncachebuilder._maxNeighbors() do
         result[i] = sortedIndices[i]
      end
   end

   v('result', result)
   --if obsIndex > 10 then stub('check') end

   return result
end -- _nearest

function Nncachebuilder:_shardFilePath(filePathPrefix, shardNumber)
   return filePathPrefix .. string.format('shard-%d.txt', shardNumber)
end -- _shardFilePath
