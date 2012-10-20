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
   Nncachebulder.mergeShards(nShards, 'filePathPrefix')

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

function Nncachebuilder:__init(allXs, nShards)
   local v, isVerbose = makeVerbose(false, 'Nncachebuilder:__init')
   verify(v, isVerbose,
          {{allXs, 'allXs', 'isTensor2D'},
           {nShards, 'nShards', 'isIntegerPositive'}})
   -- an index must fit into an integer
   assert(allXs:size(1) <= 2147483647,  -- about 2 billion
          'more than 2^31 - 1 rows in the tensor')
   self._allXs = allXs
   self._nShards = nShards
   self._cache = nil  -- _cache[index]=<1D tensor of obsIndices in tensor2D>

   self._kernelSmoother = KernelSmoother()
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

function Nncachebuilder.maxNeighbors()
   -- number of neighbor indices stored; size of cache[index]
   return 256
end

--------------------------------------------------------------------------------
-- PRIVATE CLASS METHODS
--------------------------------------------------------------------------------

function Nncachebuilder._format()
   -- format used to serialize the cache
   return 'ascii' -- 'binary' is faster
end -- _format

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
   local v, isVerbose = makeVerbose(true, 'createShard')
   verify(v, isVerbose,
          {{shardNumber, 'shardNumber', 'isIntegerPositive'},
           {filePathPrefix, 'filePathPrefix', 'isString'}})
   v('self', self)
   assert(shardNumber <= self._nShards)

   local tc = TimerCpu()
   local cache = {}
   local count = 0
   local shard = 0
   local roughCount = self._allXs:size(1) / self._nShards
   for obsIndex = 1, self._allXs:size(1) do
      shard = shard + 1
      if shard > self._nShards then
         shard = 1
      end
      if shard == shardNumber then
         -- observation in shard, so create its neighbors indices
         local query = self._allXs[obsIndex]:clone()
         collectgarbage()
         local _, allIndices = KernelSmoother.nearest(self._allXs,
                                                      query)
         -- NOTE: creating a view of the storage seems like a good idea
         -- but fails when the tensor is serialized out
         local n = math.min(Nncachebuilder.maxNeighbors(), self._allXs:size(1))
         local firstIndices = torch.Tensor(n)
         for i = 1, n do
            firstIndices[i] = allIndices[i]
         end
         cache[obsIndex] = firstIndices
         count = count + 1
         if false then 
            v('count', count)
            v('obsIndex', obsIndex)
            v('firstIndices', firstIndices)
         end
         if count % 10000 == 1 then
            local rate = tc:cumSeconds() / count
            print(string.format(
                     'Nncachebuilder:createShard: create %d indices' ..
                        ' at %f CPU sec each',
                     count, rate))
            local remaining = roughCount - count
            print(string.format('need %f CPU hours to finish remaining %d',
                                rate * remaining / 60 / 60, remaining))
            --halt()
         end
      end
   end
   v('count', count)
   -- halt()

   -- write by serializing
   local filePath = 
      filePathPrefix .. Nncachebuilder._shardFileSuffix(shardNumber)
   v('filePath', filePath)
   torch.save(filePath, cache, Nncachebuilder._format())
   return filePath
end -- createShard


function Nncachebuilder.mergeShards(nShards, filePathPrefix)
   -- RETURN
   -- number of records in merged file
   -- file path where merged cache data were written
   local v, isVerbose = makeVerbose(false, 'mergeShards')
   verify(v, isVerbose,
          {{nShards, 'nShards', 'isIntegerPositive'},
           {filePathPrefix, 'filePathPrefix', 'isString'}})

   local cache = {}
   local countAll = 0
   for n = 1, nShards do
      local path = filePathPrefix .. Nncachebuilder._shardFileSuffix(n)
      print('reading shard cache file ', path)
      local shard = torch.load(path, Nncachebuilder._format())
      affirm.isTable(shard, 'shard')
      local countShard = 0
      for key, value in pairs(shard) do
         countShard = countShard + 1
         countAll = countAll + 1
         assert(cache[key] == nil)  -- no duplicates across shards
         cache[key] = value
      end
      print('number records inserted from shard', countShard)
   end
   print('number of records inserted from all shards', countAll)

   local mergedFilePath = filePathPrefix .. Nncachebuilder._mergedFileSuffix()
   print('writing merged cache file', mergedFilePath)
   torch.save(mergedFilePath, cache, Nncachebuilder._format())
   return countAll, mergedFilePath
end -- mergeShards

--------------------------------------------------------------------------------
-- PRIVATE INSTANCE METHODS
--------------------------------------------------------------------------------

function Nncachebuilder:_shardFilePath(filePathPrefix, shardNumber)
   return filePathPrefix .. string.format('shard-%d.txt', shardNumber)
end -- _shardFilePath
