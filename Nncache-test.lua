-- Nncache-test.lua

require 'all'

tester = Tester()
test = {}

function makeNncache()
   local v = makeVerbose(false, 'makeNncache')
   v('Nncache', Nncache)
   local nnc = Nncache()
   v('nnc', nnc)
   local nDims = 3

   local value1 = torch.Tensor(nDims):fill(1)
   nnc:setLine(1, value1)

   local value2 = torch.Tensor(nDims):fill(2)
   nnc:setLine(2, value2)

   return nnc
end -- makeNncache

function testElements(nnc)
   local results1 = nnc:getLine(1)
   tester:asserteq(1, results1[1])

   local results2 = nnc:getLine(2)
   tester:asserteq(2, results2[3])

   local results3 = nnc:getLine(3)
   tester:assert(results3 == nil)
end -- nnc

function test.apply()
   local nnc = makeNncache()
   local function f(key, value)
      tester:assert(key == 1 or key == 2)
      if key == 1 then
         tester:assert(1, value[1])
      elseif key == 2 then
         tester:asserteq(2, value[2])
      else
         tester:assert(false, 'bad value = ' .. tostring(value))
      end
   end
end -- apply

function test.get()
   local nnc = makeNncache()
   testElements(nnc)
end -- get

function test.set()
   local nnc = makeNncache()

   -- attempt to set existing obsIndex
   function call(obsIndex, value)
      nnc:set(obsIndex, value)
   end
   local value = torch.Tensor(3):fill(27)
   status, result  = pcall(call, 1, value)
   tester:assert(status == false)
   tester:assert(type(result) == 'string')
end -- set

function test.saveLoad()
   local v = makeVerbose(false, 'test.saveLoad')
   local nnc = makeNncache()
   local filePath = '/tmp/Nncache-test.txt'
   nnc:save(filePath)

   nnc = nil
   fromDisk = Nncache.load(filePath)
   v('fromDisk', fromDisk)
   testElements(fromDisk)
end -- writeRead

tester:add(test)
tester:run(true) -- true ==> verbose