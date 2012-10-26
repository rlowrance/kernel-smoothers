-- test-hastie.lua
-- replicate the example in Hastie-01 that compares losses on
-- various kernel smoothers

require 'all'

--------------------------------------------------------------------------------
-- FUNCTIONS
--------------------------------------------------------------------------------

function generateData(options, log)
   -- return data table with components xs and ys
   -- from hastie-01 page 166:
   -- Y = sin(4X) + epsilon
   -- X ~ Uniform[0,1]
   -- epsilon ~ N(0, 1/3)
   local v, isVerbose = makeVerbose(true, 'generateData')
   verify(v,
          isVerbose,
          {{options, 'options', 'isTable'},
           {log, 'log', 'isTable'}})
   local xs = torch.Tensor(options.nObs, 1)
   local ys = torch.Tensor(options.nObs)
   for obsIndex = 1, options.nObs do
      local x= torch.uniform(0,1)
      local epsilon = torch.normal(0, 1/3)
      local y = math.sin(4 * x)
      xs[obsIndex][1] = x
      ys[obsIndex] = y
   end
   local data = {}
   data.xs = xs
   data.ys = ys
   v('data', data)
   return data
end -- generateData

function modelFit(alpha, removedFold, kappa, 
                  trainingXs, trainingYs, 
                  modelState, options)
   -- return model training on data not in the removed fold
   local v, isVerbose = makeVerbose(true, 'modelFit')
   verify(v, isVerbose,
          {{alpha, 'alpha', 'isSequence'},
           {removedFold, 'removedFold', 'isIntegerPositive'},
           {kappa, 'kappa', 'isSequence'},
           {trainingXs, 'trainingXs', 'isTensor2D'},
           {trainingYs, 'trainingYs', 'isTensor1D'},
           {modelState, 'modelState', 'isTable'},
           {options, 'options', 'isTable'}})

   local nObs = trainingYs:size(1)
   assert(nObs == trainingXs:size(1))

   -- build visible tensor
   local visible = torch.Tensor(trainingYs:size(1)):fill(1)
   for i = 1, nObs do
      if kappa[i] == removedFold then
         visible[i] = 0
      end
   end

   -- construct the model
   local name = alpha[1]
   local radius = alpha[2]
   if name == 'knn' then
      local model = SmootherAvg(trainingXs, 
                                trainingYs,
                                visible,
                                modelState.nncache)
      return model
   elseif name == 'kwavg'then
      local model = SmootherKwavg(trainingXs, 
                                  trainingYs,
                                  visible,
                                  modelState.nncache,
                                  'epanechnikov quadratic')
      return model
   elseif name == 'llr' then
      local model = SmootherLlr(trainingXs, 
                                trainingYs,
                                visible,
                                modelState.nncache,
                                'epanechnikov quadratic')
      return model
   else
      error('bad name = ' .. tostring(name))
   end
end -- modelFit

function modelUse(alpha, model, queryIndex, modelState, options)
   -- return true, estimate from trained model (built to alpha specs)
   local v, isVerbose = makeVerbose(true, 'modelUse')
   verify(v, isVerbose,
          {{alpha, 'alpha', 'isSequence'},
           {model, 'model', 'isNotNil'},
           {queryIndex, 'queryIndex', 'isIntegerPositive'},
           {modelState, 'modelState', 'isTable'},
           {options, 'options', 'isTable'}})
   local name = alpha[1]
   local radius = alpha[2]
   local regularizer = alpha[3] -- nil if name ~= 'Llr'
   if name == 'knn' or name == 'kwavg' then
      return model:estimate(queryIndex, radius)
   elseif name == 'llr' then
      local params = {}
      params.k = radius
      params.regularizer = regularizer
      return model:estimate(queryIndex, params)
   else
      error('bad name = ' .. tostring(name))
   end
end -- modelUse

function splitData(allData, options, log)
   -- return testData, trainingData
   local v, isVerbose = makeVerbose(true, 'splitData')
   verify(v,
          isVerbose,
          {{allData, 'allData', 'isTable'},
           {options, 'options', 'isTable'},
           {log, 'log', 'isTable'}})

   local isTest = {}
   local nTest = 0
   for i = 1, options.nObs do
      if torch.uniform(0,1) < options.fractionTest then
         isTest[i] = true
         nTest = nTest + 1
      end
   end

   local testXs = torch.Tensor(nTest, 1)
   local testYs = torch.Tensor(nTest)
   local testIndex = 0
   local trainingXs = torch.Tensor(allData.xs:size(1) - nTest, 1)
   local trainingYs = torch.Tensor(allData.ys:size(1) - nTest)
   local trainingIndex = 0
   for i = 1, options.nObs do
      if isTest[i] then
         testIndex = testIndex + 1
         testXs[testIndex][1] = allData.xs[i]
         testYs[testIndex] = allData.ys[i]
      else
         trainingIndex = trainingIndex + 1
         trainingXs[trainingIndex][1] = allData.xs[i]
         trainingYs[trainingIndex] = allData.ys[i]
      end
   end

   log:log('split %d observations', options.nObs)
   log:log(' %d (%f) to test', testIndex, testIndex / options.nObs)
   log:log(' %d (%f) to training', trainingIndex, trainingIndex / options.nObs)
   assert(testIndex + trainingIndex == options.nObs)
   
   testData = {}
   testData.xs = testXs
   testData.ys = testYs

   trainingData = {}
   trainingData.xs = trainingXs
   trainingData.ys = trainingYs

   return testData, trainingData
   
end -- splitData


--------------------------------------------------------------------------------
-- MAIN
--------------------------------------------------------------------------------

local v = makeVerbose(true, 'main')

local debug
debug = 0  -- no debugging code is active

local options  =
   mainStart(arg,
             'compare kernel smoother accuracy on hastie example',
             {{'-cvLoss', 'abs', 'alt is squared'},
              {'-dataDir', '../../data/', 'path to data directory'},
              {'-debug', 0, '0 for no debugging code'},
              {'-fractionTest', 0.20, 
               'fraction of data reserved for testing'},
              {'-nObs', 100, 'number of observation'},
              {'-programName', 'test-hastie', 'Name of program'},
              {'-seed', 27, 'random number seed'},
              {'-test', 1, '0 for production, 1 to test'}})

local log = options.log

if  options.debug > 0 then 
   options.debug = 1
   log:log('DEBUGGING: toss results')
end

-- validate options
assert(options.cvLoss == 'abs' or options.cvLoss == 'squared',
       'invalid options.cvLoss')
assert(options.test == 0 or options.test == 1,
       'invalid options.test')
   
if options.test == 1 then
   log:log('TESTING')
end

if true then
   setRandomSeeds(options.seed)
else
   log:log('STUB')
end

-- maybe limit number of training observations read
local inputLimit = 0   -- 0 ==> no limit
if options.test == 1 then
   -- if testing, read only some of the input
   inputLimit = 1000
end

-- generate all the data
local allData = generateData(options, log)

-- split into test and training
local testData, trainingData = splitData(allData, options, log)
v('testData', testData)
v('trainingData', trainingData)

-- build the nearest neighbors cache
local nShards = 1
local nncb = Nncachebuilder(trainingData.xs, nShards)
local filePathPrefix = '/tmp/test-hastie-'
nncb:createShard(1, filePathPrefix)
Nncachebuilder.mergeShards(nShards, filePathPrefix)
local nncache = Nncache.loadUsingPrefix(filePathPrefix)



-- perform crossValidation to select best model

local regularizer = 1e-7
alphas = {{'knn', 1},
          {'knn', 10},
          {'knn', 20},
          {'kwavg', 1},
          {'kwavg', 10},
          {'kwavg', 20},
          {'llr', 3, regularizer},
          {'llr', 10, regularizer},
          {'llr', 20, regularizer}}

local nFolds = 5
local nTrainingObs = trainingData.xs:size(1)
local modelState = {}
modelState.nncache = nncache

local alphaStar, lossTable, coverageTable = crossValidation(alphas,
                                                            nFolds,
                                                            nTrainingObs,
                                                            trainingData.xs,
                                                            trainingData.ys,
                                                            modelFit,
                                                            modelUse,
                                                            modelState,
                                                            options)

v('alphaStar', alphaStar)
v('lossTable', lossTable)
v('coverageTable', coverageTable)

print(' algo  radius   loss coverage')
for key in pairs(lossTable) do
   print(string.format('%5s   %5.1f %6.4f   %6.4f',
                       key[1], key[2], lossTable[key], coverageTable[key]))
end

print('best smoother', alphaStar[1])
print('best radius', alphaStar[2])
if alphaStar[3] ~= nil then
   print('best regularizer', alphaStar[3])
end



log:log('STUB: determine test error using best model')

printOptions(options, log)

if options.test == 1 then
   log:log('TESTING')
   log:log('input limit %d', inputLimit)
end

log:log('consider comiting the code')
