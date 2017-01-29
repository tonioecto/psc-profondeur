require 'paths'
local optim = require 'optim'
require 'image'

local M = {}
local Trainer = torch.class('resunpooling.Trainer', M)
local unpack = unpack or table.unpack

function Trainer:__init(model, criterion, optimState, opt)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState
   self.params, self.gradParams = model:getParameters()
   self.batchSize = opt.batchSize
   self.opt = opt
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch

   self.dataloader = dataloader

   -- set learning rate
   self.optimState.learningRate = self:learningRate(epoch)
   print(self.optimState.learningRate)

   -- set up timer to calculate training time cost
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   -- feval function for optim.stochastic training
   local function feval()
      return self.criterion.output, self.gradParams
   end

   -- size of the input
   local trainSize = #self.dataloader.trainImageTable

   -- training batch counter
   local N = 0

   local loss
   local dataTime

   print('=> Training epoch # ' .. epoch)

   -- set the batch norm to training mode
   self.model:training()

   local indexbegin = 1
   while(indexbegin < trainSize+1) do
        sample = self.dataloader:loadDatafromtable(indexbegin)
        indexbegin = indexbegin + self.dataloader.size

        for i = 1, sample:size(), 1 do
            dataTime = dataTimer:time().real

            -- Copy input and target to the GPU
            self:copyInputs(sample.image[i],sample.depth[i])

            local output = self.model:forward(self.input):float()
            local batchSize = output:size(1)
            loss = self.criterion:forward(self.model.output, self.target)

            self.model:zeroGradParameters()
            self.criterion:backward(self.model.output, self.target)
            self.model:backward(self.input, self.criterion.gradInput)

            --print(self.params:size())



            optim.sgd(feval, self.params, self.optimState)

            N = N + batchSize

            -- print training infos
            print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f '):format(
                epoch, N, trainSize, timer:time().real, dataTime, loss))

            -- check that the storage didn't get changed due to an unfortunate getParameters call
            assert(self.params:storage() == self.model:parameters()[1]:storage())

            timer:reset()
            dataTimer:reset()
        end

    end

end

function Trainer:copyInputs(image,depth)
   self.input = image:cuda()
   self.target = depth:cuda()
end

function Trainer:saveLoss(trainErr, valErr)
    local lossFilePath = paths.concat((opt.lossFile), 'training_loss.t7')

    if self.opt.resume == 'none' then
        local trainingTrack = {}
    else
        local trainingTrack = torch.load(lossFilePath)
    end

    local loss = {trainErr, valErr}

    table.insert(trainingTrack, loss)
    torch.save(lossFilePath, trainingTrack)
end

function Trainer:sampleTrainingLoss(num)
	-- sample of size num
	-- self.dataloader:tableShuffle('train')
  local setSize = #self.dataloader.trainImageTable
  local indexTable = torch.randperm(setSize)
  local depthReal = torch.Tensor(num,unpack(self.opt.outputSize))
  local imageSample = tprch.Tensor(num,unpack(self.opt.inputSize))
  for i=1,num,1 do
    imageSample[i] = image.loadJPG(self.dataloader.trainImageTable[indexTable[i]])
    depthReal[i] = image.loadJPG(self.dataloader.trainDepthTable[indexTable[i]])
  end
  local depthPred = self.model:forward(imageSample)
  local loss = self.criterion:forward(depthPred, depthReal)
  return loss
end

function Trainer:computeScore(validationSet)
    -- Compute error for validation set
    local depthPred = self.model:forward(validationSet.image)
    local loss = self.criterion:forward(depthPred,validationSet.depth)

    return loss
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = math.floor((epoch - 1) / 10)

   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
