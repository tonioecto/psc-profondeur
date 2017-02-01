require 'paths'
local optim = require 'optim'
require 'image'

local evaluate = require 'evaluate'
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

function Trainer:saveLoss(epoch, trainErr, valErr)
    local lossFilePath = paths.concat((self.opt.lossFile), 'loss.t7')
	local trainingTrack

    if self.opt.resume == 'none' then
        trainingTrack = {}
        self.opt.resume = 'model_trained'
    else
        trainingTrack = torch.load(lossFilePath)
    end

    local loss = {epoch, trainErr, valErr}

    table.insert(trainingTrack, loss)
    torch.save(lossFilePath, trainingTrack)
end

function Trainer:sampleTrainingLoss(num)
	-- sample of size num
	-- self.dataloader:tableShuffle('train')
    local setSize = #self.dataloader.trainImageTable
    local indexTable = torch.randperm(setSize)
    local depthReal = torch.Tensor(num,unpack(self.opt.outputSize))
    local imageSample = torch.Tensor(num,unpack(self.opt.inputSize))
    for i=1,num,1 do
        imageSample[i] = image.loadJPG(self.dataloader.trainImageTable[indexTable[i]])
        depthReal[i] = image.loadJPG(self.dataloader.trainDepthTable[indexTable[i]])
    end
    local depthPred = self.model:forward(imageSample:cuda())
    local loss = self.criterion:forward(depthPred, depthReal:cuda())
    return loss
end

function Trainer:computeScore(validationSet)
    -- Compute error for validation set
    local depthPred = self.model:forward(validationSet.image:cuda())
    local loss = self.criterion:forward(depthPred,validationSet.depth:cuda())

    return loss
end

function Trainer:showDepth(str,num)
  if str == "train" then
    for i=1,num,i do
      local rand = math.random(#self.dataloader.trainImageTable)
      local depthPred = self.model:forward(image.loadJPG(self.dataloader.trainImageTable[rand]):cuda())
      local Pred = depthPred[1]
      local depthReal = image.loadJPG(self.dataloader.trainDepthTable[rand])
      local Real = depthReal[1]
      evaluate.Display(Pred,Real)
    end
  elseif str == "val" then
    for i=1,num,i do
      local rand = math.random(#self.dataloader.valImageTable)
      local depthPred = self.model:forward(image.loadJPG(self.dataloader.valImageTable[rand]):cuda())
      local depthReal = image.loadJPG(self.dataloader.valDepthTable[rand])
      local Pred = depthPred[1]
      local Real = depthReal[1]
      evaluate.Display(Pred,Real)
    end
  end
end




function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = math.floor((epoch - 1) / 10)

   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
