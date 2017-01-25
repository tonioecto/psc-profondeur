local optim = require 'optim'

local Dataloader = require 'dataloader'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

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
   
   -- set learning rate
   self.optimState.learningRate = self:learningRate(epoch)

   -- set up timer to calculate training time cost
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   -- feval function for optim.stochastic training
   local function feval()
      return self.criterion.output, self.gradParams
   end

   -- size of the input
   local trainSize = #dataloader.trainImageTable

   -- training batch counter 
   local N = 0

   local loss
   local dataTime

   print('=> Training epoch # ' .. epoch)
   
   -- set the batch norm to training mode
   self.model:training()

   local indexbegin = 1
   while(indexbegin < trainSize+1) do
        sample = dataloader:loadDatafromtable(indexbegin)
        indexbegin = indexbegin + dataloader.size

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
            
            -- check that the storage didn't get changed due to an unfortunate getParameters call
            assert(self.params:storage() == self.model:parameters()[1]:storage())

            timer:reset()
            dataTimer:reset()
        end

        -- print training infos
        print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f '):format(
            epoch, N, trainSize, timer:time().real, dataTime, loss))
    end

end

function Trainer:copyInputs(image,depth)
   self.input = image:cuda()
   self.target = depth:cuda()
end

function Trainer:computeScore(validationSet)
    -- Compute error for validation set 

    return 0.1, 0.1
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   end

   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
