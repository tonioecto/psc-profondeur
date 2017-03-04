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
    self.sampleSize = opt.sampleSize
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
    -- local trainSize = self.dataloader.dataset:size()
    local trainSize = 10
    
    -- training batch counter
    local N = 0

    -- training loss
    local loss
    local dataTime

    print('=> Training epoch # ' .. epoch)

    -- set the batch norm to training mode
    self.model:training()

    local indexbegin = 1
    while(indexbegin <= trainSize) do
        -- load part of the dataset
        local sz = math.min(self.sampleSize, trainSize - indexbegin + 1)
        local sample = self.dataloader:loadDataset(indexbegin, indexbegin + sz - 1)
        -- convert the dataset to mini batch
        sample = self.dataloader:miniBatchload(sample)
        indexbegin = indexbegin + sz

        for i = 1, sample.size, 1 do
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

-- get tensor type to use
local function getCudaTensorType(tensorType)
    if tensorType == 'torch.CudaHalfTensor' then
        return cutorch.createCudaHostHalfTensor()
    elseif tensorType == 'torch.CudaDoubleTensor' then
        return cutorch.createCudaHostDoubleTensor()
    else
        return cutorch.createCudaHostTensor()
    end
end

-- copy input as different tensor type
function Trainer:copyInputs(image, depth)
    self.input = getCudaTensorType(self.opt.tensorType)
    self.input:resize(image:size()):copy(image)
    self.input = self.input:cuda()
    self.target = getCudaTensorType(self.opt.tensorType)
    self.target:resize(depth:size()):copy(depth)
    self.target = self.target:cuda()
end

-- save training and validation loss after every epoch
function Trainer:saveLoss(epoch, trainErr, valErr)
    local lossFilePath = paths.concat((self.opt.lossFile), 'loss.t7')
    local trainingTrack

    if self.opt.resume == 'none' then
        trainingTrack = {}
    else
        trainingTrack = torch.load(lossFilePath)
    end

    local loss = {epoch, trainErr, valErr}

    table.insert(trainingTrack, loss)
    torch.save(lossFilePath, trainingTrack)
end

-- compute trainingd loss for a sample part of train dataset
function Trainer:sampleTrainingLoss(num)
    -- sample of size num
    if self.split ~= 'train' then
        print ('not a train dataset, cannot sample from '..self.split..' dataset.')
        return nil
    end

    local setSize = #self.dataloader.dataset:size()
    local indexTable = torch.randperm(setSize)
    local depthSample = torch.Tensor(num, unpack(self.opt.outputSize))
    local imageSample = torch.Tensor(num, unpack(self.opt.inputSize))

    for i=1, num, 1 do
        imageSample[i], depthSample[i] = self.dataset.get(indexTable[i])
    end

    local depthPred = self.model:forward(imageSample:cuda())
    local loss = self.criterion:forward(depthPred, depthSample:cuda())
    return loss
end

-- compute score on validation set
function Trainer:computeValScore(valLoader, num)

    -- load permutation table for val set
    valLoader:loadPerm(torch.randperm(valLoader.dataset:size()))
    -- load images and depths
    local valSample = valLoader:loadDataset(1, num)
    local img = valSample.imageSet
    local depth = valSample.depthSet
    -- normalise depth map
    depth = valLoader:normalise(depth, 70)
    
    img = img:cuda()
    depth = depth:cuda()
    
    local pred = self.model:forward(img)
    local loss = self.criterion:forward(pred, depth)
    return loss
end

-- show the prediction of a random image in the dataset 
-- of the loader
function Trainer:showDepth(loader)

    local index = torch.random(loader.dataset:size())
    local element = loader.dataset:get(index)
    local img = element.image:cuda()
    local depth = element.depth:cuda()
    local prediction = self.forward(img):reshape(unpack(opt.outputSize))
    local res = {}
    res.image = img:float()
    res.pred = prediction:float()
    res.groundTruth = depth:float()
    path = paths.concat('result')
    torch.save(path,res)
    
    return res
end

-- decrease learning rate according to epoch
function Trainer:learningRate(epoch)
    -- Training schedule
    local decay = math.floor((epoch - 1) / 6)

    return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
