require 'paths'
local optim = require 'optim'
require 'image'
require 'gnuplot'

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

local function plot(n, loss, frequence, lossTrace, toShow)
    lossTrace[torch.floor(n/frequence)] = loss / frequence
    if toShow then
        gnuplot.plot(lossTrace)
    end
end

function Trainer:train(epoch, dataloader, lossTrace, frequence)
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
    local trainSize = self.dataloader.dataset:size()

    -- training batch counter
    local N = 0
    local n = 0

    -- training loss
    local loss
    local dataTime
    local lossAverage = 0

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
            n = n + 1
            lossAverage = lossAverage + loss

            if n%frequence==0 then
                plot(n, lossAverage, frequence, lossTrace, self.opt.plot)
                lossAverage = 0
            end

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
function Trainer:saveLoss(__epoch, __valErr, __lossTrace)

    if not paths.dirp(paths.concat(self.opt.lossFile)) then
        paths.mkdir(paths.concat(self.opt.lossFile))
    end

    local lossFilePath = paths.concat((self.opt.lossFile), 'loss.t7')
    local trainingTrack

    if paths.filep(lossFilePath) then
        trainingTrack = torch.load(lossFilePath)
    else
        trainingTrack = {}
    end

    local loss = {
        epoch = __epoch,
        valErr = __valErr,
        lossTrace = __lossTrace:float()
    }

    table.insert(trainingTrack, loss)
    torch.save(lossFilePath, trainingTrack)
end

-- compute trainingd loss for a sample part of train dataset
function Trainer:sampleTrainingLoss(num)
    print('==> calculate loss of a trainset sample randomly of size '..num)
    -- sample of size num
    if self.dataloader.split ~= 'train' then
        print ('not a train dataset, cannot sample from '..self.split..' dataset.')
        return nil
    end

    -- load images and depths
    local trainSample = self.dataloader:loadDataset(1, num)
    local img = trainSample.image
    local depth = trainSample.depth

    img = img:cuda()
    depth = depth:cuda()

    local loss = 0

    for i=1, num, 1 do
        local pred = self.model:forward(img[i])
        loss = loss + self.criterion:forward(pred, depth[i])
    end

    return loss/num
end

-- compute score on validation set
function Trainer:computeValScore(epoch, valLoader, num)

    print('==> calculate val loss from a val sample randomly of size # '..num)

    local PROGRESS_INDICATOR = 10

    -- load permutation table for val set
    valLoader:loadPerm(torch.randperm(valLoader.dataset:size()))
    -- load images and depths
    local valSample = valLoader:loadDataset(1, num)
    local loss = 0
    local pred

    for i=1, num, 1 do
        pred = self.model:forward(valSample.image[i]:cuda())
        loss = loss + self.criterion:forward(pred, valSample.depth[i]:cuda())
        -- print val error infos
        if i % PROGRESS_INDICATOR == 0 then
            print((' | Epoch: [%d][%d/%d] accumulated validation error:  %1.6f '):format(
            epoch, i, num, loss))
        end
    end

    collectgarbage()

    return loss / num
end

-- show the prediction of a random image in the dataset
-- of the loader
function Trainer:predict(num, img, depth, dataloader)

    local res = {}
    res.image = img:float()
    img = img:cuda()
    local prediction = self.model:forward(img)
    prediction = dataloader:denormalise(prediction, 70)
    res.pred = prediction:float()
    res.groundTruth = depth:float()
    path = paths.concat('result','version_t7', 'visual-epoch-'..self.opt.epochNumber..'-example'..num..'.t7')
    if not paths.dirp(paths.concat('result', 'version_t7')) then
        paths.mkdir(paths.concat('result', 'version_t7'))
    end
    torch.save(path,res)

    return res
end

-- decrease learning rate according to epoch
function Trainer:learningRate(epoch)
    -- Training schedule recipe
    local decay = math.floor((epoch - 1) / 5)

    return self.opt.LR * math.pow(0.5, decay)
end

return M.Trainer
