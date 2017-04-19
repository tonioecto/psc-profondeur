require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local unpack = unpack or table.unpack
local M = {}
local DataLoader = torch.class('resnetUnPooling.DataLoader', M)

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

function DataLoader:__init(dataset, opt, split)

    self.dataset = dataset
    self.info = dataset.info
    self.split = split
    self.sampleSize = opt.sampleSize
    self.batchSize = opt.batchSize
    self.opt = opt

    -- manually generate RNG
    local manualSeed = opt.manualSeed
    local function init()
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        _G.dataset = dataset
        _G.preprocess = dataset:preprocessOnline()
        return dataset:size()
    end

    -- initialize threads
    local threads, size = Threads(opt.nThreads, init, main)
    self.threads = threads
    self.__size = dataset:size()
    self.batchSize = opt.batchSize
    local function getCPUType(tensorType)
        if tensorType == 'torch.CudaHalfTensor' then
            return 'HalfTensor'
        elseif tensorType == 'torch.CudaDoubleTensor' then
            return 'DoubleTensor'
        else
            return 'FloatTensor'
        end
    end
    self.cpuType = getCPUType(opt.tensorType)
end

-- load permutation table
function DataLoader:loadPerm(perms)
    self.perms = perms
end

-- load normalisation info
function DataLoader:loadNormInfo(normInfo)
    self.normInfo = normInfo
end

-- create val and train dataset path file
function DataLoader.create(opt, info)

    -- The train and val loader
    local loaders = {}

    for i, split in ipairs{'train', 'val'} do
        local dataset = datasets.create(opt, split, info)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end

    return table.unpack(loaders)
end

function DataLoader.createTest(opt)
    -- The train and val loader
    local info = 
    for i, split in ipairs{'train', 'val'} do
        local dataset = datasets.create(opt, split, info)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end

    return table.unpack(loaders)
end

--after the transformation offline
--Load the images and depthMap, and generate dataset for training
--load a part of dataset from startIndex to endIndex randomly
function DataLoader:loadDataset(startIndex, endIndex, flag)

    local imagePath = self.info.imagePath
    local depthPath = self.info.depthPath

    print('=> The total number of image is:'..#imagePath)
    print('=> The total number of correponding depth map is:'..#depthPath)
    print('=> load '..self.split..' dataset from index '..startIndex..' to '..endIndex)

    local size = endIndex - startIndex + 1

    local imageSet = torch.Tensor(size, unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(size, unpack(self.opt.outputSize))

    for i = 1, size, 1 do
        local index = self.perms[i]
        local element = self.dataset:get(index)
        imageSet[i]:copy(element.image)
        depthSet[i]:copy(element.depth)
    end

    local datasetSample = {
        image = imageSet,
        depth = depthSet,
        size =  function()
            return imageSet:size()[1]
        end
    }

    -- normalise image and depth map
    -- depth map is normalised in [0, 1]
    if flag ~= 'norm' then
        self:normalise(datasetSample, 70)
    end

    setmetatable(datasetSample,
    {__index = function(t, i)
        return {t.image[i], t.depth[i]}
    end}
    )

    return datasetSample
end

--create mini batch after offline transformation preprocess
--for every source image and depth
function DataLoader:miniBatchload(dataset)

    local numBatch = math.floor(dataset:size() / self.batchSize)

    local imageBatchs = dataset.image:narrow(1, 1, numBatch * self.batchSize)
    imageBatchs = imageBatchs:view(numBatch, self.batchSize, table.unpack(self.opt.inputSize))
    local depthBatchs = dataset.depth:narrow(1, 1, numBatch * self.batchSize)
    depthBatchs = depthBatchs:view(numBatch, self.batchSize, table.unpack(self.opt.outputSize))

    local dataBatchSample = {
        image = imageBatchs,
        depth = depthBatchs,
        size = numBatch
    }

    setmetatable(dataBatchSample,
    {__index = function(t, i)
        return {t.image[i], t.depth[i]}
    end}
    )

    return dataBatchSample
end

function DataLoader:computeNormInfo()

    print('=> start to compute norm info')
    -- pick a temporary permutation table
    self:loadPerm(torch.randperm(self.dataset:size()))

    -- load entire dataset
    local data = self:loadDataset(1, self.__size, 'norm')

    imgMean = {} -- store the mean, to normalize the test set in the future
    imgStd  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        imgMean[i] = data.image[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        print('Channel ' .. i .. ', Mean: ' .. imgMean[i])

        imgStd[i] = data.image[{ {}, {i}, {}, {}  }]:std() -- std estimation
        print('Channel ' .. i .. ', Standard Deviation: ' .. imgStd[i])
    end

    local normInfo = {}
    normInfo.imgMean = imgMean
    normInfo.imgStd = imgStd

    return normInfo
end

-- normalise data
function DataLoader:normalise(data, coef)

    assert(self.normInfo ~= nil, 'normalisation info is not yet computed')

    local mean = self.normInfo.imgMean
    local stdv  = self.normInfo.imgStd
    for i=1, 3 do -- over each image channel
        data.image[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        data.image[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    data.depth:div(coef)

    return data
end

function DataLoader:normaliseImage(image)
    local mean = self.normInfo.imgMean
    for i=1, 3 do -- over each image channel
        image[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        image[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    return image
end

function DataLoader:denormaliseDepth(depth,coef)
    return depth:mul(coef)
    return depth
end



-- denormalise data
function DataLoader:denormalise(data, coef)

    assert(self.normInfo ~= nil, 'normalisation info is not yet computed')

    local mean = self.normInfo.imgMean
    local stdv  = self.normInfo.imgStd
    for i=1, 3 do -- over each image channel
        data.image[{ {}, {i}, {}, {}  }]:add(mean[i]) -- add mean
        data.image[{ {}, {i}, {}, {}  }]:mul(stdv[i]) -- std multiplication
    end

    data.depth:mul(coef)
    return data
end

function DataLoader:normaliseSingle(data, coef)
    assert(self.normInfo ~= nil, 'normalisation info is not yet computed')

    local mean = self.normInfo.imgMean
    local stdv  = self.normInfo.imgStd
    for i=1, 3 do -- over each image channel
        data.image[{ {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        data.image[{ {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    data.depth:div(coef)

    return data
end

function DataLoader:denormaliseSingle(data, coef)
    assert(self.normInfo ~= nil, 'normalisation info is not yet computed')

    local mean = self.normInfo.imgMean
    local stdv  = self.normInfo.imgStd
    for i=1, 3 do -- over each image channel
        data.image[{ {i}, {}, {}  }]:add(mean[i]) -- add mean
        data.image[{ {i}, {}, {}  }]:mul(stdv[i]) -- std multiplication
    end

    data.depth:mul(coef)
    return data
end

-----------------------Multithreads part-------------------------

-- multi threads solution to get dataset batchs for start to end
function DataLoader:run(starIndex, endIndexss)
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    local perm = self.perms

    local idx, sample = startIndex, nil
    local function enqueue()
        while idx <= endIndex and threads:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(batchSize, endIndex - idx + 1))

            threads:addjob(
            function(indices, cpuType)
                -- final batch size
                local sz = indices:size(1)
                local images, depths
                local imageSize, depthSize
                for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    local input, output = _G.preprocess(sample.image, sample.depth)
                    if not images then
                        imageSize = input:size():totable()
                        images = torch[cpuType](sz, table.unpack(imageSize))
                    end
                    if not depths then
                        depthSize = output:size():totable()
                        depths = torch[cpuType](sz, table.unpack(depthSize))
                    end
                    images[i]:copy(input)
                    depths[i]:copy(output)
                end
                collectgarbage()
                return {
                    image = images,
                    depth = depths,
                }
            end,
            function(_sample_)
                sample = _sample_
            end,
            indices,
            self.cpuType
            )

            idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
