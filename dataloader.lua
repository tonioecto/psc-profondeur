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

-- load permutation table
function DataLoader:loadPerm(perms)
    self.perms = perms
end

-- create val and train dataset path file
function DataLoader.create(opt)

    -- The train and val loader
    local loaders = {}

    for i, split in ipairs{'train', 'val'} do
        local dataset = datasets.create(opt, split)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end

    return table.unpack(loaders)
end

-- load dataset from path tables
function DataLoader:__init(dataset, opt, split)
    self.dataset = dataset
    self.info = data.info
    self.split = split
    self.sampleSize = opt.sampleSize
    self.batchSize = opt.batchSize
    self.opt = opt
end

--transformation offline
--Load the images and depthMap, and generate dataset for training
--load a part of dataset from startIndex to endIndex randomly
function DataLoader:loadDataset(startIndex, endIndex)

    local imagePath = self.info.imagePath
    local depthPath = sefl.info.depthPath

    print('=> load '..self.split..' dataset')

    print('The number of image is:'..#imagePath)
    print('The number of correponding depthmap is:'..#depthPath)

    local imageSet = torch.Tensor(#imagePath, unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(#depthPath, unpack(self.opt.outputSize))

    for i = startIndex, endIndex, 1 do
        local index = self.perms[i]
        imageSet[i], depthSet[i] = self.dataset:get(index)
    end

    local datasetSample = {
        image = imageSet,
        depth = depthSet,
        size =  function()
            return imageSet:size(1)
        end
    }

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

    local numBatch = math.ceil(dataset:size() / self.batchSize)

    local imageBatchs = torch.Tensor(numBatch, self.batchSize, unpack(self.opt.inputSize))
    local depthBatchs = torch.Tensor(numBatch, self.batchSize, unpack(self.opt.outputSize))

    for i = 1, numBatch, 1 do
        local batch = math.min(dataset:size() - (i - 1) * self.batchSize, self.batchSize)
        local index = (i - 1) * self.batchsize + 1
        for k = 1, batch, 1 do
            imageBatchs[i][k]:copy(dataset.image[index])
            depthBatchs[i][k]:copy(dataset.depth[index])
            index = index + 1
        end
    end

    local dataBatchSample = {
        image = imageBatchs,
        depth = depthBatchs,
        size = dataset:size()
    }

    setmetatable(dataBatchSample,
    {__index = function(t, i)
        return {t.image[i], t.depth[i]}
    end}
    )

    return dataBatchSample
end

return M.DataLoader

