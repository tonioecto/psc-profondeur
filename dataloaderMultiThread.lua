require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'

local unpack = unpack or table.unpack
local M = {}
local DataLoader = torch.class('resnetUnPooling.DataLoader', M)

function M.load()
end

-- load dataset from path tables
function DataLoader:__init(info, opt)
    self.info = info
    self.val = info.val
    self.train = info.train
    self.size = opt.sampleSize
    self.batchSize = opt.batchSize
    self.opt = opt
end

--Load the images and depthMap, and generate dataset for training
function DataLoader:loadDataset(split)

    local imagePath
    local depthPath

    if split == "val" then
        print('=> load validation dataset')
        imagePath = self.val.imagePath
        depthPath = self.val.depthPath
    elseif split == "test" then
        print('=> load test dataset')
        imagePath = self.test.imagePath
        depthPath = self.test.depthPath
    elseif s == "train" then
        print('=> load train data set')
        imagePath = self.train.imagePath
        depthPath = self.train.depthPath
    else
        error('Invalid split input '..split)
    end

    print('The number of image is:'..#imagePath)
    print('The number of correponding depthmap is:'..#depthPath)

    local imageSet = torch.Tensor(#imagePath, unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(#depthPath, unpack(self.opt.outputSize))

    for i = 1, #imagePath, 1 do
        local img = image.loadJPG(imagePath[i])
        imageSet[i] = img
        local dep = torch.load(depthPath[i])
        depthSet[i] = dep
    end

    local dataset = {
        image = imageSet,
        depth = depthSet,
        size =  function()
            return imageSet:size(1)
        end
    }

    setmetatable(dataset,
    {__index = function(t, i)
        return {t.image[i], t.depth[i]}
    end}
    )

    return dataset
end

--create mini batch
function DataLoader:miniBatchload(dataset)

    local perm = torch.randperm(dataset:size())

    local numBatch = torch.round(dataset:size() / self.batchSize)
    local imageSet = torch.Tensor(numBatch, self.batchSize, unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(numBatch, self.batchSize, unpack(self.opt.outputSize))

    for index = 1, numBatch, 1 do
        local numRemain = dataset:size() - (index - 1) * self.batchSize
        if(numRemain >= self.batchSize) then
            local indexBegin = (index - 1) * self.batchsize + 1
            local indexEnd = indexBegin + self.batchsize - 1
            for k = 1, self.batchsize, 1 do
                imageSet[{index,k,{}}] = dataset.image[indexBegin]
                depthSet[{index,k,{}}] = dataset.depth[indexBegin]
                indexBegin = indexBegin + 1
            end
        end
    end

    local data = {
        image = imageSet,
        depth = depthSet,
        size =  function() return imageSet:size(1) end
    }

    setmetatable(data,
    {__index = function(t, i)
        return {t.image[i], t.depth[i]}
    end}
    )

    return data
end

return M.DataLoader

