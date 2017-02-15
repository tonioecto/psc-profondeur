require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'

local unpack = unpack or table.unpack
local M = {}
local DataLoader = torch.class('resnetUnPooling.DataLoader', M)

function M.load()
    local trainSet
    local validationSet
    return trainSet, validationSet
end


function DataLoader:__init(imageset, depthset,opt)
    self.imageset =imageset
    self.depthset = depthset
    self.size = opt.sampleSize
    self.batchsize = opt.batchSize
    self.opt = opt
end

function DataLoader:loadDataset(s)       --Load the images and depthMap, and generate dataset for trainning
    local imagetable = {}
    local depthtable = {}
    if s=="val" then
        print('loading validation dataset')
        imagetable = self.valImageTable
        depthtable = self.valDepthTable
    elseif s=="test" then
        print('loading test dataset')
        imagetable = self.testImageTable
        depthtable = self.testDepthTable
    end
    print('The number of image is:'..#imagetable)
    print('The number of correponding depthmap is:'..#depthtable)

    if #self.imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    local imageSet = torch.Tensor(#imagetable,unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(#depthtable,1,unpack(self.opt.outputSize))
    --local mat = require 'matio'

    for i,file in ipairs(imagetable) do
        local m = image.loadJPG(file)
        --m = image.scale(m,304,228,'bicubic')
        imageSet[i] = m
    end

    for i,file in ipairs(depthtable) do
        --local m = mat.load(file,'depthMap')
        --m = image.scale(m,128 ,160,'bicubic')
        local m = image.loadJPG(file)
        depthSet[i] = m
    end

    local dataset = {
        image = imageSet,
        depth = depthSet,
        size =  function() return imageSet:size(1) end
    }

    setmetatable(dataset,
    {__index = function(t, i)
        return {t.image[i], t.depth[i]}
    end}
    )

    return dataset

end


function DataLoader:miniBatchload(dataset)   --create mini batch
    --randomize the data firstly
    --local shuffle = torch.randperm(dataset:size())
    --local imageSize = dataset.image[1]:size()
    --local depthSize = dataset.depth[1]:size()

    local numBatch = math.floor(dataset:size()/self.batchsize)
    --print(numBatch)
    local imageSet = torch.Tensor(numBatch,self.batchsize,unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(numBatch,self.batchsize,unpack(self.opt.outputSize))

    for index = 1,numBatch,1 do
        local numRemain = dataset:size() - (index-1)*self.batchsize
        if(numRemain >=self.batchsize) then
            local indexBegin = (index-1)*self.batchsize + 1
            local indexEnd = indexBegin + self.batchsize - 1
            for k = 1,self.batchsize,1 do
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


function DataLoader:loadDatafromtable(indexstart)
    if #self.trainImageTable == 0 then
        error('trainSet doesn\'t contain any JPG files')
    end

    local imageRemain = #self.trainImageTable - indexstart + 1
    local sampleRealsize = self.size
    print(imageRemain)
    if imageRemain < self.size then
        sampleRealsize = imageRemain
    end
    local imageSet = torch.Tensor(sampleRealsize,unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(sampleRealsize,unpack(self.opt.outputSize))
    local mat = require 'matio'

    for i = 1,sampleRealsize,1 do
        local index = indexstart + i - 1
        local m1 = image.loadJPG(self.trainImageTable[index])
        local m2 = image.loadJPG(self.trainDepthTable[index])

        imageSet[i] = m1
        depthSet[i] = m2
        
    end
    local dataset = {
        image = imageSet,
        depth = depthSet,
        size =  function() return imageSet:size(1) end
    }

    setmetatable(dataset,
    {__index = function(t, i)
        return {t.image[i], t.depth[i]}
    end}
    )

    local input = self:miniBatchload(dataset)

    return input
end

return M.DataLoader

