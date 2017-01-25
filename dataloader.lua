require 'image'
require 'paths'
require 'os'
require 'math'
--require 'matio'
require 'xlua'

local unpack = unpack or table.unpack
local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function M.load()
    local trainSet
    local validationSet
    return trainSet, validationSet
end

function DataLoader:__init(imageset, depthset,opt)
   self.imageset =imageset
   self.depthset = depthset
   --self.indexstart = indexstart
   self.size = opt.samplesize
   self.batchsize = opt.batchSize
end

function DataLoader:loadDataset()       --Load the images and depthMap, and generate dataset for trainning

    print('The self.sizeber of image is:'..#self.imagename)
    print('The self.sizeber of correponding depthmap is:'..#self.depthname)

    if #self.imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    local imageSet = torch.Tensor(#self.imagename,3,228,304)
    local depthSet = torch.Tensor(#self.depthname,160,128)
    local mat = require 'matio'

    for i,file in ipairs(self.imagename) do
        local m = image.loadJPG(file)
        m = image.scale(m,304,228,'bicubic')
        imageSet[i] = m
    end

    for i,file in ipairs(self.depthname) do
        local m = mat.load(file,'depthMap')
        m = image.scale(m,128,160,'bicubic')

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
    local shuffle = torch.randperm(dataset:size())
    local imageSize = dataset.image[1]:size()
    local depthSize = dataset.depth[1]:size()

    local self.sizeBatch = math.floor(dataset:size()/self.batchsize)
    --print(self.sizeBatch)
    local imageSet = torch.Tensor(self.sizeBatch,self.batchsize,unpack(imageSize:totable()))
    local depthSet = torch.Tensor(self.sizeBatch,self.batchsize,unpack(depthSize:totable()))

    for index = 1,self.sizeBatch,1
    do
        local self.sizeRemain = dataset:size() - (index-1)*self.batchsize
        if(self.sizeRemain >=self.batchsize)then
            local indexBegin = (index-1)*self.batchsize + 1
            local indexEnd = indexBegin + self.batchsize - 1
            for k = 1,self.batchsize,1 do
                imageSet[{index,k,{}}] = dataset.image[shuffle[indexBegin]]
                depthSet[{index,k,{}}] = dataset.depth[shuffle[indexBegin]]
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

function DataLoader:creDatatable()
    self.imagename = {}
    self.depthname = {}

    for file in paths.files(self.imageset) do
        if file:find(".*(jpg)$") then
            table.insert(self.imagename, paths.concat(self.imageset,file))
        end
    end

    if #self.imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    table.sort(self.imagename, function (a,b) return a < b end)
    print(#self.imagename)

    local indexFakefile = {}
    for i,file in ipairs(self.imagename) do
        local name = file:match(".+/img(.*).jpg$")
        local index = 'depth'..name
        local fullname = index..'.mat'
        local matname = paths.concat(self.depthset,fullname)
        if paths.filep(matname)then
            table.insert(self.depthname,matname)
        else
            table.insert(indexFakefile,i)
        end
    end
    for i=#indexFakefile,1,-1 do
        table.remove(self.imagename,indexFakefile[i])
    end

    print('The self.sizeber of image is:'..#self.imagename)
    print('The self.sizeber of correponding depthmap is:'..#self.depthname)

    if #self.imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    return self.imagename,self.depthname
end

function DataLoader:tableShuffle()
    local shuffle = torch.randperm(#self.imagename)
    local image = {}
    local depth = {}
    for index = 1,#self.imagename,1 do
        image[index] = self.imagename[shuffle[index]]
        depth[index] = self.depthname[shuffle[index]]
    end
    return image,depth
end

function DataLoader:loadDatafromtable(indexstart)
    if #self.imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    local imageRemain = #self.imagename - indexstart + 1
    local sampleRealsize = self.size
    if imageRemain < self.size then
        sampleRealsize = imageRemain
    end
    local imageSet = torch.Tensor(sampleRealsize,3,228,304)
    local depthSet = torch.Tensor(sampleRealsize,160,128)
    local mat = require 'matio'

    for i = 1,sampleRealsize,1 do
        local index = indexstart + i - 1
        local m1 = image.loadJPG(self.imagename[index])
        local m2 = mat.load(self.depthname[index],'depthMap')
        m1 = image.scale(m1,304,228,'bicubic')
        m2 = image.scale(m2,128,160,'bicubic')
        imageSet[i] = m1
        depthSet[i] = m2
        --indexstart = indexstart + 1
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