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
   self.size = opt.sampleSize
   self.batchsize = opt.batchsize
   self.opt = opt
end



function DataLoader:creDatatable()
    self.imagename = {}
    self.depthname = {}

    self.trainImageTable = {}
    self.trainDepthTable = {}

    self.valImageTable = {}
    self.valDepthTable = {}

    self.testImageTable = {}
    self.testDepthTable = {}

    for file in paths.files(self.imageset) do
        if file:find(".*(jpeg)$") then
            table.insert(self.imagename, paths.concat(self.imageset,file))
        end
    end

    if #self.imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    table.sort(self.imagename, function (a,b) return a < b end)
    --print(#self.imagename)

    local indexFakefile = {}
    for i,file in ipairs(self.imagename) do
        --local name = file:match(".+/img(.*).jpg$")
        local name = file:match(".+/(.*)$")
        --local index = 'depth'..name
        --local fullname = index..'.mat'
        local matname = paths.concat(self.depthset,name)
        if paths.filep(matname)then
            table.insert(self.depthname,matname)
        else
            table.insert(indexFakefile,i)
        end
    end
    for i=#indexFakefile,1,-1 do
        table.remove(self.imagename,indexFakefile[i])
    end

    print('The number of image is:'..#self.imagename)
    print('The number of correponding depthmap is:'..#self.depthname)

    if #self.imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    --create table of trainset,validationset and testset
    self:tableShuffle("all")    --shuffle the table firstly

    local dataSetSize = #self.imagename

    local trainSetSize = math.ceil(0.8 * dataSetSize)
    local valiSetSize = math.ceil(0.9 * dataSetSize)

    for i=1,trainSetSize,1 do
      table.insert(self.trainImageTable,self.imagename[i])
      table.insert(self.trainDepthTable,self.depthname[i])
    end

    for i = trainSetSize+1,valiSetSize,1 do
      table.insert(self.valImageTable,self.imagename[i])
      table.insert(self.valDepthTable,self.depthname[i])
    end

    for i = valiSetSize+1,dataSetSize,1 do
      table.insert(self.testImageTable,self.imagename[i])
      table.insert(self.testDepthTable,self.depthname[i])
    end
    --return self.imagename,self.depthname
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

    local imageSet = torch.Tensor(#self.imagename,unpack(self.opt.inputSize))
    local depthSet = torch.Tensor(#self.depthname,unpack(self.opt.outputSize))
    local mat = require 'matio'

    for i,file in ipairs(self.imagename) do
        local m = image.loadJPG(file)
        --m = image.scale(m,304,228,'bicubic')
        imageSet[i] = m
    end

    for i,file in ipairs(self.depthname) do
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

    for index = 1,numBatch,1
    do
        local numRemain = dataset:size() - (index-1)*self.batchsize
        if(numRemain >=self.batchsize)then
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



function DataLoader:tableShuffle(s)
  local rand = math.random
  if s=='all' then
    local iteration = #self.imagename
    local j

    for i=iteration,2,-1 do
      j = rand(i)
      self.imagename[i],self.imagename[j] = self.imagename[j],self.imagename[i]
      self.depthname[i],self.depthname[j] = self.depthname[j],self.depthname[i]
    end
  elseif s=='train' then
    local iteration = #self.trainImageTable
    local j

    for i=iteration,2,-1 do
      j = rand(i)
      self.trainImageTable[i],self.trainImageTable[j] = self.trainImageTable[j],self.trainImageTable[i]
      self.trainDepthTable[i],self.trainDepthTable[j] = self.trainDepthTable[j],self.trainDepthTable[i]
    end

  end
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
        --local m2 = mat.load(self.trainDepthTable[index],'depthMap')
        --m1 = image.scale(m1,304,228,'bicubic')
        --m2 = image.scale(m2,128,160,'bicubic')

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
