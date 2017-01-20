-- We import necessairy packages

require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'


-- Load dataset and prepare our traindata and testdata
function loadImage(imageSet)
    local imagename = {}
    for file in paths.files(imageSet) do
        if file:find(".*(jpg)$") then
            table.insert(imagename, paths.concat(imageSet,file))
        end
    end
    
    local imagest = torch.Tensor(#imagename,3,228,304)
    for i,file in ipairs(imagename) do
        local m = image.load(file)
        m = image.scale(m,304,228,'bicubic')
        imagest[i] = m
    end
    
    return imagest
end

function loadDataset(imageSet,depthSet)       --Load the images and depthMap, and generate dataset for training
    local imagename = {}
    local depthname = {}

    for file in paths.files(imageSet) do
        if file:find(".*(jpg)$") then
            table.insert(imagename, paths.concat(imageSet,file))
        end
    end

    if #imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    table.sort(imagename, function (a,b) return a < b end)
    print(#imagename)

    local indexFakefile = {}
    for i,file in ipairs(imagename) do
        print(file)
        local name = file:match('.+/.+img(.*).jpg$')
        print(name)
        local index = 'depth'..name
        local fullname = index..'.mat'
        local matname = paths.concat(depthSet,fullname)
        if paths.filep(matname)then
            table.insert(depthname,matname)
        else
            --table.remove(imagename,i)
            --i = i -1
            table.insert(indexFakefile,i)
        end
    end
    for i=#indexFakefile,1,-1 do
        table.remove(imagename,indexFakefile[i])
    end

    print('The number of image is:'..#imagename)
    print('The number of correponding depthmap is:'..#depthname)

    if #imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    local imageSet = torch.Tensor(#imagename,3,228,304)
    --local depthSet = torch.Tensor(#depthname,160,128)
    local depthSet = torch.Tensor(#depthname,128,160)
    local mat = require 'matio'

    for i,file in ipairs(imagename) do
        local m = image.load(file)
        m = image.scale(m,304,228,'bicubic')
        imageSet[i] = m
    end

    for i,file in ipairs(depthname) do
        local m = mat.load(file,'depthMap')
        m = image.scale(m,128,160,'bicubic')
        
        m = image.hflip(m)
        m = m:transpose(1,2)
        
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

function miniBatchload(dataset, batchSize)
    local shuffle = torch.randperm(dataset:size())
    local imageSize = dataset.image[1]:size()
    local depthSize = dataset.depth[1]:size()

    local numBatch = 2
    local imageSet = torch.Tensor(numBatch,batchSize,unpack(imageSize:totable()))
    local depthSet = torch.Tensor(numBatch,batchSize,unpack(depthSize:totable()))

    for index = 1,numBatch,1
    do
        local numRemain = dataset:size() - (index-1)*batchSize
        if(numRemain >=batchSize)then
            local indexBegin = (index-1)*batchSize + 1
            local indexEnd = indexBegin + batchSize - 1
            for k = 1,batchSize,1 do
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
