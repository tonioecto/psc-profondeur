--We import necessairy packages

require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'

----------------------------------------------
--1,Load dataset and prepare our traindata and testdata
----------------------------------------------
function loadImage(imageSet)
    local imagename = {}
    for file in paths.files(imageSet) do
        if file:find(".*(jpg)$") then
            table.insert(imagename, paths.concat(imageSet,file))
        end
    end
    
    print(#imagename)
    print(imagename[1])
    local imagest = torch.Tensor(#imagename,3,228,304)
    for i,file in ipairs(imagename) do
        local m = image.load(file)
        m = image.scale(m,304,228,'bicubic')
        imagest[i] = m
    end
    
    return imagest
end

function loadDataset(imageSet,depthSet)       --Load the images and depthMap, and generate dataset for trainning
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
        --local m = image.loadJPG(file)
        --print(file)
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
    --randomize the data firstly
    local shuffle = torch.randperm(dataset:size())
    local imageSize = dataset.image[1]:size()
    local depthSize = dataset.depth[1]:size()

    --local numBatch = math.floor(dataset:size()/batchSize)
    local numBatch = 2
    --print(numBatch)
    local imageSet = torch.Tensor(numBatch,batchSize,unpack(imageSize:totable()))
    local depthSet = torch.Tensor(numBatch,batchSize,unpack(depthSize:totable()))

    --local index = 0
    for index = 1,numBatch,1
    do
        local numRemain = dataset:size() - (index-1)*batchSize
        if(numRemain >=batchSize)then
            --local imageBatch = torch.Tensor(batchSize,unpack(imageSize:totable()))
            --local depthBatch = torch.Tensor(batchSize,unpack(depthSize:totable()))
            local indexBegin = (index-1)*batchSize + 1
            local indexEnd = indexBegin + batchSize - 1
            for k = 1,batchSize,1 do
                imageSet[{index,k,{}}] = dataset.image[shuffle[indexBegin]]
                depthSet[{index,k,{}}] = dataset.depth[shuffle[indexBegin]]
                indexBegin = indexBegin + 1
            end
                --imageSet[index] = dataset.image[{{indexBegin,indexEnd},{}}]
                --depthSet[index] = dataset.depth[{{indexBegin,indexEnd},{}}]
        --[[
        else
            local indexBegin = (index-1)*batchSize + 1
            local indexEnd = indexBegin + numRemain - 1
            imageSet[index] = dataset.image[{{indexBegin,indexEnd},{}}]
            depthSet[index] = dataset.depth[{{indexBegin,indexEnd},{}}]
            ]]
        end
        --index = index + 1
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


--Download the datasets
--[[
if (not paths.filep("Train400Img.tar.gz")) then
    os.execute('wget -c http://cs.stanford.edu/group/reconstruction3d/Train400Img.tar.gz')
end
if (not paths.dirp("Train400Img")) then
    os.execute('tar zxvf Train400Img.tar.gz')
end


if (not paths.filep("Train400Depth.tgz")) then
    os.execute('wget -c http://cs.stanford.edu/group/reconstruction3d/Train400Depth.tgz')
end
if (not paths.dirp("Train400Depth")) then
    os.execute('tar zxvf Train400Depth.tar.gz')
end


if (not paths.filep("Test134.tar.gz")) then
    os.execute('wget -c http://www.cs.cornell.edu/~asaxena/learningdepth/Test134.tar.gz')
end
if (not paths.dirp("Test134Img")) then
    os.execute('tar zxvf Test134.tar.gz')
end


if (not paths.filep("Test134Depth.tar.gz")) then
    os.execute('wget -c http://www.cs.cornell.edu/~asaxena/learningdepth/Test134Depth.tar.gz')
end
if (not paths.dirp("Test134Depth")) then
    os.execute('tar zxvf Test134Depth.tar.gz')
end
]]


--testSet = loadDataset('imageTest','depthTest')
--print('testSet size:'..testSet:size())

--imageset = loadImage('imageTrain')


-------------------------------------------------------
-- 2,Define our neural network
-------------------------------------------------------

--Up-projection
-- implement simple version of up-convolution
function upConvolution(net, d1, d2)
    net:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    net:add(nn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    net:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    net:add(nn.ReLU())
end

-- implement simple version of up-projection
function upProjection(net, d1, d2)
    local cat = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch1:add(nn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch1:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    branch1:add(nn.ReLU())
    branch1:add(nn.SpatialConvolution(d2, d2, 3, 3, 1, 1, 1, 1))

    local branch2 = nn.Sequential()
    branch2:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch2:add(nn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch2:add(nn.SpatialZeroPadding(0, -1, 0, -1))

    cat:add(branch1)
    cat:add(branch2)
    net:add(cat)
    net:add(nn.CAddTable())

    net:add(nn.ReLU())
end

-- input size 304x228x3
-- first part pre-trained by Facebook
-- https://github.com/facebook/fb.resnet.torch
resnet = torch.load('ResNet50.t7')

-- verify resnet-50 structure

-- Second step : different kinds of up-projection implementations
net = nn.Sequential()
--net:add(nn.View(-1):setNumInputDims(10))
net:add(resnet)
--print('CRN net\n' .. resnet:__tostring())


d0 = 2048
d1 = 1024
net:add(nn.SpatialConvolution(d0, d1, 1, 1, 1, 1))
--net:add(nn.BatchNormalization(1024))

--build up projection blocks
up_projection = nn.Sequential()
upProjection(up_projection, 1024, 512)
upProjection(up_projection, 512, 256)
upProjection(up_projection, 256, 128)
upProjection(up_projection, 128, 64)

net:add(up_projection)

d_final = 64
net:add(nn.SpatialConvolution(d_final, 1, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:evaluate()
net = net:cuda()

-------------------------------------------------------
--3,Define the loss function
-------------------------------------------------------
criterion = nn.MSECriterion()    --we use the mean square error
criterion = criterion:cuda()


--trainSet = trainSet:cuda()
--imageset = imageset:cuda()
--errTest(net,testSet)
--[[
imageoutput = resnet:forward(trainSet.image)
traindata = {
        image = imageoutput,
        depth = trainSet.depth,
        size =  function() return imageoutput:size(1) end
    }

setmetatable(traindata,
    {__index = function(t, i)
                return {t.image[i], t.depth[i]}
               end})
]]

-------------------------------------------------------
--4,Train the neural network
-------------------------------------------------------
trainSet = loadDataset('minibatch','depthTest')
print('trainSet size:'..trainSet:size())


trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 1 -- just do 5 epochs of training.

--trainSet.image = trainSet.image:cuda()
--trainSet.depth = trainSet.depth:cuda()
--trainset = trainset:cuda()
for i=1,20,1 do
    local input = miniBatchload(trainSet,5)  
    --input = input:cuda()
    input.image = input.image:cuda()
    input.depth = input.depth:cuda()
    trainer:train(input)
end

--torch.save('upprojectionnet.t7',net)
--trainpredi = net:forward(traindSet.image[1])
--
--traindata_rel = traindata.depth[1]:double()
--trainpredi = trainpredi:double()

--torch.save('depth_pre_test.t7',traindata_rel)
--torch.save('depth_real_test.t7',trainpredi)

------------------------------------------------------
--5,Test the network, caculate accuracy
-----------------------------------------------------

function Relerror(predicted, groundtruth)
    local err = 0
    local Tsize = predicted:size(1)*predicted:size(2)
    Tsize:mul(predicted:size(3))
    for i =1,predicted:size(1) do
        local dis = torch.abs(predicted(i)-groundtruth(i))
        dis:cdiv(groundtruth(i))
        err:add(torch.sum(dis))
    end
    err:div(Tsize)
    return err
end

function Rmserror(predicted,groudtruth)
    local err = 0
    local Tsize = predicted:size(1)*predicted:size(2)
    Tsize:mul(predicted:size(3))
    --local size = predicted:size(1)*predicted:size(2)
    --size:mul(predicted:size(3))
    for i=1,predicted:size(1) do
        local dis = torch.dist(predisted(i),groundtruth(i))
        dis:mul(dis)
        err:add(dis)
    end
    err:div(Tsize)
    err = math.sqrt(err)
    return err
end

function Rmslogerr(predicted,groudtruth)
    local err = 0
    local Tsize = predicted:size(1)*predicted:size(2)
    Tsize:mul(predicted:size(3))
    local term = math.log(10)
    for i=1,predicted:size(1) do
        local pre = torch.log(predicted(i))
        pre:div(term)
        local truth = torch.log(groudtruth(i))
        truth:div(term)
        local dis = torch.dist(pre,truth)
        dis:mul(dis)
        err:add(dis)
    end
    err:div(Tsize)
    err = math.sqrt(err)
    return err
end

function Logerr(predicted,groundtruth)
    local err = 0
    local Tsize = predicted:size(1)*predicted:size(2)
    Tsize:mul(predicted:size(3))
    local term = math.log(10)
    for i=1,predicted:size(1) do
        local pre = torch.log(predicted(i))
        pre:div(term)
        local truth = torch.log(groudtruth(i))
        truth:div(term)
        local dis = torch.abs(pre-truth)
        err = err + torch.sum(dis)
    end
    err:div(Tsize)
    return err
end

function Thresherr(predicted,groundtruth,i)
    local Thresh = math.pow(1.25,i)
    local Tsize = predicted:size(1)*predicted:size(2)
    Tsize:mul(predicted:size(3))
    local err = 0
    for i=1,predicted:size(1) do
        local a = torch.cdiv(predicted(i),groundtruth(i))
        local b = torch.cdiv(groundtruth(i),predicted(i))
        local c = torch.div(torch.abs(a-b),2)
        c:add((a+b)/2)
        c = Thresh - c
        c:sign()
        c:add(1)
        err = err + torch.sum(c)/2
    end
    err = err/Tsize
    return err*100
end

function errTest(net,testSet)
    local predicted = torch.Tensor(testSet.depth:size())
    for i = 1,predicted:size(1) do
        predicted[i] = net:forward(testSet.image[i])
    end
    return Rmserror(predicted,testSet.depth)
end

--img = image.load("img-combined1-p-139t0.jpg")
--img:cuda()
--img = image.loadJPG("img-combined1-p-15t0.extraperson
--torch.typename(trainSet)


--print('The error caculated for the testSet is:'..err)
--output=net:forward(img)
