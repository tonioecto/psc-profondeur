require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'
local gnu = require 'gnuplot'

local M = {}

function M.plotTrainLoss(opt)
    local lossFilePath = paths.concat(opt.lossFile, 'loss.t7')
    local loss = torch.load(lossFile)
    -- convert a table to tensor, and transpose it
    loss = torch.Tensor(loss)
    loss = loss:transpose(1, 2)
    -- plot trainErr and validationErr graph
    gnuplot.plot({'TrainErr', loss[2]},
    {'ValErr', loss[3]})
end

function M.Display(pred,real,preName,realName)
    gnu.pdffigure(preName)
    gnu.imagesc(pred,'color')
    gnuplot.plotflush()
    gnu.pdffigure(realName)
    gnu.imagesc(real,'color')
    gnuplot.plotflush()
end


function M.Relerror(predicted, groundtruth)
    print(predicted)
    print(groundtruth)
    local err = 0
    local Tsize = predicted:size(1)*predicted:size(2)
    Tsize = Tsize * predicted:size(3)
    for i =1,predicted:size(1) do
        local dis = torch.abs(
            predicted[i] - groundtruth[i]
            )
        dis:cdiv(groundtruth[i])
        print(dis)
        err = err + torch.sum(dis)
        print(err)
    end
    err = err / (Tsize * 1.0)
    return err
end

function M.Rmserror(predicted,groudtruth)
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

function M.Rmslogerr(predicted,groudtruth)
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

function M.Logerr(predicted,groundtruth)
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

function M.Thresherr(predicted,groundtruth,i)
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

function M.errTest(net,testSet)
    local predicted = torch.Tensor(testSet.depth:size())
    for i = 1,predicted:size(1) do
        predicted[i] = net:forward(testSet.image[i])
    end
    return Rmserror(predicted,testSet.depth)
end

return M
