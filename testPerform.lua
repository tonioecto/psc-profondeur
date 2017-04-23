-- import packages
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local model = require '/models/init'
local Trainer = require 'train'
local checkpoints = require 'checkpoints'
local opts = require 'opts'
local datasetInit = require 'datasets/init'

-- Create options
-- define batch-size, data-set to load, learning rate, max iteration times
-- and resume flag
print '==> set up training options'
local opt = opts.parse(arg)

print '==> load dataset'
-- Data loading
datasetInit.init(opt, {'train', 'val'})
local info = datasetInit.getInfo(opt)
--local dataloader, valLoader = DataLoader.create(opt, info)
local testloader = DataLoader.createTest(opt)

-- Load previous checkpoint, if it exists
local checkpoint, optimState, normInfo = checkpoints.latest(opt)

if normInfo == nil then
    normInfo = dataloader:computeNormInfo()
    local dir = paths.concat(opt.save)
    if not paths.dirp(dir) then
        paths.mkdir(dir)
    end
    torch.save(paths.concat(dir, 'norm.t7'), normInfo)
end

--dataloader:loadNormInfo(normInfo)
--valLoader:loadNormInfo(normInfo)
testloader:loadNormInfo(normInfo)

-- Create model
print '==> create model'
local net, criterion = model.setup(opt, checkpoint)
-- print to verify the structure of the neural network created
-- print('ResNet and up-projection \n' .. net:__tostring())

print '==> configuring optimizer'
-- Create optimizer
-- if there is no stored optim state file,
-- set defaut parameters for optimized stochastic gradient descent method
if optimState == nil then
    optimState = {
        learningRate = opt.LR,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = 0,
        precision = opt.precision,
        nesterov = true,
        dampening = 0.0,
    }
end

-- create Trainer class
local trainer = Trainer(net, criterion, optimState, opt)
local perms = torch.randperm(testloader.dataset:size())
testloader:loadPerm(perms)

net:evaluate()
--[[for num = 1, opt.exampleNum, 1 do

    -- generate a new permutation table

    local pair = dataloader.dataset:get(dataloader.perms[num])

    -- get predicted results
    trainer:predict(num, pair, dataloader)
end]]

local res = trainer:getPredictResult(testloader,100)
local Evaluate = require 'evaluate.lua'
local rel = Evaluate.errEvaluate(res.pred,res.groundTruth)
