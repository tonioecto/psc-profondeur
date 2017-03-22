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
local dataloader, valLoader = DataLoader.create(opt, info)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

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
local perms = torch.randperm(dataloader.dataset:size())
dataloader:loadPerm(perms)

net:evaluate()
for num = 1, opt.exampleNum, 1 do

    -- generate a new permutation table

    local pair = dataloader.dataset:get(dataloader.perms[num])
    local img = pair.image:cuda()
    local depth = pair.depth:cuda()

    -- get predicted results
    trainer:predict(num, img, depth, dataloader)
end
