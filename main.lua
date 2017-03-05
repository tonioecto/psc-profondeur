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
local opt = opts.parse()

print '==> load dataset'
-- Data loading
datasetInit.init(opt, {'train', 'val'})
local info = datasetInit.getInfo(opt)
local dataloader, valLoader = DataLoader.create(opt, info)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

print(checkpoint)

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

-- start or resume training precedure
local bestValErr = math.huge
for epoch = opt.epochNumber, opt.nEpochs+opt.epochNumber, 1 do

    -- generate a new permutation table
    local perms = torch.randperm(dataloader.dataset:size())
    dataloader:loadPerm(perms)
    trainer:train(epoch, dataloader)

    -- Run model on validation set
    net:evaluate()

    local valErr = trainer:computeValScore(valLoader, 10)
    local trainErr = trainer:sampleTrainingLoss(2)

    -- trainer:showDepth(dataloader, opt.example)

    local bestModel = false

    if valErr < bestValErr then
        bestModel = true
        bestValErr = valErr
        print(' * Best model ', valErr)
    end

    trainer:saveLoss(epoch, valErr, trainErr)

    -- save latest model
    checkpoints.saveCurrent(epoch, net, trainer.optimState, bestModel, opt)
    
end
