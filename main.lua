-- import packages
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local model = require '/models/init'
local Trainer = require 'train'
local checkpoints = require 'checkpoints'

-- Creat options
-- define batch-size, data-set to load, learning rate, max iteration times
-- and resume flag
print '==> set up training options'
local opt = {
    save = 'model_trained',
    sampleSize = 200,
    batchSize = 10,
    dataset  = 'imagenet',
    LR = 0.01,
    maxIteration = 100,
    tensorType = 'cuda',
    resume = 'none',
    lossFile = 'loss_track'
}

print '==> load dataset'
-- Data loading
local dataloader = DataLoader("","",opt)
dataloader:creDatatable()

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
print '==> create model'
local net, criterion = model.setup(opt)
-- verify the structure of the neural network created
print('ResNet and up-projection \n' .. net:__tostring())

print '==> configuring optimizer'
-- Create optimizer
-- set parameters for optimized stochastic gradient descent method
local optimState = {
    learningRate = opt.LR,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 0,
    precision = 0.1
}

-- Create Trainer class
local trainer = Trainer(net, criterion, optimState, opt)

-- Start or resume training precedure
besValErr = math.huge
for epoch = 1, opt.maxIteration, 1 do
    dataloader:tableShuffle('train')
    trainer:train(epoch, dataloader)

    -- get loss for a ransom sample
    --trainer:saveLoss()

    --[[
    -- Run model on validation set
    local valErr= trainer:test(epoch, valLoader)
    
    local bestModel = false
    if valErr < bestValErr then
        bestModel = true
        bestValErr = valErr
        print(' * Best model ', valErr)
    end
    
    checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
    --]]
end
