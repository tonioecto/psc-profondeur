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
    inputSize = {3, 228, 304},
    outputSize = {128, 160},
    save = 'model_trained',
    sampleSize = 3,
    batchSize = 1,
    dataset  = 'imagenet',
    LR = 0.01,
    maxIteration = 1000,
    tensorType = 'cuda',
    resume = 'model_trained ',
    lossFile = 'loss_track',
    nesterov = true,
    dampening = 0.0
}

print '==> load dataset'
-- Data loading
local dataloader = DataLoader('imagetest', 'depth',opt)
dataloader:creDatatable()

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
print '==> create model'
local net, criterion = model.setup(opt)
-- verify the structure of the neural network created
-- print('ResNet and up-projection \n' .. net:__tostring())

print '==> configuring optimizer'
-- Create optimizer
-- set parameters for optimized stochastic gradient descent method
local optimState = {
    learningRate = opt.LR,
    weightDecay = 1e-3,
    momentum = 0.9,
    learningRateDecay = 0,
    precision = 'double',
    nesterov = true,
    dampening = 0.0
}

-- Create Trainer class
local trainer = Trainer(net, criterion, optimState, opt)

-- Start or resume training precedure
local num = 100
local bestValErr = math.huge
for epoch = 1, opt.maxIteration, 1 do
    dataloader:tableShuffle('train')
    trainer:train(epoch, dataloader)

    -- Run model on validation set
    model:evaluate()

    local valSet = dataloader:loadDataset('val')
    local valErr = trainer:computeScore(valSet)
    local trainErr = trainer:sampleTrainingLoss(2)

    trainer:showDepth('train',2)

    local bestModel = false
    if valErr < bestValErr then
        bestModel = true
        bestValErr = valErr
        print(' * Best model ', valErr)
    end

    trainer:saveLoss(epoch, valErr, trainErr)

    checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end
