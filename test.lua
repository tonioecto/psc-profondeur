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

-- Create options
-- define batch-size, data-set to load, learning rate, max iteration times
-- and resume flag
print '==> set up training options'
local opt = opts.parse()

-- Create model
print '==> create model'
local net, criterion = model.setup(opt)
net:evaluate()
print(#net:forward(torch.rand(3, 304, 228):cuda()))
print(#net:forward(torch.rand(3, 173, 230):cuda()))
-- verify the structure of the neural network created
-- print('ResNet and up-projection \n' .. net:__tostring())

