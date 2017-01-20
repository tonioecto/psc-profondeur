
-- import packages 
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local model = require 'model'
local Trainer = require 'train'


-- Create model
local model, criterion = models.init()

-- Creat criterion function
local criterion = nn.MSECriterion()    --we use the mean square error
criterion = criterion:cuda()

-- Data loading
local trainSet = DataLoader.loadDataset('minibatch','depthTest')
print('trainSet size:'..trainSet:size())

-- Train

Trainer.train(net, criterion, batchLoader, trainSet)