-- import packages
local gen = require 'datasets/init'
local opts = require 'opts'
require 'nn'
require 'models/modules/MaskMSECriterion'

local opt = opts.parse(arg)
input = torch.rand(2, 2, 2):cuda()
target = torch.rand(2, 2, 2):cuda()
print(input)
print(target)
local criterion = nn.MaskMSECriterion(0.5, 0, true):cuda()
print(criterion:forward(input, target))
