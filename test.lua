require 'cudnn'
require 'nn'
require 'cunn'
require 'cutorch'
local model = require '/models/init/'

local net, criterion = model.create()

local input = torch.Tensor(3, 3, 173, 230)
input = input:cuda()
local output = net:forward(input)
print(#output)
