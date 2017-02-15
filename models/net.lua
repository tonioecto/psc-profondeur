require 'nn'
require 'torch'
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'nn'

local M = {}

--convBlock1
function M.convBlock1(net, d0, d1, d2)
    local cat = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(nn.SpatialConvolution(d0, d1, 1, 1))
    -- branch1:add(nn.SpatialBatchNormalization(d1))
    branch1:add(nn.ReLU())
    branch1:add(nn.SpatialConvolution(d1, d1, 3, 3, 1, 1, 1, 1))
    -- branch1:add(nn.SpatialBatchNormalization(d1))
    branch1:add(nn.ReLU())
    branch1:add(nn.SpatialConvolution(d1, d2, 1, 1))
    -- branch1:add(SpatialBatchNormalization(d2))
    local branch2 = nn.Sequential()

    branch2:add(nn.Identity())

    cat:add(branch1)
    cat:add(branch2)

    net:add(cat)
    net:add(nn.CAddTable())
    net:add(nn.ReLU)
end

--conBlock2
function M.convBlock2(net, s, d0, d1, d2)
    local cat = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(nn.SpatialConvolution(d0, d1, 1, 1, s, s))
    --branch1:add(nn.SpatialBatchNormalization(d1))
    branch1:add(nn.ReLU())
    branch1:add(nn.SpatialConvolution(d1, d1, 3, 3, 1, 1, 1, 1))
    -- branch1:add(nn.SpatialBatchNormalization(d1))
    branch1:add(nn.ReLU())
    branch1:add(nn.SpatialConvolution(d1, d2, 1, 1, 1, 1))
    -- branch1:add(nn.SpatialBatchNormalization(d2))

    local branch2 = nn.Sequential()
    branch2:add(nn.SpatialConvolution(d0, d2, 1, 1, s, s))
    -- branch2:add(nn.SpatialBatchNormalization(d2))

    cat:add(branch1)
    cat:add(branch2)

    net:add(cat)
    net:add(nn.CAddTable())
    net:add(nn.ReLU())
end

-- implement simple version of up-convolution
function M.upConvolution(net, d1, d2)
    net:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    net:add(cudnn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    net:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    net:add(cudnn.ReLU())
end

-- implement simple version of up-projection
function M.upProjection(net, d1, d2)
    local cat = nn.ConcatTable()

    local branch1 = nn.Sequential()

    branch1:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch1:add(cudnn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch1:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    branch1:add(cudnn.ReLU())
    branch1:add(cudnn.SpatialConvolution(d2, d2, 3, 3, 1, 1, 1, 1))

    local branch2 = nn.Sequential()
    branch2:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch2:add(cudnn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch2:add(nn.SpatialZeroPadding(0, -1, 0, -1))

    cat:add(branch1)
    cat:add(branch2)
    net:add(cat)
    net:add(nn.CAddTable())

    net:add(cudnn.ReLU())
end
