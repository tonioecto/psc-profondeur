--We import necessairy packages

require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'

-- Defin simple up-convolution block
function upConvolution(net, d1, d2)
    net:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    net:add(nn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    net:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    net:add(nn.ReLU())
end

-- Define simple up-projection block
function upProjection(net, d1, d2)
    local cat = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch1:add(nn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch1:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    branch1:add(nn.ReLU())
    branch1:add(nn.SpatialConvolution(d2, d2, 3, 3, 1, 1, 1, 1))

    local branch2 = nn.Sequential()
    branch2:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch2:add(nn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch2:add(nn.SpatialZeroPadding(0, -1, 0, -1))

    cat:add(branch1)
    cat:add(branch2)
    net:add(cat)
    net:add(nn.CAddTable())

    net:add(nn.ReLU())
end


-- input size 304x228x3
-- First step: load resnet-50 pre-trained model
function init()
    local resnet = torch.load('ResNet50.t7')

    -- verify resnet-50 structure

    -- Second step: simple up-projection implementations
    local net = nn.Sequential()

    -- add resnet-50
    net:add(resnet)

    -- add several modules befor up-projection
    d0 = 2048
    d1 = 1024
    net:add(nn.SpatialConvolution(d0, d1, 1, 1, 1, 1))
    -- net:add(nn.BatchNormalization(1024))

    -- build up projection blocks
    up_projection = nn.Sequential()
    upProjection(up_projection, 1024, 512)
    upProjection(up_projection, 512, 256)
    upProjection(up_projection, 256, 128)
    upProjection(up_projection, 128, 64)

    net:add(up_projection)

    d_final = 64
    net:add(nn.SpatialConvolution(d_final, 1, 3, 3, 1, 1, 1, 1))
    net:add(nn.ReLU())
    net = net:cuda()

    return net
end
