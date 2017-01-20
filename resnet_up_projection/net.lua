require 'nn'
require 'torch'
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'nn'

--convBlock1
function convBlock1(net, d0, d1, d2)
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
function convBlock2(net, s, d0, d1, d2)
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

-- implement simple version of up-projection
function upProjection(net, d1, d2)
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

model_resnet = torch.load('resnet-50.t7')

net = nn.Sequential()

net:add(model_resnet)

d0 = 2048
d1 = 1024
net:add(nn.SpatialConvolution(d0, d1, 1, 1, 1, 1))
-- net:add(nn.BatchNormalization(d1))

--build up projection blocks
up_projection = nn.Sequential()
upProjection(up_projection, 1024, 512)
upProjection(up_projection, 512, 256)
upProjection(up_projection, 256, 128)
upProjection(up_projection, 128, 64)

net:add(up_projection)

net = net:cuda()

print('CRN net\n' .. net:__tostring())
