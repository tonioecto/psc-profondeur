require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'

local model = require '/models/model'
local upProjection = model.upProjection
local weightInit = require '/models/weight_init'

local M = {}

function M.setup(opt, checkpoint)

    if (checkpoint == nil) then
        return M.create()
    end

    -- load latest models
    local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
    print('=> Resuming model from ' .. modelPath)
    model = torch.load(modelPath)
    net = net:cuda()

    -- define criterion
    local criterion = nn.MSECriterion()
    criterion = criterion:cuda()
    return net, criterion

end

function M.create()

    -- input size 3x228x304
    -- First step: load modified pre-trained model resnet-50
    local resnet = torch.load('ResNet50.t7')

    -- verify resnet-50 structure
    -- print('Pre-trained ResNet 50 model\n' .. resnet:__tostring())
    local net = nn.Sequential()

    -- add resnet-50
    net:add(resnet)

    -- Second step: simple up-projection implementations
    -- build up projection blocks
    local up_projection = nn.Sequential()
    -- add several modules before up-projection blocks
    d0 = 2048
    d1 = 1024
    -- 1-convolution, input depth 2048, output depth 1024
    up_projection:add(cudnn.SpatialConvolution(d0, d1, 1, 1, 1, 1))
    -- input depth 1024, SpatialBatchNormalization
    up_projection:add(cudnn.SpatialBatchNormalization(1024))
    upProjection(up_projection, 1024, 512)
    upProjection(up_projection, 512, 256)
    upProjection(up_projection, 256, 128)
    upProjection(up_projection, 128, 64)

    -- add final modules
    local d_final = 64
    -- input depth 64, output depth 1, kernel 3X3
    up_projection:add(cudnn.SpatialConvolution(d_final, 1, 3, 3, 1, 1, 1, 1))
    up_projection:add(cudnn.ReLU())
    -- convert net to cuda model
    up_projection = up_projection:cuda()

    -- set up weights of un_projection
    weightInit.w_init(up_projection)

    net:add(up_projection)
    net = net:cuda()

    -- define criterion
    local criterion = nn.MSECriterion()
    criterion = criterion:cuda()

    return net, criterion
end

return M
