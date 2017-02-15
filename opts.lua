--We import necessairy packages

require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'
require 'cudnn'
require 'cutorch'
--require 'cunn'

local M = { }

function M.parse(arg)

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 ResNet Up Pooling Script')
    cmd:text('See psc-profondeur descriptions for examples')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-precision',       'double',    'Options: single | double | half')
    --------------- Training options --------------------
    cmd:option('-nEpochs',         0,           'Number of total epochs to run')
    cmd:option('-epochNumber',     1,           'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       32,          'mini-batch size (1 = pure stochastic)')
    cmd:option('-testOnly',        'false',     'Run on validation set only')
    cmd:option('-sampleSize',      200,         'Number of datas to load to memory')
    ------------- Checkpointing options ---------------
    cmd:option('-resume',          'none',      'Resume from the latest checkpoint in this directory')
    --------- Optimization options ----------------------
    cmd:option('-LR',              0.1,         'initial learning rate')
    cmd:option('-momentum',        0.9,         'momentum')
    cmd:option('-weightDecay',     1e-4,        'weight decay')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.testOnly = opt.testOnly ~= 'false'

    -- Default nEpochs=90
    opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs

    if opt.precision == nil or opt.precision == 'single' then
        opt.tensorType = 'torch.CudaTensor'
    elseif opt.precision == 'double' then
        opt.tensorType = 'torch.CudaDoubleTensor'
    elseif opt.precision == 'half' then
        opt.tensorType = 'torch.CudaHalfTensor'
    else
        cmd:error('unknown precision: ' .. opt.precision)
    end

    -- Defaut dataset options
    opt.dataset = 'imagenet'
    opt.data = 'data'
    opt.format = 't7'
    opt.depthOrigin = 'Train400Depth_t7'
    opt.imageOrigin = 'Train400Image'
    opt.testDepth = 'Test134Depth_t7'
    opt.testImage = 'Test134Image_t7'
    opt.sizeAugmented = 16000

    -- Default opt save and resume options 
    opt.save = 'model_trained'
    opt.lossFile = 'loss_track'

    -- Default input and output size informations
    opt.inputSize = {3, 228, 304}
    opt.outputSize = {128, 160}

    return opt
end

return M
