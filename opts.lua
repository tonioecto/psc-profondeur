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
    cmd:option('-precision',       'single',        'Options: single | double | half')
    --------------- Training options --------------------
    cmd:option('-nEpochs',         10,              'Number of total epochs to run')
    cmd:option('-epochNumber',     1,               'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       10,              'mini-batch size (1 = pure stochastic)')
    cmd:option('-testOnly',        'false',         'Run on validation set only')
    cmd:option('-sampleSize',      1000,            'Number of datas to load to memory')
    cmd:option('-manualSeed',      2000,            'Manually set RNG seed')
    cmd:option('-nThreads',        10,              'Number of threads')
    cmd:option('-criterion',        'l2',          'loss function')
    ------------- Checkpointing options ---------------
    cmd:option('-resume',          'none',          'Resume from the latest checkpoint in this directory')
    --------- Optimization options ----------------------
    cmd:option('-LR',              0.005,           'initial learning rate')
    cmd:option('-momentum',        0.9,             'momentum')
    cmd:option('-weightDecay',     0.0003,            'weight decay')
    --------- Dataset options ----------------------
    cmd:option('-dataset',         'make3d',        'dataset to train, make3d or nyu')
    --------- Visulization options ----------------------
    cmd:option('-plot',            'false',          'plot online show option')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.testOnly = opt.testOnly ~= 'false'

    if opt.criterion ~= 'l2' then
        opt.criterion = 'hu'
    end

    opt.plot = opt.plot == 'true'

    if opt.precision == nil or opt.precision == 'single' then
        opt.tensorType = 'torch.CudaTensor'
    elseif opt.precision == 'double' then
        opt.tensorType = 'torch.CudaDoubleTensor'
    elseif opt.precision == 'half' then
        opt.tensorType = 'torch.CudaHalfTensor'
    else
        cmd:error('unknown precision: ' .. opt.precision)
    end

    if opt.dataset == 'make3d' then
        opt.LR = 0.05
    elseif opt.dataset == 'nyu' then
        opt.LR = 0.05
    else
        cmd:error('unknown dataset: '..opt.dataset)
    end

    -- Defaut dataset options
    opt.data = 'data'
    opt.format = 't7'
    opt.depthRotation = 'true'
    opt.depthOrigin = 'Train400Depth_t7'
    opt.imageOrigin = 'Train400Image'
    opt.testDepth = 'Test134Depth_t7'
    opt.testImage = 'Test134Image'
    opt.incre = 40

    -- Defaut val and train repartition
    opt.trainDataPortion = 0.8

    -- Default opt save and resume options
    opt.save = 'model_trained'
    opt.lossFile = 'loss_track'

    -- Default input and output size informations
    opt.inputSize = {3, 230, 173}
    opt.outputSize = {128, 96}

    opt.exampleNum = 100

    return opt
end

return M
