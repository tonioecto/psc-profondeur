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

local M = { }

function M.parse(arg)

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 ResNet Up Pooling Script')
    cmd:text('See psc-profondeur descriptions for examples')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-precision', 'single',    'Options: single | double | half')
    --------------- Training options --------------------
    cmd:option('-nEpochs',         0,       'Number of total epochs to run')
    cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
    cmd:option('-testOnly',        'false', 'Run on validation set only')
    ------------- Checkpointing options ---------------
    cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
    --------- Optimization options ----------------------
    cmd:option('-LR',              0.1,   'initial learning rate')
    cmd:option('-momentum',        0.9,   'momentum')
    cmd:option('-weightDecay',     1e-4,  'weight decay')
    cmd:text()
    
    local opt = cmd:parse(arg or {})

    opt.testOnly = opt.testOnly ~= 'false'
    
    if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
        cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
    end
    
    local trainDir = paths.concat(opt.data, 'train')
    
    if not paths.dirp(opt.data) then
        cmd:error('error: missing ImageNet data directory')
    elseif not paths.dirp(trainDir) then
        cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
    end
    
    -- Default shortcutType=B and nEpochs=90

    opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
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
    
    if opt.resetClassifier then
        if opt.nClasses == 0 then
            cmd:error('-nClasses required when resetClassifier is set')
        end
    end
    
    if opt.shareGradInput and opt.optnet then
        cmd:error('error: cannot use both -shareGradInput and -optnet')
    end
    
    return opt
end

return M
