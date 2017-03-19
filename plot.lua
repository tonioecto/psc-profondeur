require 'gnuplot'
require 'opts'

local M = {}

function M.plotTrainLoss(lossFile)
    local opt = opts.parse(arg)
    local lossFilePath = paths.concat(opt.lossFile, lossFile)

    assert(paths.filep(lossFilePath), 'loss file does not exist')

    local loss = torch.load(lossFilePath)
    local lossTrace = torch.Tensor()
    
    for k, v in ipairs(loss) do
        lossTrace = torch.cat(lossTrace, v.lossTrace)
    end

    gnuplot.epsfigure(paths.concat(opt.lossFile, 'training-loss.eps'))
    gnuplot.xlabel('epoch-batch')
    gnuplot.ylabel('batch-loss')
    gnuplot.plot(lossTrace, '-')
    gnuplot.plotflush()

end

return M
