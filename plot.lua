require 'gnuplot'
--require 'opts'

local M = {}

function M.plotTrainLoss(lossFile, startEpoch, endEpoch)
    --local opt = opts.parse(arg)
    --local lossFilePath = paths.concat(opt.lossFile, lossFile)
    local lossFilePath = paths.concat('loss_track', lossFile)

    assert(paths.filep(lossFilePath), 'loss file does not exist')

    local loss = torch.load(lossFilePath)
    local lossTrace = nil

    if startEpoch == nil then
        startEpoch = 1
    end

    if endEpoch == nil then
        endEpoch = #loss
    end
    
    for epoch = startEpoch, endEpoch, 1 do
        temp = loss[epoch].lossTrace:reshape(20, 32)
        if lossTrace == nil then
            lossTrace = temp[1]
        else
            lossTrace = torch.cat(lossTrace, temp[1])
        end
    end

    print(('==> print figure of training loss evolution from epoch %d to %d'):format(startEpoch, endEpoch))
    --gnuplot.epsfigure(paths.concat(opt.lossFile, 'training-loss-'..startEpoch..'-'..endEpoch..'.eps'))
    gnuplot.pdffigure(paths.concat('loss_track', 'training-loss-'..startEpoch..'-'..endEpoch..'.pdf'))
    gnuplot.xlabel('batch (every 40 mini-batches)')
    gnuplot.ylabel('loss')
    gnuplot.title('Training mini-batch loss')
    gnuplot.plot(lossTrace, '-')
    gnuplot.plotflush()

end

return M
