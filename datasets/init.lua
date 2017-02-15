
local M = {}

function M.create(opt, split)
    local Dataset = require('datasets/' .. opt.dataset)
    Dataset.preprocess()
    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cacheFile = paths.concat(opt.data, 'cache')
    local info = gen.exec(opt, cacheFile)
    return Dataset(imageInfo, opt, split)
end

return M
