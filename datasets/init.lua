
local M = {}

function M.create(opt, split)
    print(opt.dataset)
    local Dataset = require('datasets/' .. opt.dataset)
    Dataset.preprocess(opt, split)
    Dataset.preprocess(opt, 'val')
    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cacheFile = paths.concat(opt.data, 'cache')
    local info = gen.exec(opt, cacheFile)
    return Dataset(imageInfo, opt, split)
end

return M
