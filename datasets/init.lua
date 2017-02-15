
local M = {}

local function isValid(path)
    return paths.filep(path)
end

function M.create(opt, split)
    local Dataset = require('datasets/' .. opt.dataset)
    if not (
        isValid(paths.concat(opt.data, split, 'image')) and 
        isValid(paths.concat(opt.data, split, 'depth'))
        ) then
        Dataset.preprocess(opt, split)
    end

    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cacheFile = paths.concat(opt.data, 'cache')
    local info = gen.exec(opt, cacheFile)
    return Dataset(info, opt, split)
end

return M
