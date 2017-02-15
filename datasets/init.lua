
local M = {}

local function isValid(path)
    return paths.filep(path)
end

function M.__init(opt, set)
    
    local Dataset = require('datasets/' .. opt.dataset)
    
    if opt.resume == 'none' then
        for _,split in ipairs(set) do
            if not (
                isValid(paths.concat(opt.data, split, 'image')) and 
                isValid(paths.concat(opt.data, split, 'depth'))
                ) then
                Dataset.preprocess(opt, split)
            end
        end
    end
end

function M.create(opt, split)

    local gen = require('datasets/' .. opt.dataset .. '-gen')

    local cacheFile = paths.concat(opt.data, 'cache')
    local info = gen.exec(opt, cacheFile)
    return Dataset(info, opt, split)
end

function M.info(opt)

    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cacheFile = paths.concat('data', 'cache')

    return gen.exec(opt, cache)
end

return M
