
local M = {}

local function isValid(path)
    return paths.filep(path)
end

function M.init(opt, set)

    local Dataset = require('datasets/' .. opt.dataset)
    
    -- if original dataset isn't yet preprocessed, we generate a new dataset using
    -- data augmentation method
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

    local Dataset = require('datasets/' .. opt.dataset)
    assert(self.info ~= nil, 'info is not yet initialized!')
    return Dataset(self.info, opt, split)

end

function M.info(opt)

    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cacheFile = paths.concat('data', 'cache')

    self.info = gen.exec(opt, cache)

    return self.info
end

return M
