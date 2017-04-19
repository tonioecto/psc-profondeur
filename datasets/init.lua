
local M = {}

local function isValid(path)
    return paths.dirp(path)
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

function M.create(opt, split, info)

    local Dataset = require('datasets/' .. opt.dataset)
    assert(info ~= nil, 'info is not yet initialized!')
    return Dataset(info, opt, split)

end

-- generate a path info file
function M.getInfo(opt)

    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cache= paths.concat(opt.data, 'info-cache')

    local info = {
        val = gen.exec(opt, cache..'-val.t7', 'val'),
        train = gen.exec(opt, cache..'-train.t7', 'train')
    }
    
    return info
end

function M.getTestInfo(opt)
    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cache= paths.concat(opt.data, 'info-cache')

    local info = gen.exec(opt, cache..'-test.t7', 'test')
    
    return info
end

return M
