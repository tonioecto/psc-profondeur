
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
    
    -- create path info file
    self.info = self.info(opt)
end

function M.create(opt, split)

    local Dataset = require('datasets/' .. opt.dataset)
    assert(self.info ~= nil, 'info is not yet initialized!')
    return Dataset(self.info, opt, split)

end

-- generate a path info file
function M.info(opt)

    local gen = require('datasets/' .. opt.dataset .. '-gen')
    local cache= paths.concat(opt.data, 'info-cache')

    self.info = {
        val = gen.exec(opt, cache..'-val.t7', 'val'),
        train = gen.exec(opt, cache'-tain.t7', 'train')
    }
    
    return self.info
end

return M
