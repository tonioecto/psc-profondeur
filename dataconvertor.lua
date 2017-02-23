local matio = require 'matio'
require 'paths'

local M = {}

-- convert .mat file to torch tensor format
--- depth is stored as Position3DGrid enty in a table
function M.convertMatTensor(opt, path)
    local loadPath = paths.concat(opt.data, path)
    local savePath = paths.concat(opt.data, path..'_t7')    -- create new directory to store generated files

    paths.mkdir(savePath)

    local tmp
    local fileToLoad
    local fileToSave
    for file in paths.files(loadPath) do
        if file:find(".*(mat)$") then
            fileToLoad = paths.concat(loadPath, file)
            fileToSave = paths.concat(savePath, file:match('(.*).mat$')..".t7")
            tmp = matio.load(fileToLoad,opt.depthName)
            if opt.depthName=='Position3DGrid' then
              tmp = tmp:select(3,4)
            end
            if opt.depthRotation then
              tmp = image.hflip(tmp)
              tmp = tmp:transpose(1,2)
            end
            torch.save(fileToSave,tmp)
        end
    end
end

return M
