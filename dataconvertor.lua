local matio = require 'matio'
require 'paths'

local M = {}

-- convert .mat file to torch tensor format
function M.convertMatTensor(opt)
    local loadPath = paths.concat(opt.data, opt.trainData)
    local savePath = paths.concat(opt.data, opt.trainData..'_t7')

    -- create new directory to store generated files
    paths.mkdir(pathSave)

    local tmp
    local fileToLoad
    local fileToSave
    for file in paths.files() do
        if file:find(".*(mat)$") then
            fileToLoad = paths.concat(loadPath, file)
            fileToSave = file:match('(.*).mat$').."t7"
            print(fileToSave)
            tmp = matio.load(filePath)
            torch.save(fileToSave, tmp)
        end
    end
end

return M
