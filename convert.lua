local convertor = require 'dataconvertor'
local opts = require 'opts'

local opt = opts.parse(arg)
print(opt.testDepth)
convertor.convertMatTensor(opt, opt.testDepth)
--convertor.convertMatTensor(opt, opt.depthOrigin)

return M
