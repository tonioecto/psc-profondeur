local opts = require 'opts'
local convertor = require 'dataconvertor'

-- generate an opt file
local opt = opts.parse(arg)

-- convert trainDataDepth and testDataDepth
convertor.convertMatTensor(opt, opt.trainDepth)
convertor.convertMatTensor(opt, opt.testDepth)
