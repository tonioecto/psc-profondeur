-- import packages
local gen = require 'dataset/make3d-gen'
local opts = require 'opts'

local opt = opts.parse(arg)
gen.augmentation(opt.imageOrigin, opt.depthOrigin, opt)
