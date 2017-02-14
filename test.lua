-- import packages
local gen = require 'dataset/make3d-gen'
local opts = require 'opts'

local opt = opts.parse(arg)
gen.augmentation('Train400Image', 'Train400Depth_t7', opt)
