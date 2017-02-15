-- import packages
local gen = require 'datasets/init'
local opts = require 'opts'

local opt = opts.parse(arg)
gen.create()
