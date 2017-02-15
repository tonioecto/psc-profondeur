local image = require 'image'
local paths = require 'paths'
local T = require 'datasets/transforms'
local G = require 'datasets/make3d-gen'
local ffi = require 'ffi'

local M = {}
local Make3dDataset = torch.class('resnetUnPooling.Make3dDataset', M)

function Make3dDataset:__init(info, opt, split)
    self.info = info[split]
    self.opt = opt
    self.split = split
    self.dir = paths.concat(opt.data, split)
    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function Make3dDataset:_loadImage(path)
end

function Make3dDataset:size()
    return #self.info.imagePath
end

function Make3dDataset.preprocess(opt)
    -- transformation combination
    local trans = T.Compose({
        T.RandomScale(1, 1.5),
        T.HorizontalFlip(0.5),
        T.Rotation(5),
        T.Color(0.8, 1,2),
        T.RandomCrop(173, 230, 96, 128)
    })
    -- from the original dataset, generate val and train set
    G.augmentation('Train400Image', 'Train400Depth_t7', opt,
    split, opt.trainDataPortion)
end

return M.Make3dDataset
