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

function Make3dDataset:get(i)
    local element = {}
    local img, depth = self:__loadImageDepth(self.info.imagePath[i],
    self.info.depthPath[i])
    element.image = img
    element.depth = depth
    return element
end

function Make3dDataset:__loadImageDepth(img, depth)
    return image.loadJPG(img), torch.load(depth)
end

function Make3dDataset:size()
    return #self.info.imagePath
end

function Make3dDataset.preprocess(opt, split)
    -- transformation combination
    local trans = T.Compose({
        T.RandomScale(1, 1.5),
        T.HorizontalFlip(0.5),
        T.Rotation(5),
        T.Color(0.8, 1,2),
        T.RandomCrop(173, 230, 96, 128)
    })
    -- from the original dataset, generate val and train set
    G.augmentation(opt.imageOrigin, opt.depthOrigin, opt,
    split, opt.trainDataPortion, trans)
end

function Make3dDataset.preprocessOnline()
    local trans = T.Compose({
        T.RandomScale(1, 1.5),
        T.HorizontalFlip(0.5),
        T.Rotation(5),
        T.Color(0.8, 1,2),
        T.RandomCrop(173, 230, 96, 128)
    })

    return G.augOneMatch(trans)
end

function Make3dDataset.info(opt, cache)
    return G.exec(opt, cache)
end

return M.Make3dDataset
