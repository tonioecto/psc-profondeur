require 'paths'
require 'ffi'
local T = require 'datasets/transforms'
local unpack = unpack or table.unpack

local M = {}

local function findImageDepthMatches(imageDir, depthDir)
    
    local dirsImage = paths.dir(imageDir)
    table.sort(dirsImage)
    local imagePath = {}
    local depthPath = {}

    -- Generate a list of all the images and their depths
    for _,img in ipairs(dirsImage) do
        if not (img:find('.jpg') == nil) then
            local basename = img:match('img(.*).jpg$')
            local depth = 'depth_sph_corr'..basename..'.t7'
            if paths.filep(paths.concat(depthDir, depth)) then
                table.insert(imagePath, paths.concat(imageDir, img))
                table.insert(depthPath, paths.concat(depthDir, depth))
            end
        end
    end

    print('=> Generate image path file and depth path file with size '
    ..#imagePath..' and '..#depthPath)

    return imagePath, depthPath
end

function M.exec(opt, cacheFile)
    -- find the image and depth matches
    local trainImageDir = paths.concat(opt.data, 'train', 'image')
    local trainDepthDir = paths.concat(opt.data, 'train', 'depth')
    local valImageDir = paths.concat(opt.data, 'val', 'image')
    local valDepthDir = paths.concat(opt.data, 'val', 'depth')

    assert(trainImageDir, 'train image directory not found: ' .. trainImageDir)
    assert(trainDepthDir, 'train depth directory not found: ' .. trainDepthDir)

    assert(valImageDir, 'val image directory not found: ' .. valImageDir)
    assert(valDepthDir, 'val depth directory not found: ' .. valDepthDir)

    print("=> Generating list of images and corresponding depths for trainset and valset")
    local trainImagePath, trainDepthPath = findImageDepthMatches(
    trainImageDir, trainDepthDir
    )
    local valImagePath, valDepthPath = findImageDepthMatches(
    valImageDir, valDepthDir
    )

    local info = {
        val = {
            basedir = paths.concat(opt.data, 'val'),
            imagePath = valImagePath,
            depthPath = valDepthPath,
        },
        train = {
            basedir = opt.concat(opt.data, 'train'),
            imagePath = trainImagePath,
            depthPath = trainDepthPath,
        },
    }

    print(" | saving list of images and depths to " .. cacheFile)
    torch.save(cacheFile, info)
    
    return info
end

function M.augmentation(imageDirOrigin, depthDirOrigin, opt, split, trainDataPortion, trans)

    local imagePath = paths.concat(opt.data, split, 'image')
    local depthPath = paths.concat(opt.data, split, 'depth')
    paths.mkdir(imagePath)
    paths.mkdir(depthPath)

    -- find all origin image and depth matches
    imageDirOrigin = paths.concat(opt.data, imageDirOrigin)
    depthDirOrigin = paths.concat(opt.data, depthDirOrigin)
    local imagePathOrigin, depthPathOrigin = findImageDepthMatches(
    imageDirOrigin, depthDirOrigin
    )
    local sizeOrigin = #imagePathOrigin

    local startIndex
    local endIndex
    local size

    if split == 'train' then
        startIndex = 1
        endIndex = torch.round(sizeOrigin * trainDataPortion)
    else
        startIndex = torch.round(sizeOrigin * trainDataPortion) + 1
        endIndex = sizeOrigin
    end

    size = endIndex - startIndex + 1
    local targetSize =size * opt.incre

    -- firstly we scale depth image 
    -- number of data to generate for  each origin image
    local num = opt.incre
    print('=> original size of '..split..' '..size)
    print('=> after augmentation '..split..' '..targetSize)

    -- data augmentation compose
    local imageScale = T.Scale(345, 460)
    local depthScale = T.Scale(192, 256)

    for i = 1, sizeOrigin, 1 do
        local img = image.loadJPG(imagePathOrigin[i])
        img = imageScale(img)

        local depth = torch.load(depthPathOrigin[i])
        depth = depth:select(3, 4)
        depth = depthScale(depth)

        local basename = paths.basename(imagePathOrigin[i])
        basename = basename:match('img(.*).jpg$')

        local tmpImg
        local tmpDep

        print('=>process image: '..basename)

        for j = 1, num, 1 do
            tmpImg, tmpDep = img,depth
            tmpImg, tmpDep = trans(tmpImg, tmpDep)
            image.save(paths.concat(imagePath, 'img'..basename..'-'..j..'.jpg'), img)
            torch.save(paths.concat(depthPath, 'depth'..basename..'-'..j..'.t7'), depth)
        end
    end

end

return M
