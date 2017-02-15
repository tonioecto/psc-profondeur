require 'paths'
require 'ffi'
local T = require 'dataset/transforms'
local unpack = unpack or table.unpack

local M = {}

local function findImageDepthMatches(imageDir, depthDir, opt)
    imageDir = paths.concat(opt.data, imageDir)
    depthDir = paths.concat(opt.data, depthDir)
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
    local imageDir = 'image'
    local depthDir = 'depth'

    assert(paths.dirp(paths.concat(opt.data, imageDir)), 'image directory not found: ' .. 
    paths.concat(opt.data, imageDir))
    assert(paths.dirp(paths.concat(opt.data, depthDir)), 'depth directory not found: ' .. 
    paths.concat(opt.data, depthDir))

    print("=> Generating list of images and corresponding depths")
    local imagePath, depthPath = findImageDepthMatches(
    imageDir, depthDir, opt
    )

    local info = {
        basedir = opt.data,
        imageSet = imagePath,
        depthSet = depthPath
    }

    print(" | saving list of images and depths to " .. cacheFile)
    torch.save(cacheFile, info)
    return info
end

function M.augmentation(imageDirOrigin, depthDirOrigin, opt)
    local imagePath = paths.concat(opt.data, 'image')
    local depthPath = paths.concat(opt.data, 'depth')
    paths.mkdir(imagePath)
    paths.mkdir(depthPath)

    -- find all origin image and depth matches
    local imagePathOrigin, depthPathOrigin = findImageDepthMatches(
    imageDirOrigin, depthDirOrigin, opt
    )
    local sizeOrigin = #imagePathOrigin

    -- firstly we scale depth image 
    -- number of data to generate for  each origin image
    local num = opt.sizeAugmented/sizeOrigin
    print('=> original size '..sizeOrigin)
    print('=> after augmentation '..opt.sizeAugmented)
    -- data augmentation compose
    -- create transform function table
    local trans = T.Compose({
        T.RandomScale(1, 1.5),
        T.HorizontalFlip(0.5),
        T.Rotation(5),
        T.Color(0.8, 1,2),
        T.RandomCrop(173, 230, 96, 128)
    })

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

        print('=>processe image: '..basename)

        for j = 1, num, 1 do
            tmpImg, tmpDep = img,depth
            tmpImg, tmpDep = trans(tmpImg, tmpDep)
            image.save(paths.concat(imagePath, 'img'..basename..'-'..j..'.jpg'), img)
            torch.save(paths.concat(depthPath, 'depth'..basename..'-'..j..'.t7'), depth)
        end
    end

end

return M
