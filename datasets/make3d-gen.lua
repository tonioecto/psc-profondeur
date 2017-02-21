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
            -- create possible corresponding depth name
            local depth = {
                'depth_sph_corr'..basename..'.t7',
                'depth'..basename..'.t7',
            }
            for _,d in ipairs(depth) do
                if paths.filep(paths.concat(depthDir, d)) then
                    table.insert(imagePath, paths.concat(imageDir, img))
                    table.insert(depthPath, paths.concat(depthDir, d))
                    break
                end
            end
        end
    end

    print('=> Generate image path file and depth path file with size '
    ..#imagePath..' and '..#depthPath)

    return imagePath, depthPath
end

-- generate info file for val and train dataset
function M.exec(opt, cacheFile, split)
    -- find the image and depth matches
    if split ~= 'train' and (split ~= 'val' and split ~= 'test') then
        error('not a valid split label: '..split)
    end
    
    local imageDir = paths.concat(opt.data, split, 'image')
    local depthDir = paths.concat(opt.data, split, 'depth')

    assert(paths.dirp(imageDir), split..' image directory not found: ' .. imageDir)
    assert(paths.dirp(depthDir), split..'depth directory not found: ' .. depthDir)

    print("=> Generating list of images and corresponding depths for "..split..' set')
    local imagePath, depthPath = findImageDepthMatches(imageDir, depthDir)

    local info = {
        basedir = paths.concat(opt.data, split),
        imagePath = imageDir,
        depthPath = depthDir,
    }

    print(" | saving list of images and depths to " .. cacheFile)
    torch.save(cacheFile, info)
    
    return info
end

-- data augementation process for all inputs in the original directory
function M.augmentation(imageDirOrigin, depthDirOrigin, opt, split, trainDataPortion, trans)

    -- whole data augmentation for all orginal path images
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

    for i = startIndex, endIndex, 1 do
        local img = image.loadJPG(imagePathOrigin[i])
        img = imageScale(img)

        local depth = torch.load(depthPathOrigin[i])
        depth = depth:select(3, 4)
        depth = depthScale(depth)

        local basename = paths.basename(imagePathOrigin[i])
        basename = basename:match('img(.*).jpg$')

        local tmpImg
        local tmpDep

        print('=> process image: '..basename)

        for j = 1, num, 1 do
            tmpImg, tmpDep = trans(img, depth)
            image.save(paths.concat(imagePath, 'img'..basename..'-'..j..'.jpg'), tmpImg)
            torch.save(paths.concat(depthPath, 'depth'..basename..'-'..j..'.t7'), tmpDep)
        end
    end

end

function M.augOneMatch(img, depth, trans)
    return trans(img, depth)
end

return M
