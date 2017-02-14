require 'paths'
require 'ffi'

local M = {}

local function findImageDepthMatches(imageDir, depthDir, opt)
    imageDir = paths.concat(opt.data, imageDir)
    depthDir = paths.conca(opt.data, depthDir)
    local dirsImage = paths.dir(imageDir)
    table.sort(dirs)
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
    local sizeOrigin = #imagePath
    paths.mkdir(imagePath)
    paths.mkdir(depthPath)

    -- find all origin image and depth matches
    local imagePathOrigin, depthPathOrigin = findImageDepthMatches(
     imageDirOrigin, depthDirOrigin, opt
    )

    -- number of data to generate for  each origin image
    local num = opt.sizeAugmented/#imagePath
    -- data augmentation compose
    for i = 1, sizeOrigin, 1 do
        for j = 1, num, 1 do
            
end

return M
