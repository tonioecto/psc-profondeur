require 'image'

local M = {}

function M.Compose(transforms)
    return function(img, depth)
        for _, transform in ipairs(transforms) do
            img, depth = transform(img, depth)
            print(#img)
            print(#depth)
        end
        return img, depth
    end
end

-- Scales the smaller edge to size
function M.Scale(width, height, interpolation)
    interpolation = interpolation or 'bicubic'
    return function(input)
        return image.scale(input, width, height, interpolation)
    end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minRatio, maxRatio)
    return function(img, depth)
        print('RamdomScale')
        local ratio = torch.random(minRatio, maxRatio)
        print(ratio)

        local w, h = img:size(3), img:size(2)
        local wDepth, hDepth = depth:size(2), depth:size(1)

        local targetW = torch.round(w / ratio)
        local targetH = torch.round(h / ratio)
        local targetWDepth = torch.round(wDepth / ratio)
        local targetHDepth = torch.round(hDepth / ratio)

        -- divide depth by the ratio 
        depth = depth / ratio

        return image.scale(img, targetW, targetH, 'bicubic'), 
        image.scale(depth, targetWDepth, targetHDepth, 'bicubic')
    end
end

-- Random crop from larger image
function M.RandomCrop(widthImage, heightImage, widthDepth, heightDepth)
    return function(img, depth)
        print('RandomCrop')
        local w, h = img:size(3), img:size(2)
        local wD, hD = depth:size(2), depth:size(1)

        print(w..'  '..h..'  '..widthImage..'  '..heightImage)
        print(wD..'  '..hD..'  '..widthDepth..'  '..heightDepth)
        assert(w >= widthImage and h >= heightImage, 'wrong crop size for image')
        assert(wD >= widthDepth and hD >= heightDepth, 'wrong crop size for depth')

        if w == widthImage and h == heightImage then
            return img, depth
        end

        local x1, y1 = torch.random(0, w - widthImage), torch.random(0, h - heightImage)
        local x2, y2 = torch.round(x1 / w * wD), torch.round(y1 / h * hD)

        local outImg = image.crop(img, x1, y1, x1 + widthImage, y1 + heightImage)
        local outDep = image.crop(depth, x2, y2, x2 + widthDepth, y2 + widthDepth)
        return outImg, outDep
    end
end

-- flip the image horizontally with probability prob
function M.HorizontalFlip(prob)
    return function(img, depth)
        print('RandomFlip')
        if torch.uniform() < prob then
            img = image.hflip(img)
            depth = image.hflip(depth)
        end
        return input, depth
    end
end

-- rotate deg degrees from -deg to deg
function M.Rotation(deg)
    return function(img, depth)
        print('RandomRotate')
        local ratio = (torch.uniform() - 0.5) * 2
        if deg ~= 0 then
            print(deg * ratio)
            img = image.rotate(img, ratio * deg * math.pi / 180, 'bilinear')
            depth = image.rotate(depth, ratio * deg * math.pi / 180, 'bilinear')
        end
        return img, depth
    end
end

return M
