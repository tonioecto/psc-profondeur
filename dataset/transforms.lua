require 'image'

local M = {}

function M.Compose(transforms)
    return function(input)
        for _, transform in ipairs(transforms) do
            input = transform(input)
        end
        return input
    end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
    interpolation = interpolation or 'bicubic'
    return function(input)
        local w, h = input:size(3), input:size(2)
        if (w <= h and w == size) or (h <= w and h == size) then
            return input
        end
        if w < h then
            return image.scale(input, size, h/w * size, interpolation)
        else
            return image.scale(input, w/h * size, size, interpolation)
        end
    end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
    return function(input)
        local w1 = math.ceil((input:size(3) - size)/2)
        local h1 = math.ceil((input:size(2) - size)/2)
        return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
    end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
    padding = padding or 0

    return function(input)
        if padding > 0 then
            local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
            temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
            input = temp
        end

        local w, h = input:size(3), input:size(2)
        if w == size and h == size then
            return input
        end

        local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
        local out = image.crop(input, x1, y1, x1 + size, y1 + size)
        assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
        return out
    end
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
    local centerCrop = M.CenterCrop(size)

    return function(input)
        local w, h = input:size(3), input:size(2)

        local output = {}
        for _, img in ipairs{input, image.hflip(input)} do
            table.insert(output, centerCrop(img))
            table.insert(output, image.crop(img, 0, 0, size, size))
            table.insert(output, image.crop(img, w-size, 0, w, size))
            table.insert(output, image.crop(img, 0, h-size, size, h))
            table.insert(output, image.crop(img, w-size, h-size, w, h))
        end

        -- View as mini-batch
        for i, img in ipairs(output) do
            output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
        end

        return input.cat(output, 1)
    end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
    return function(input)
        local w, h = input:size(3), input:size(2)

        local targetSz = torch.random(minSize, maxSize)
        local targetW, targetH = targetSz, targetSz
        if w < h then
            targetH = torch.round(h / w * targetW)
        else
            targetW = torch.round(w / h * targetH)
        end

        return image.scale(input, targetW, targetH, 'bicubic')
    end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size)
    local scale = M.Scale(size)
    local crop = M.CenterCrop(size)

    return function(input)
        local attempt = 0
        repeat
            local area = input:size(2) * input:size(3)
            local targetArea = torch.uniform(0.08, 1.0) * area

            local aspectRatio = torch.uniform(3/4, 4/3)
            local w = torch.round(math.sqrt(targetArea * aspectRatio))
            local h = torch.round(math.sqrt(targetArea / aspectRatio))

            if torch.uniform() < 0.5 then
                w, h = h, w
            end

            if h <= input:size(2) and w <= input:size(3) then
                local y1 = torch.random(0, input:size(2) - h)
                local x1 = torch.random(0, input:size(3) - w)

                local out = image.crop(input, x1, y1, x1 + w, y1 + h)
                assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

                return image.scale(out, size, size, 'bicubic')
            end
            attempt = attempt + 1
        until attempt >= 10

        -- fallback
        return crop(scale(input))
    end
end

function M.HorizontalFlip(prob)
    return function(input)
        if torch.uniform() < prob then
            input = image.hflip(input)
        end
        return input
    end
end

function M.Rotation(deg)
    return function(input)
        if deg ~= 0 then
            local input = image.rotate(input, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
        end
        return input
    end
end


function M.RandomOrder(ts)
    return function(input)
        local img = input.img or input
        local order = torch.randperm(#ts)
        for i=1,#ts do
            img = ts[order[i]](img)
        end
        return img
    end
end

function M.mask(depth, threshold)
    return torch.clamp(depth, 0, threshold)
end

return M
