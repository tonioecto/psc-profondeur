require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
--require 'matio'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'

function loadDataset(imageSet,depthSet)       --Load the images and depthMap, and generate dataset for trainning
    local imagename = {}
    local depthname = {}

    for file in paths.files(imageSet) do
        if file:find(".*(jpg)$") then
            table.insert(imagename, paths.concat(imageSet,file))
        end
    end

    if #imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    table.sort(imagename, function (a,b) return a < b end)
    --print(#imagename)

    local indexFakefile = {}
    for i,file in ipairs(imagename) do
        --print(file)
        local name = file:match('.+/.+img(.*).jpg$')
        --print(name)
        local index = 'depth'..name
        local fullname = index..'.mat'
        local matname = paths.concat(depthSet,fullname)
        if paths.filep(matname)then
            table.insert(depthname,matname)
        else
            --table.remove(imagename,i)
            --i = i -1
            table.insert(indexFakefile,i)
        end
    end
    for i=#indexFakefile,1,-1 do
        table.remove(imagename,indexFakefile[i])
    end

    print('The number of image is:'..#imagename)
    print('The number of correponding depthmap is:'..#depthname)

    if #imagename == 0 then
        error('given directory doesn\'t contain any JPG files')
    end

    local imageSet = torch.Tensor(#imagename,3,456,608)  -- On peut adapter la taille pour avoir une coupe plus ou moins grande
    --local depthSet = torch.Tensor(#depthname,160,128)
    local depthSet = torch.Tensor(#depthname,456,608)
    local mat = require 'matio'

    for i,file in ipairs(imagename) do
        --local m = image.loadJPG(file)
        --print(file)
        local m = image.load(file)
        m = image.scale(m,608,456,'bicubic')
        imageSet[i] = m
    end

    for i,file in ipairs(depthname) do
        local m = mat.load(file,'depthMap')
        m = image.scale(m,456,608,'bicubic')

        m = image.hflip(m)
        m = m:transpose(1,2)

        depthSet[i] = m
    end

    return imageSet, depthSet, #imagename

end

function boot(image1, image2,i, nombre)

  --print(#image2)

  for j=1,nombre,1 do

    local symh = (math.random(1, 10) > 5);
    local symv = (math.random(1, 10) > 5);
    local x = math.random(1,303)
    local y = math.random(1,227)

    image3 = image.crop(image1, x,y, x+304,y+228)
    image4 = image.crop(image2, x,y ,x+304,y+228)


    if symh then
      image3 =  image.hflip(image3)
      image4 =  image.hflip(image4)
    end

    if symv then
      image3 =  image.vflip(image3)
      image4 =  image.vflip(image4)
    end

    path = '/home/niva/Desktop/psc/bootstrap/'..i..'-'..j..'.jpeg'
    image.save(path, image3)
    --image.display(image3)

    path = '/home/niva/Desktop/psc/bootstrapDepth/'..i..'-'..j..'.jpeg'
    image4=image.scale(image4,160,128,'bicubic')
    --image.display(image4)

    image.save(path, image4)  -- A changer si vous voulez un .mat
  end
end



imageSet, depthSet, taille = loadDataset("minibatch", "depthTest")

print(taille)
for i=1, taille, 1 do
  boot(imageSet[i], depthSet[i],i,20)
  print(i)
end


--local img = image.load(imagename[1],3,'byte')
--image.display(img)
