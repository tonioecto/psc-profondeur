require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
--require 'matio'
require 'xlua'
require 'torch'
require 'cutorch'
require 'cudnn'
require 'cunn'


--Up-projection
-- implement simple version of up-convolution
function upConvolution(net, d1, d2)
    net:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    net:add(cudnn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    net:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    net:add(cudnn.ReLU())
end

-- implement simple version of up-projection
function upProjection(net, d1, d2)
    local cat = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch1:add(cudnn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch1:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    branch1:add(cudnn.ReLU())
    branch1:add(cudnn.SpatialConvolution(d2, d2, 3, 3, 1, 1, 1, 1))

    local branch2 = nn.Sequential()
    branch2:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    branch2:add(cudnn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    branch2:add(nn.SpatialZeroPadding(0, -1, 0, -1))

    cat:add(branch1)
    cat:add(branch2)
    net:add(cat)
    net:add(nn.CAddTable())

    net:add(cudnn.ReLU())
end


--model_resnet = torch.load('nvnet.t7')
model = torch.load('resnet-50.t7')
nombre = 49
model:replace(function(module)
    if torch.typename(module)== 'cudnn.ReLU' then
        nombre = nombre - 1
        if nombre == 0  then
            return nn.Identity()
        end
    end
    if torch.typename(module) == 'cudnn.SpatialAveragePooling' then
        return nn.Identity()
    elseif torch.typename(module) == 'nn.Linear' then
        return nn.Identity()
    elseif torch.typename(module) == 'nn.View' then
        return nn.Identity() 
    elseif torch.typename(module)== 'nn.SpatialBatchNormalization' then
        return nn.Identity()
    else

        return module
    end

end)

torch.save('ResNet50woBN.t7', model)

--print('CRN net\n' .. model:__tostring())
--a=torch.Tensor(10,3,304,228)
--a=a:cuda()
--model=model:cuda()
--print(#model:forward(a))

-- verify resnet-50 structure



net = nn.Sequential()
--net:add(model)

d0 = 2048
d1 = 1024
net:add(nn.SpatialConvolution(d0, d1, 1, 1, 1, 1))
--net:add(nn.BatchNormalization())


--build up projection blocks
up_projection = nn.Sequential()
upProjection(up_projection, 1024, 512)
upProjection(up_projection, 512, 256)
upProjection(up_projection, 256, 128)
upProjection(up_projection, 128, 64)

net:add(up_projection)

--d_final = 64
--net:add(cudnn.ReLU())

print('CRN net\n' .. model:__tostring())
a = torch.Tensor(2048, 10, 8)
a=a:cuda()
net = net:cuda()
print(#net:forward(a))
