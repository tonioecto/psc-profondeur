require 'nn'
require 'torch'
require 'cutorch'
require 'cudnn'
--require 'cunn'

-- load initial resnet-50
model = torch.load('resnet-50.t7')
print('ResNet and up-projection \n' .. model:__tostring())

--get the location of 'cudnn.ReLU'
num = 0
model:replace(function(module)
    if torch.typename(module)== 'cudnn.ReLU' then
        num = num - 1
        if num == 0  then
            return nn.Identity()
        end
    end
end)
print(num)

-- delete modules: 'cudnn.ReLU', 'cudnn.SpatialAveragePooling',
-- 'nn.Linear' and 'nn.View'
model:replace(function(module)
    if torch.typename(module)== 'cudnn.ReLU' then
        num = num - 1
        if num == 0  then
            return nn.Identity()
        end
    end
    if torch.typename(module) == 'cudnn.SpatialAveragePooling' then
        return nn.Identity()
    elseif torch.typename(module) == 'nn.Linear' then
        return nn.Identity()
    elseif torch.typename(module) == 'nn.View' then
        return nn.Identity()
    else
        return module
    end
end)

-- save the changed model
print('ResNet and up-projection \n' .. model:__tostring())
torch.save('ResNet50.t7', model)
