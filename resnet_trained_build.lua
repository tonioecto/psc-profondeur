require 'nn'
-- load pretrained ResNet-50 model implemented by facebook
-- based on ImageNet Dataset
model = torch.load('resnet-50.t7')
-- try to convert CUDA model to a CPU version for computers without GPU
model = model:float()
--print pre-trained ResNet model to observe its structure
print('CRN net\n' .. model:__tostring())

--[[
--delete extra final ReLU module 
threshold_nodes, container_nodes = model:findModules('nn.ReLU')
for i = 1, #threshold_nodes do
    print(i)
  -- Search the container for the current threshold node
  for j = 1, #(container_nodes[i].modules) do
        print(j)
    if container_nodes[i].modules[j] == threshold_nodes[i] then
      -- Replace with a new instance
      container_nodes[i].modules[j] = nn.Tanh()
    end
  end
end

--delete Avg : Average Pooling Module
--delete View Module
--delete Linear Module
model:replace(function(module)
   if torch.typename(module) == 'nn.SpatialAveragePooling'
            or torch.typename(module) == 'nn.View'
            or torch.typename(module) == 'nn.Linear'
      return nn.Identity()
   else
      return module
   end
end)

--verify new structure of ResNet-50
print('CRN net\n' .. model:__tostring())
]]