require 'nn'
require 'cudn'
-- load pretrained ResNet-50 model implemented by facebook
-- based on ImageNet Dataset
model = torch.load('resnet-50.t7')
-- try to convert CUDA model to a CPU version for computers without GPU
model = model:float()
--print pre-trained ResNet model to observe its structure
--print('CRN net\n' .. model:__tostring())

--delete extra final ReLU module 
threshold_nodes, container_nodes = model:findModules('nn.ReLU')
n1 = #threshold_nodes
n2 = #threshold_nodes[n1].modules


--verify new structure of ResNet-50
print('CRN net\n' .. model:__tostring())