require 'nn'
require 'models/modules/MaskMSECriterion'

input = torch.rand(300, 300)
--print(input)

target = torch.rand(300, 300)
--print(target)

criterion = nn.MaskMSECriterion(0.5, 0, false)

print(criterion:forward(input, target))
print(criterion:backward(input, target))
