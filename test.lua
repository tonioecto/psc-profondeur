require 'nn'
require 'models/modules/MaskMSECriterion'

input = torch.rand(2, 2)
print(input)

target = torch.eye(2, 2)
print(target)

criterion = nn.MaskMSECriterion(0.5, 0, false)

print(criterion:forward(input, target))
