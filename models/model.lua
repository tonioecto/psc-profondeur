--We import necessairy packages
require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'
require '/models/modules/UnPoolingCuda'

local M = {}

-- Defin simple up-convolution block
function M.upConvolution(net, d1, d2)
    net:add(nn.SpatialZeroPadding(0, 1, 0, 1))
    net:add(cudnn.SpatialFullConvolution(d1, d2, 5, 5, 2, 2, 2, 2))
    net:add(nn.SpatialZeroPadding(0, -1, 0, -1))
    net:add(cudnn.ReLU())
end

-- Define simple up-projection block
function M.upProjection(net, d1, d2)
    
    net:add(nn.UnPooling(2))
    local cat = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(cudnn.SpatialConvolution(d1, d2, 5, 5, 1, 1, 2, 2))
    branch1:add(cudnn.ReLU())
    branch1:add(cudnn.SpatialConvolution(d2, d2, 3, 3, 1, 1, 1, 1))

    local branch2 = nn.Sequential()
    branch2:add(cudnn.SpatialConvolution(d1, d2, 5, 5, 1, 1, 2, 2))

    cat:add(branch1)
    cat:add(branch2)
    net:add(cat)
    net:add(nn.CAddTable())

    net:add(cudnn.ReLU())
end

-- max un pooling module constructor
function maxUnPoolingModule(batchSize, d1, d2, d3)
    
    local c = torch.Tensor(batchSize, d1, d2, d3)
    c = c:cuda()
    local t = torch.Tensor(d2, d3)
    t = t:cuda()
    
    for i = 1, d3, 1 do
        for j = 1, d2, 1 do
            if i%2 == 0 then
                t[j][i] = 0
            elseif j%2 == 0 then
                t[j][i] = 0
            else
                t[j][i] = 1
            end
        end
    end
    
    for k = 1, d1, 1 do
        for l = 1, batchSize, 1 do
            c[l][k] = t
        end
    end
    
    local mp = nn.SpatialMaxPooling(2, 2, 2, 2)
    mp = mp:cuda()
    mp:forward(c)
    
    local mup = nn.SpatialMaxUnpooling(mp)
    
    return mup

end

return M
