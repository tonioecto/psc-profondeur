require 'nn'
require 'distributions'

local M = {}

function sampleGauss(m, mu, sigma)
    local w, g = m:getParameters()
    for i = 1, w:size()[1], 1 do
        w[i] = distributions.mvn.rnd(mu, sigma)
    end
    return m
end


function M.w_init(net)

    local mu = torch.Tensor({0})
    local sigma = torch.Tensor({0.01})
    
    for i,m in ipairs(net:listModules()) do
        if m.modules == nil then
            if m.__typename == 'nn.SpatialConvolution' then
                M.sampleGauss(m, mu, sigma)
            elseif m.__typename == 'cudnn.SpatialConvolution' then
                M.sampleGauss(m, mu, sigma)       
            end

            if m.bias then
                m.bias:zero()
            end

        end
    end

   return net
end


return w_init
