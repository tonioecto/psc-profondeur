require 'nn'
require 'distributions'

local M = {}

function M.sampleGauss(m, gen, mu, sigma)
    local w, g = m:getParameters()
    local size = w:size()
    w:copy(sigma * torch.randn(size))
    return m
end


function M.w_init(net)

    local mu = 0
    local sigma = 0.01
    local gen = torch.Generator()
    
    for i,m in ipairs(net:listModules()) do
        if m.modules == nil then
            if m.__typename == 'nn.SpatialConvolution' then
                M.sampleGauss(m, gen, mu, sigma)
            elseif m.__typename == 'cudnn.SpatialConvolution' then
                M.sampleGauss(m, gen, mu, sigma)       
            end

            if m.bias then
                m.bias:zero()
            end

        end
    end

   return net
end


return M
