local MaskMSECriterion, parent = torch.class('nn.MaskMSECriterion', 'nn.Criterion')

function MaskMSECriterion:__init(highMask, lowMask, sizeAverage)
    parent.__init(self)
    self.highMask = highMask
    self.lowMask = lowMask
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
end

function MaskMSECriterion:updateOutput(input, target)
    self.m, self.mInverse = self:mask(target)

    input:maskedFill(self.m, 0)
    target:maskedFill(self.m, 0)
    
    self.nValid = torch.sum(self.mInverse)
    
    self.output_tensor = self.output_tensor or input.new(1)
    input.THNN.MSECriterion_updateOutput(
    input:cdata(),
    target:cdata(),
    self.output_tensor:cdata(),
    false
    )
    
    if(self.sizeAverage) then
        self.output = self.output_tensor[1] / self.nValid
    else
        self.output = self.output_tensor[1]
    end
    
    return self.output
end

function MaskMSECriterion:updateGradInput(input, target)
    self.m, self.mInverse = self:mask(target)

    input:maskedFill(self.m, 0)
    target:maskedFill(self.m, 0)

    self.nValid = torch.sum(self.mInverse)

    input.THNN.MSECriterion_updateGradInput(
    input:cdata(),
    target:cdata(),
    self.gradInput:cdata(),
    false
    )
    
    if self.sizeAverage then
        return self.gradInput / self.nValid
    else
        return self.gradInput
    end
end

function MaskMSECriterion:mask(target)
    local g = torch.ge(target, self.highMask)
    local l = torch.eq(target, self.lowMask)
    return l + g, 1 - (l + g)
end
