local MaskMSECriterion, parent = torch.class('nn.MaskMSECriterion', 'nn.Criterion')

function MaskMSECriterion:__init(highMask, lowMask, sizeAverage)
    parent.__init(self)
    self.highMask = highMask
    self.lowMask = lowMask
    self.mse = nn.MSECriterion(false)
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

    self.output = self.mse:updateOutput(input, target)

    if(self.sizeAverage) then
        self.output = self.output / self.nValid
    end
    
    return self.output
end

function MaskMSECriterion:updateGradInput(input, target)
    self.m, self.mInverse = self:mask(target)

    input:maskedFill(self.m, 0)
    target:maskedFill(self.m, 0)

    self.nValid = torch.sum(self.mInverse)

    self.gradInput = self.mse:updateGradInput(input, target)

    if self.sizeAverage then
        self.gradInput = self.gradInput / self.nValid
    end
    
    return self.gradInput
end

function MaskMSECriterion:mask(target)
    local g = torch.ge(target, self.highMask)
    local l = torch.eq(target, self.lowMask)
    return l + g, 1 - (l + g)
end
