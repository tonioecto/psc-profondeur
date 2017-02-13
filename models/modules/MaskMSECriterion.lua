local MaskMSECriterion, parent = torch.class('nn.MaskMSECriterion', 'nn.Criterion')

function MaskMSECriterion:__init(highMask, lowMask, sizeAverage)
    parent.__init(self)
    self.highMask = highMask
    print(self.highMask)
    self.lowMask = lowMask
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
end

function MaskMSECriterion:updateOutput(input, target)
    self.m = self:mask(target)
    input = input:cmul(self.m)
    target = target:cmul(self.m)
    self.output_tensor = self.output_tensor or input.new(1)
    input.THNN.MSECriterion_updateOutput(
    input:cdata(),
    target:cdata(),
    self.output_tensor:cdata(),
    self.sizeAverage
    )
    self.output = self.output_tensor[1]
    return self.output
end

function MaskMSECriterion:updateGradInput(input, target)
    self.m = self:mask(target)
    input = input:cmul(self.m)
    target = target:cmul(self.m)
    input.THNN.MSECriterion_updateGradInput(
    input:cdata(),
    target:cdata(),
    self.gradInput:cdata(),
    self.sizeAverage
    )
    return self.gradInput
end

function MaskMSECriterion:mask(target)
    local g = torch.ge(target, self.highMask)
    local l = torch.eq(target, self.lowMask)
    print(l + g)
    return l + g
end
