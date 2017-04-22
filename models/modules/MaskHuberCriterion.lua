local MaskHuberCriterion, parent = torch.class('nn.MaskHuberCriterion', 'nn.Criterion')

-- forward propagation function 
local function f(c)
  return function (x)
    if x > c then
      return (x*x + c*c) / (2.0 * c)
    else
      return x
    end
  end
end

-- back propagation function with threshold c
local function g(c)
  return function(x)
    if x > c then
      return x / c
    else
      return 1
    end
   end
end

function MaskHuberCriterion:__init(low, high, delta)
  parent.__init(self)
  self.highMask = high
  self.lowMask = low
  self.c = self.delta or delta -- Boundary
  self.alpha = torch.Tensor() -- Residual
end

function MaskHuberCriterion:updateOutput(input, target)
  -- Calculate residual
  
  self.alpha = target - input
  
  self.m, self.mInverse = self:mask(target)
  self.nValid = torch.sum(self.mInverse)

  self.absAlpha = torch.abs(self.alpha)

  local temp = self.absAlpha:clone()
  temp:maskedFill(self.m, 0)
  -- update threshold
  self.delta = 0.2 * temp:max()
  -- apply herbu function f to absAlpha
  self.diffAlpha = self.absAlpha:apply(f(self.c))

  self.diffAlpha:maskedFill(self.m, 0)

  self.output = self.diffAlpha:sum() / self.nValid
  
  return self.output
end

function MaskHuberCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(target)

  self.gradInput = self.alpha:sign():cmul(self.absAlpha:apply(g(self.c)))
  -- mask invalid pixels
  self.gradInput:maskedFill(self.m, 0)
  self.gradInput = self.gradInput / self.nValid

  local temp = self.c
  self.c = self.delta

  print('=> change huber c from : '..temp..'  to  '..self.c)

  return self.gradInput
end

function MaskHuberCriterion:mask(target)
    local g = torch.ge(target, self.highMask)
    local l = torch.eq(target, self.lowMask)
    return l + g, 1 - (l + g)
end

return nn.MaskHuberCriterion