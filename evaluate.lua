require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'
local gnu = require 'gnuplot'

local M = {}

-- highmask = 70
-- lowmask = 0
local function mask(prediction, groundtruth)
  local g = torch.ge(groundtruth, 70)
  local l = torch.eq(groundtruth, 0)
  local m = l + g
  local mInverse = 1 - (l+g)
  prediction:maskedFill(m, -1)
  groundtruth:maskedFill(m, -1)
  local nValid = torch.sum(mInverse)
  return prediction, groundtruth, nValid
end

function M.plotTrainLoss(opt)
  local lossFilePath = paths.concat(opt.lossFile, 'loss.t7')
  local loss = torch.load(lossFile)
  -- convert a table to tensor, and transpose it
  loss = torch.Tensor(loss)
  loss = loss:transpose(1, 2)
  -- plot trainErr and validationErr graph
  gnuplot.plot({'TrainErr', loss[2]},
    {'ValErr', loss[3]})
end

function M.Display(pred,real,preName,realName)
  gnu.pdffigure(preName)
  gnu.imagesc(pred,'color')
  gnuplot.plotflush()
  gnu.pdffigure(realName)
  gnu.imagesc(real,'color')
  gnuplot.plotflush()
end

local function Relerror(p, gt)
  local dis = torch.abs(p - gt)
  dis:cdiv(gt)
  return torch.sum(dis)
end

local function Rmserror(p,gt)
  local dis = torch.dist(p,gt)
  return dis * dis
end

local function Rmslogerr(p,gt)
  local term = math.log(10)
  local pre = torch.log(p)
  pre:div(term)
  local truth = torch.log(gt)
  truth:div(term)
  local dis = torch.dist(pre,truth)
  return dis * dis
end

local function Logerr(p,gt)
  local pre = torch.log(p)
  pre:div(term)
  local truth = torch.log(gt)
  truth:div(term)
  local dis = torch.abs(pre-truth)
  return torch.sum(dis)
end

local function Thresherr(p,gt,i)
  local Thresh = math.pow(1.25,i)
    local a = torch.cdiv(p,gt)
    local b = torch.cdiv(gt,p)
    local c = torch.div(torch.abs(a-b),2)
    c:add((a+b)/2)
    c = Thresh - c
    c:sign()
    c:add(1)
    return torch.sum(c)/2
  --err = err/Tsize
  --return err*100
end

function M.errEvaluate(predicted,groudtruth)
  local relErr = 0
  local rmsErr = 0
  local rmLogErr = 0
  local logErr = 0
  local threshErr1 = 0
  local threshErr2 = 0
  local threshErr3 = 0
  local Tsize = 0
  for i =1, predicted:size(1),1 do
    local p, gt, nvalid = mask(predicted[i],groundtruth[i])
    relErr = relErr + Relerror(p,gt)
    rmsErr = rmsErr + Rmserror(p,gt)
    rmLogErr = rmLogErr + Rmslogerr(p,gt)
    logErr = logErr + Logerr(p,gt)
    threshErr1 = threshErr1 + Thresherr(p,gt,1)
    threshErr2 = threshErr2 + Thresherr(p,gt,2)
    threshErr3 = threshErr3 + Thresherr(p,gt,3)
    Tsize = Tsize + nvalid
  end
  Tsize = Tsize * 1.0
  relErr = relErr / Tsize

  rmsErr = rmsErr / Tsize
  rmsErr = math.sqrt(rmsErr)

  rmLogErr = rmLogErr / Tsize
  rmLogErr = math.sqrt(rmLogErr)

  logErr = logErr / Tsize

  threshErr1 = threshErr1/Tsize
  threshErr1 = threshErr1 * 100

  threshErr2 = threshErr2/Tsize
  threshErr2 = threshErr2 * 100

  threshErr3 = threshErr3/Tsize
  threshErr3 = threshErr3 * 100

  print("Mean Absolute Relative Error : "..relErr)
  print('Root Mean Squared Error (rms) : '..rmsErr)
  print('Root Mean Squared Log-Error(rms(log)) : '..rmLogErr)
  print('Mean log10 Error(log10) : '..logErr)
  print('Thresh err 1 : '..threshErr1)
  print('Thresh err 2 : '..threshErr2)
  print('Thresh err 3: '..threshErr3)

  --return err*100
end

return M
