local threads = require 'threads'

local nThreads = 10
local size, batchSize = 99, 10
local perm = torch.randperm(size)


local manualSeed = 1000

local function main(idx)
    return size
end

-- initialize threads
local threads,_ = threads(nThreads, main)

--print(threads)


local idx, sample = 1, nil
local function enqueue()
    while idx <= size and threads:acceptsjob() do
        local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))

        threads:addjob(
        function(indices)
            return indices
        end,
        function(_sample_)
            sample = _sample_
        end,
        indices
        )
        idx = idx + batchSize
    end
end

local n = 0

local function loop()
    enqueue()
    if not threads:hasjob() then
        return nil
    end
    threads:dojob()
    if threads:haserror() then
        threads:synchronize()
    end
    enqueue()
    n = n + 1
    return n, sample
end

for i, sample in loop do
    print(i)
    print(sample)
end
