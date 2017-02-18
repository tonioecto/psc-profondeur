local threads = require 'threads'

local nthread = 4
local njob = 20
local msg = "hello from a satellite thread"

local ts = threads.Threads(
nthread, 
function (idx)
    print(idx) end
    )

local value = torch.zeros(20, 20)
-- add jobs
for i = 1, 20, 1 do
    for j = 1, 20, 1 do
        ts:addjob(
        function (i, j)
            queuevalue = value
            print(i+j)
            print ('Thread id -- %d -- is working.', __threadid)
            return i, j, 1
        end,
        function (i, j, inc)
            m = math.max(i-1, 1)
            n = math.max(j-1, 1)
            value[j][i] = value[n][m] + inc
        end, i + 100, j + 100
        )
    end
end

ts:dojob()
ts:synchronize()
print(value)
