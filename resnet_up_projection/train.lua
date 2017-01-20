require 'nn'
require 'image'
require 'paths'
require 'os'
require 'math'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'


function train(net, criterion, batchLoader, trainSet)

    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 1 -- just do 5 epochs of training.

    for i=1,20,1 do
        local input = batchLoader.miniBatchload(trainSet, 5)  
        input.image = input.image:cuda()
        input.depth = input.depth:cuda()
        trainer:train(input)
    end
    
end