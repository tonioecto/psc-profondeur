# psc-profondeur

## Data Augmentation Process
We use the method indicated in the article of `Eigen et al`. 

We augment the training data with random online transformations: 

- `Scale`: Input and target images are scaled by `s` in [1, 1.5], and 
the depths are divided by `s`. 
- `Rotation`: Input and target are rotated by `s` in [−5, 5] degrees. 
- `Translation`: Input and target are randomly cropped to the sizes we 
need. 
- `Color`: Input values are multiplied globally by a random RGB value 
`c` in `[0.8, 1.2]^3`. 
- `Flips`: Input and target are horizontally flipped with 0.5 probability. 

For the Make3D dataset, we use mask method introduced in the article.

Firstly, We resize all images to `345x460` and further reduce the resolution of
the RGB inputs to the network by half `(173x230)` because of the large 
architecture and hardware limitations. To get the output depth size, we resize 
mapping depth image to `96x128`.

During training, most of the target depth maps will have some missing 
values, particularly near object boundaries, windows and specular 
surfaces. We deal with these simply by masking them out and evaluating 
the loss only on valid points, *i.e.* we replace `n` in the loss function with the 
number of pixels that have a target depth, and perform the sums
excluding pixels i that have no depth value.

Precisely, we train against ground truth depth maps with masked out 
invalid pixels, where the depth is zero, as well as pixels that 
correspond to distances over 70m.

```lua
MaskMSECriterion

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



```
