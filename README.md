# psc-profondeur

## Data Augmentation Process
We use the method indicated in the article of Eigen et al. 

We augment the training data with random online transformations: 

- `Scale`: Input and target images are scaled by `s` in [1, 1.5], and 
the depths are divided by `s`. 
- `Rotation`: Input and target are rotated by `s` in [−5, 5] degrees. 
- `Translation`: Input and target are randomly cropped to the sizes we 
need. 
- `Color`: Input values are multiplied globally by a random RGB value 
`c` in [0.8, 1.2]. 
- `Flips`: Input and target are horizontally flipped with 0.5 probability. 

For the Make3D dataset, we use mask method introduced in the article.

We resize all images to `345x460` and further reduce the resolution of
the RGB inputs to the network by half `(173x230)` because of the large 
architecture and hardware limitations.

To reduce the effect of meaningless candidates in sky regions, we used a 
classifier to label sky pixels and for the depth of the corresponding 
superpixels to take the value (0, 0, 1, 80).

T
