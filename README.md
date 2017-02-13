# psc-profondeur

## Data Augmentation Process
We use the method indicated in the article of Eigen et al. 

We augment the training data with random online transformations: 

- `Scale`: Input and target images are scaled by $s \in [1, 1.5]$, and the depths are divided by s
- `Rotation`: Input and target are rotated by r \in [−5, 5] degrees
- `Translation`: Input and target are randomly cropped to the sizes indicated in Fig. 1
- `Color`: Input values are multiplied globally by a random RGB value c ∈ [0.8, 1.2]3
- `Flips`: Input and target are horizontally flipped with 0.5 probability

