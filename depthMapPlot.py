from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchfile

#depth = np.random.random((128,64));
#np.savetxt('depthTest.txt',depth);

result = torchfile.load('result/visual-r-22.t7');
image = result[b'image'];


image = image.transpose(1,2,0);
print(image.shape);
#iamge = image.transpose(0,2,1);
#print(image.shape);

pred = np.reshape(result['pred'],(128,96));
groundTruth = result['groundTruth'];

f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)

ax1.imshow(groundTruth,cmap='gist_rainbow',interpolation='none');
ax2.imshow(pred,cmap='gist_rainbow',interpolation='none');

plt.figure();
plt.imshow(image);
plt.show();
#print(result[b'image'].shape);
