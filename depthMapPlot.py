from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

#f, (ax1,ax2) = plt.subplots(1, 2,sharey=True)

#ax1.imshow(groundTruth,cmap='rainbow',interpolation='none');
#ax2.imshow(pred,cmap='rainbow',interpolation='none');
#ax3.imshow(image);

#plt.figure();
#plt.imshow(image);
gs = gridspec.GridSpec(1, 3,
                       width_ratios=[1,1,1]
                       );
ax1 = plt.subplot(gs[0]);
ax2 = plt.subplot(gs[1]);
ax3 = plt.subplot(gs[2]);
ax2.imshow(groundTruth,cmap='rainbow',interpolation='none');
ax3.imshow(pred,cmap='rainbow',interpolation='none');
ax1.imshow(image);
plt.show();
#print(result[b'image'].shape);
