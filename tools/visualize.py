import matplotlib.pyplot as plt
import numpy as np

import doodle

generator = doodle.NeuralGenerator()
generator.prepare_network()

def calculate_patch_coordinates(l, j, i):
    ys, xs, ye, xe = j, i, j, i
    while hasattr(l, 'filter_size'):
        after = l.filter_size[0]//2
        before = l.filter_size[0] - 1 - after
        ys -= before
        xs -= before
        ye += after
        xe += after
        ys *= l.stride[0]
        xs *= l.stride[0]
        ye *= l.stride[0]
        xe *= l.stride[0]
        l = l.input_layer
    return ys, xs, ye, xe

import glob
import collections


candidates = collections.defaultdict(list)
for content in glob.glob(doodle.args.content):
    image, mask = generator.load_images('content', content, scale=1.0)

    feature = generator.model.prepare_image(image)
    for layer, encoder in reversed(list(zip(doodle.args.layers, generator.encoders))):
        feature = encoder(feature, mask)
        
        x = feature.reshape(feature.shape[:2]+(-1,))[:,:-3,:]
        # x = (x - x.mean(axis=(0,2), keepdims=True)) # / x.std(axis=(0,2), keepdims=True)
        covariance = np.tensordot(x, x, axes=([2], [2])).mean(axis=(0,2)) / x.shape[2]
        np.fill_diagonal(covariance, 0.0)
        # print(covariance.shape, covariance.min(), covariance.max())

        # subplot.imshow(covariance, interpolation='nearest')

        for i in range(feature.shape[1]):
            w = feature[:,i:i+1,:,:]
            for idx in np.argsort(w.flatten())[-15:]:
                _, _, y, x = np.unravel_index(idx, w.shape)
                # print('coords', y, x, 'value', )
                a, b, c, d = calculate_patch_coordinates(generator.model.network['enc%i_1'%layer], y, x)
                img = np.copy(image[max(0,a):min(image.shape[0],c), max(0, b):min(image.shape[1],d)])
                candidates[i].append((img, w.flatten()[idx])) 

        # _, _, y, x = np.unravel_index(feature[0,0,:,:].argmax(), feature.shape)
        # print(y, x, calculate_patch_coordinates('enc%i_1'%layer, y, x))

        # subplot.set_title('Layer {}'.format(layer))

        # subplot.violinplot([feature[:,i,:,:].flatten() for i in range(feature.shape[1])], showmeans=False, showmedians=True)

        # x = np.arange(0, feature.shape[1], 1)
        # y = [feature.min(axis=(0,2,3)), feature.mean(axis=(0,2,3)), feature.max(axis=(0,2,3))]
        # for j in y:
        #     plt.errorbar(x, j)

fig, axes = plt.subplots(3, 5, figsize=(10, 6), subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.3, wspace=0.05)
# if not hasattr(axes, 'flat'): axes.flat = [plt]

for i, c in candidates.items():
    c.sort(key=lambda x: x[1])
    for (img, _), subplot in zip(c[-15:], axes.flat):
        subplot.imshow(img, interpolation='nearest')
    plt.savefig('channel_{}.png'.format(i))

# plt.show()
# print(i, c[0][1], c[-1][1])
