#!/usr/bin/env python3
#
# Neural Doodle!
# Copyright (c) 2016, Alex J. Champandard.
#
# Research and Development sponsored by the nucl.ai Conference!
#   http://events.nucl.ai/
#   July 18-20, 2016 in Vienna/Austria.
#

import os
import sys
import bz2
import math
import time
import pickle
import argparse
import itertools
import collections


# Configure all options first so we can custom load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument

add_arg('--content',        default=None, type=str,         help='Content image path as optimization target.')
add_arg('--content-weight', default=10.0, type=float,       help='Weight of content relative to style.')
add_arg('--content-layers', default='4_2', type=str,        help='The layer with which to match content.')
add_arg('--style',          default=None, type=str,         help='Style image path to extract patches.')
add_arg('--style-weight',   default=25.0, type=float,       help='Weight of style relative to content.')
add_arg('--style-layers',   default='3_1,4_1', type=str,    help='The layers to match style patches.')
add_arg('--semantic-ext',   default='_sem.png', type=str,   help='File extension for the semantic maps.')
add_arg('--semantic-weight', default=10.0, type=float,      help='Global weight of semantics vs. features.')
add_arg('--output',         default='output.png', type=str, help='Output image path to save once done.')
add_arg('--output-size',    default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--phases',         default=3, type=int,            help='Number of image scales to process in phases.')
add_arg('--slices',         default=2, type=int,            help='Split patches up into this number of batches.')
add_arg('--cache',          default=0, type=int,            help='Whether to compute matches only once.')
add_arg('--smoothness',     default=1E+0, type=float,       help='Weight of image smoothing scheme.')
add_arg('--variety',        default=0.0, type=float,        help='Bias toward selecting diverse patches, e.g. 0.5.')
add_arg('--seed',           default='noise', type=str,      help='Seed image path, "noise" or "content".')
add_arg('--seed-range',     default='16:240', type=str,     help='Random colors chosen in range, e.g. 0:255.')
add_arg('--iterations',     default=100, type=int,          help='Number of iterations to run each resolution.')
add_arg('--device',         default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
add_arg('--print-every',    default=10, type=int,           help='How often to log statistics to stdout.')
add_arg('--save-every',     default=10, type=int,           help='How frequently to save PNG into `frames`.')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information a little better, plus looks cool!
class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[0;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'

def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    sys.exit(-1)

print('{}Neural Doodle for semantic style transfer.{}'.format(ansi.CYAN_B, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,'\
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.optimize, scipy.ndimage, scipy.misc
import PIL

# Numeric Computing (GPU)
import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer

print('{}  - Using device `{}` for processing the images.{}'.format(ansi.CYAN, theano.config.device, ansi.ENDC))


#----------------------------------------------------------------------------------------------------------------------
# Convolutional Neural Network
#----------------------------------------------------------------------------------------------------------------------
class Model(object):
    """Store all the data related to the neural network (aka. "model"). This is currently based on VGG19.
    """

    def __init__(self):
        self.pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((3,1,1))

        self.setup_model()
        self.load_data()

    def setup_model(self, input=None):
        """Use lasagne to create a network of convolution layers, first using VGG19 as the framework
        and then adding augmentations for Semantic Style Transfer.
        """
        net, self.channels = {}, {}

        # Primary network for the main image. These are convolution only, and stop at layer 4_2 (rest unused).
        net['img']     = input or InputLayer((None, 3, None, None))
        net['conv1_1'] = ConvLayer(net['img'],     64, 3, pad=1)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1']   = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
        net['conv2_1'] = ConvLayer(net['pool1'],   128, 3, pad=1)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2']   = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
        net['conv3_1'] = ConvLayer(net['pool2'],   256, 3, pad=1)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
        net['pool3']   = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
        net['conv4_1'] = ConvLayer(net['pool3'],   512, 3, pad=1)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
        net['pool4']   = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
        net['conv5_1'] = ConvLayer(net['pool4'],   512, 3, pad=1)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
        net['main']    = net['conv5_4']

        # Auxiliary network for the semantic layers, and the nearest neighbors calculations.
        net['map'] = InputLayer((1, 1, None, None))
        for j, i in itertools.product(range(5), range(4)):
            if j < 2 and i > 1: continue
            suffix = '%i_%i' % (j+1, i+1)

            if i == 0:
                net['map%i'%(j+1)] = PoolLayer(net['map'], 2**j, mode='average_exc_pad')
            self.channels[suffix] = net['conv'+suffix].num_filters

            if args.semantic_weight > 0.0:
                net['sem'+suffix] = ConcatLayer([net['conv'+suffix], net['map%i'%(j+1)]])
            else:
                net['sem'+suffix] = net['conv'+suffix]

            net['dup'+suffix] = InputLayer(net['sem'+suffix].output_shape)
            net['nn'+suffix] = ConvLayer(net['dup'+suffix], 1, 3, b=None, pad=0, flip_filters=False)

        self.network = net

    def load_data(self):
        """Open the serialized parameters from a pre-trained network, and load them into the model created.
        """
        vgg19_file = os.path.join(os.path.dirname(__file__), 'vgg19_conv.pkl.bz2')
        if not os.path.exists(vgg19_file):
            error("Model file with pre-trained convolution layers not found. Download here...",
                  "https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2")

        data = pickle.load(bz2.open(vgg19_file, 'rb'))
        params = lasagne.layers.get_all_param_values(self.network['main'])
        lasagne.layers.set_all_param_values(self.network['main'], data[:len(params)])

    def setup(self, layers):
        """Setup the inputs and outputs, knowing the layers that are required by the optimization algorithm.
        """
        self.tensor_img = T.tensor4()
        self.tensor_map = T.tensor4()
        tensor_inputs = {self.network['img']: self.tensor_img, self.network['map']: self.tensor_map}
        outputs = lasagne.layers.get_output([self.network[l] for l in layers], tensor_inputs)
        self.tensor_outputs = {k: v for k, v in zip(layers, outputs)}

    def get_outputs(self, type, layers):
        """Fetch the output tensors for the network layers.
        """
        return [self.tensor_outputs[type+l] for l in layers]

    def prepare_image(self, image):
        """Given an image loaded from disk, turn it into a representation compatible with the model.
        The format is (b,c,y,x) with batch=1 for a single image, channels=3 for RGB, and y,x matching
        the resolution.
        """
        image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)[::-1, :, :]
        image = image.astype(np.float32) - self.pixel_mean
        return image[np.newaxis]

    def finalize_image(self, image, resolution):
        """Based on the output of the neural network, convert it into an image format that can be saved
        to disk -- shuffling dimensions as appropriate.
        """
        image = np.swapaxes(np.swapaxes(image[::-1], 0, 1), 1, 2)
        image = np.clip(image, 0, 255).astype('uint8')
        return scipy.misc.imresize(image, resolution, interp='bicubic')


#----------------------------------------------------------------------------------------------------------------------
# Semantic Style Transfer
#----------------------------------------------------------------------------------------------------------------------
class NeuralGenerator(object):
    """This is the main part of the application that generates an image using optimization and LBFGS.
    The images will be processed at increasing resolutions in the run() method.
    """

    def __init__(self):
        """Constructor sets up global variables, loads and validates files, then builds the model.
        """
        self.start_time = time.time()
        self.style_cache = {}
        self.style_layers = args.style_layers.split(',')
        self.content_layers = args.content_layers.split(',')
        self.used_layers = self.style_layers + self.content_layers

        # Prepare file output and load files specified as input.
        if args.save_every is not None:
            os.makedirs('frames', exist_ok=True)
        if args.output is not None and os.path.isfile(args.output):
            os.remove(args.output)

        print(ansi.CYAN, end='')
        target = args.content or args.output
        self.content_img_original, self.content_map_original = self.load_images('content', target)
        self.style_img_original, self.style_map_original = self.load_images('style', args.style)

        if self.content_map_original is None and self.content_img_original is None:
            print("  - No content files found; result depends on seed only.")
        print(ansi.ENDC, end='')

        # Display some useful errors if the user's input can't be undrestood.
        if self.style_img_original is None:
            error("Couldn't find style image as expected.",
                  "  - Try making sure `{}` exists and is a valid image.".format(args.style))

        if self.content_map_original is not None and self.style_map_original is None:
            basename, _ = os.path.splitext(args.style)
            error("Expecting a semantic map for the input style image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if self.style_map_original is not None and self.content_map_original is None:
            basename, _ = os.path.splitext(target)
            error("Expecting a semantic map for the input content image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if self.content_map_original is None:
            if self.content_img_original is None and args.output_size:
                shape = tuple([int(i) for i in args.output_size.split('x')])
            else:
                shape = self.style_img_original.shape[:2]

            self.content_map_original = np.zeros(shape+(3,))
            args.semantic_weight = 0.0

        if self.style_map_original is None:
            self.style_map_original = np.zeros(self.style_img_original.shape[:2]+(3,))
            args.semantic_weight = 0.0

        if self.content_img_original is None:
            self.content_img_original = np.zeros(self.content_map_original.shape[:2]+(3,))
            args.content_weight = 0.0

        if self.content_map_original.shape[2] != self.style_map_original.shape[2]:
            error("Mismatch in number of channels for style and content semantic map.",
                  "  - Make sure both images are RGB, RGBA, or L.")

        # Finalize the parameters based on what we loaded, then create the model.
        args.semantic_weight = math.sqrt(9.0 / args.semantic_weight) if args.semantic_weight else 0.0
        self.model = Model()


    #------------------------------------------------------------------------------------------------------------------
    # Helper Functions
    #------------------------------------------------------------------------------------------------------------------

    def load_images(self, name, filename):
        """If the image and map files exist, load them. Otherwise they'll be set to default values later.
        """
        basename, _ = os.path.splitext(filename)
        mapname = basename + args.semantic_ext
        img = scipy.ndimage.imread(filename, mode='RGB') if os.path.exists(filename) else None
        map = scipy.ndimage.imread(mapname) if os.path.exists(mapname) and args.semantic_weight > 0.0 else None

        if img is not None: print('  - Loading `{}` for {} data.'.format(filename, name))
        if map is not None: print('  - Adding `{}` as semantic map.'.format(mapname))

        if img is not None and map is not None and img.shape[:2] != map.shape[:2]:
            error("The {} image and its semantic map have different resolutions. Either:".format(name),
                  "  - Resize {} to {}, or\n  - Resize {} to {}."\
                  .format(filename, map.shape[1::-1], mapname, img.shape[1::-1]))
        return img, map

    def compile(self, arguments, function):
        """Build a Theano function that will run the specified expression on the GPU.
        """
        return theano.function(list(arguments), function, on_unused_input='ignore')

    def compute_norms(self, backend, layer, array):
        ni = backend.sqrt(backend.sum(array[:,:self.model.channels[layer]] ** 2.0, axis=(1,), keepdims=True))
        ns = backend.sqrt(backend.sum(array[:,self.model.channels[layer]:] ** 2.0, axis=(1,), keepdims=True))
        return [ni] + [ns]

    def normalize_components(self, layer, array, norms):
        if args.style_weight > 0.0:
            array[:,:self.model.channels[layer]] /= (norms[0] * 3.0)
        if args.semantic_weight > 0.0:
            array[:,self.model.channels[layer]:] /= (norms[1] * args.semantic_weight)


    #------------------------------------------------------------------------------------------------------------------
    # Initialization & Setup
    #------------------------------------------------------------------------------------------------------------------

    def rescale_image(self, img, scale):
        """Re-implementing skimage.transform.scale without the extra dependency. Saves a lot of space and hassle!
        """
        output = scipy.misc.toimage(img, cmin=0.0, cmax=255)
        output.thumbnail((int(output.size[0]*scale), int(output.size[1]*scale)), PIL.Image.ANTIALIAS)
        return np.asarray(output)

    def prepare_content(self, scale=1.0):
        """Called each phase of the optimization, rescale the original content image and its map to use as inputs.
        """
        content_img = self.rescale_image(self.content_img_original, scale)
        self.content_img = self.model.prepare_image(content_img)

        content_map = self.rescale_image(self.content_map_original, scale)
        self.content_map = content_map.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

    def prepare_style(self, scale=1.0):
        """Called each phase of the optimization, process the style image according to the scale, then run it
        through the model to extract intermediate outputs (e.g. sem4_1) and turn them into patches.
        """
        style_img = self.rescale_image(self.style_img_original, scale)
        self.style_img = self.model.prepare_image(style_img)

        style_map = self.rescale_image(self.style_map_original, scale)
        self.style_map = style_map.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

        # Compile a function to run on the GPU to extract patches for all layers at once.
        layer_outputs = zip(self.style_layers, self.model.get_outputs('sem', self.style_layers))
        extractor = self.compile([self.model.tensor_img, self.model.tensor_map], self.do_extract_patches(layer_outputs))
        result = extractor(self.style_img, self.style_map)

        # Store all the style patches layer by layer, resized to match slice size and cast to 16-bit for size.
        self.style_data = {}
        for layer, *data in zip(self.style_layers, result[0::3], result[1::3], result[2::3]):
            patches = data[0]
            l = self.model.network['nn'+layer]
            l.num_filters = patches.shape[0] // args.slices
            self.style_data[layer] = [d[:l.num_filters*args.slices].astype(np.float16) for d in data]\
                                   + [np.zeros((patches.shape[0],), dtype=np.float16)]
            print('  - Style layer {}: {} patches in {:,}kb.'.format(layer, patches.shape, patches.size//1000))

    def prepare_optimization(self):
        """Optimization requires a function to compute the error (aka. loss) which is done in multiple components.
        Here we compile a function to run on the GPU that returns all components separately.
        """

        # Feed-forward calculation only, returns the result of the convolution post-activation
        self.compute_features = self.compile([self.model.tensor_img, self.model.tensor_map],
                                             self.model.get_outputs('sem', self.style_layers))

        # Patch matching calculation that uses only pre-calculated features and a slice of the patches.

        self.matcher_tensors = {l: lasagne.utils.shared_empty(dim=4) for l in self.style_layers}
        self.matcher_history = {l: T.vector() for l in self.style_layers}
        self.matcher_inputs = {self.model.network['dup'+l]: self.matcher_tensors[l] for l in self.style_layers}
        nn_layers = [self.model.network['nn'+l] for l in self.style_layers]
        self.matcher_outputs = dict(zip(self.style_layers, lasagne.layers.get_output(nn_layers, self.matcher_inputs)))

        self.compute_matches = {l: self.compile([self.matcher_history[l]], self.do_match_patches(l))\
                                                for l in self.style_layers}

        self.tensor_matches = [T.tensor4() for l in self.style_layers]
        # Build a list of Theano expressions that, once summed up, compute the total error.
        self.losses = self.content_loss() + self.total_variation_loss() + self.style_loss()
        # Let Theano automatically compute the gradient of the error, used by LBFGS to update image pixels.
        grad = T.grad(sum([l[-1] for l in self.losses]), self.model.tensor_img)
        # Create a single function that returns the gradient and the individual errors components.
        self.compute_grad_and_losses = theano.function(
                                                [self.model.tensor_img, self.model.tensor_map] + self.tensor_matches,
                                                [grad] + [l[-1] for l in self.losses], on_unused_input='ignore')


    #------------------------------------------------------------------------------------------------------------------
    # Theano Computation
    #------------------------------------------------------------------------------------------------------------------

    def do_extract_patches(self, layers, size=3, stride=1):
        """This function builds a Theano expression that will get compiled an run on the GPU. It extracts 3x3 patches
        from the intermediate outputs in the model.
        """
        results = []
        for l, f in layers:
            # Use a Theano helper function to extract "neighbors" of specific size, seems a bit slower than doing
            # it manually but much simpler!
            patches = theano.tensor.nnet.neighbours.images2neibs(f, (size, size), (stride, stride), mode='valid')
            # Make sure the patches are in the shape required to insert them into the model as another layer.
            patches = patches.reshape((-1, patches.shape[0] // f.shape[1], size, size)).dimshuffle((1, 0, 2, 3))
            # Calculate the magnitude that we'll use for normalization at runtime, then store...
            results.extend([patches] + self.compute_norms(T, l, patches))
        return results

    def do_match_patches(self, layer):
        # Use node in the model to compute the result of the normalized cross-correlation, using results from the
        # nearest-neighbor layers called 'nn3_1' and 'nn4_1'.
        dist = self.matcher_outputs[layer]
        dist = dist.reshape((dist.shape[1], -1))
        # Compute the score of each patch, taking into account statistics from previous iteration. This equalizes
        # the chances of the patches being selected when the user requests more variety.
        offset = self.matcher_history[layer].reshape((-1, 1))
        scores = (dist - offset * args.variety)
        # Pick the best style patches for each patch in the current image, the result is an array of indices.
        # Also return the maximum value along both axis, used to compare slices and add patch variety.
        return [scores.argmax(axis=0), scores.max(axis=0), dist.max(axis=1)]


    #------------------------------------------------------------------------------------------------------------------
    # Error/Loss Functions
    #------------------------------------------------------------------------------------------------------------------

    def content_loss(self):
        """Return a list of Theano expressions for the error function, measuring how different the current image is
        from the reference content that was loaded.
        """

        content_loss = []
        if args.content_weight == 0.0:
            return content_loss

        # First extract all the features we need from the model, these results after convolution.
        extractor = theano.function([self.model.tensor_img], self.model.get_outputs('conv', self.content_layers))
        result = extractor(self.content_img)

        # Build a list of loss components that compute the mean squared error by comparing current result to desired.
        for l, ref in zip(self.content_layers, result):
            layer = self.model.tensor_outputs['conv'+l]
            loss = T.mean((layer - ref) ** 2.0)
            content_loss.append(('content', l, args.content_weight * loss))
            print('  - Content layer conv{}: {} features in {:,}kb.'.format(l, ref.shape[1], ref.size//1000))
        return content_loss

    def style_loss(self):
        """Returns a list of loss components as Theano expressions. Finds the best style patch for each patch in the
        current image using normalized cross-correlation, then computes the mean squared error for all patches.
        """
        style_loss = []
        if args.style_weight == 0.0:
            return style_loss

        # Extract the patches from the current image, as well as their magnitude.
        result = self.do_extract_patches(zip(self.style_layers, self.model.get_outputs('conv', self.style_layers)))

        # Multiple style layers are optimized separately, usually conv3_1 and conv4_1 — semantic data not used here.
        for l, matches, patches in zip(self.style_layers, self.tensor_matches, result[0::3]):
            # Compute the mean squared error between the current patch and the best matching style patch.
            # Ignore the last channels (from semantic map) so errors returned are indicative of image only.
            loss = T.mean((patches - matches[:,:self.model.channels[l]]) ** 2.0)
            style_loss.append(('style', l, args.style_weight * loss))
        return style_loss

    def total_variation_loss(self):
        """Return a loss component as Theano expression for the smoothness prior on the result image.
        """
        x = self.model.tensor_img
        loss = (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).mean()
        return [('smooth', 'img', args.smoothness * loss)]


    #------------------------------------------------------------------------------------------------------------------
    # Optimization Loop
    #------------------------------------------------------------------------------------------------------------------

    def iterate_batches(self, *arrays, batch_size):
        """Break down the data in arrays batch by batch and return them as a generator.
        """
        total_size = arrays[0].shape[0]
        indices = np.arange(total_size)
        for index in range(0, total_size, batch_size):
            excerpt = indices[index:index + batch_size]
            yield excerpt, [a[excerpt] for a in arrays]

    def evaluate_slices(self, f, l):
        if args.cache and l in self.style_cache:
            return self.style_cache[l]

        layer, data = self.model.network['nn'+l], self.style_data[l]
        history = data[-1]

        best_idx, best_val = None, 0.0
        for idx, (bp, bi, bs, bh) in self.iterate_batches(*data, batch_size=layer.num_filters):
            weights = bp.astype(np.float32)
            self.normalize_components(l, weights, (bi, bs))
            layer.W.set_value(weights)

            cur_idx, cur_val, cur_match = self.compute_matches[l](history[idx])
            if best_idx is None:
                best_idx, best_val = cur_idx, cur_val
            else:
                i = np.where(cur_val > best_val)
                best_idx[i] = idx[cur_idx[i]]
                best_val[i] = cur_val[i]

            history[idx] = cur_match

        if args.cache:
            self.style_cache[l] = best_idx
        return best_idx

    def evaluate(self, Xn):
        """Callback for the L-BFGS optimization that computes the loss and gradients on the GPU.
        """
        # Adjust the representation to be compatible with the model before computing results.
        current_img = Xn.reshape(self.content_img.shape).astype(np.float32) - self.model.pixel_mean
        current_features = self.compute_features(current_img, self.content_map)

        # Iterate through each of the style layers one by one, computing best matches.
        current_best = []
        for l, f in zip(self.style_layers, current_features):
            self.normalize_components(l, f, self.compute_norms(np, l, f))
            self.matcher_tensors[l].set_value(f)

            # Compute best matching patches this style layer, going through all slices.
            warmup = bool(args.variety > 0.0 and self.iteration == 0)
            for _ in range(2 if warmup else 1):
                best_idx = self.evaluate_slices(f, l)

            patches = self.style_data[l][0]
            current_best.append(patches[best_idx].astype(np.float32))

        grads, *losses = self.compute_grad_and_losses(current_img, self.content_map, *current_best)
        if np.isnan(grads).any():
            raise OverflowError("Optimization diverged; try using a different device or parameters.")

        # Use magnitude of gradients as an estimate for overall quality.
        self.error = self.error * 0.9 + 0.1 * min(np.abs(grads).max(), 255.0)
        loss = sum(losses)

        # Dump the image to disk if requested by the user.
        if args.save_every and self.frame % args.save_every == 0:
            frame = Xn.reshape(self.content_img.shape[1:])
            resolution = self.content_img_original.shape
            image = scipy.misc.toimage(self.model.finalize_image(frame, resolution), cmin=0, cmax=255)
            image.save('frames/%04d.png'%self.frame)

        # Print more information to the console every few iterations.
        if args.print_every and self.frame % args.print_every == 0:
            print('{:>3}   {}loss{} {:8.2e} '.format(self.frame, ansi.BOLD, ansi.ENDC, loss / 1000.0), end='')
            category = ''
            for v, l in zip(losses, self.losses):
                if l[0] == 'smooth':
                    continue
                if l[0] != category:
                    print('  {}{}{}'.format(ansi.BOLD, l[0], ansi.ENDC), end='')
                    category = l[0]
                print(' {}{}{} {:8.2e} '.format(ansi.BOLD, l[1], ansi.ENDC, v / 1000.0), end='')

            current_time = time.time()
            quality = 100.0 - 100.0 * np.sqrt(self.error / 255.0)
            print('  {}quality{} {: >4.1f}% '.format(ansi.BOLD, ansi.ENDC, quality), end='')
            print('  {}time{} {:3.1f}s '.format(ansi.BOLD, ansi.ENDC, current_time - self.iter_time), flush=True)
            self.iter_time = current_time

        # Update counters and timers.
        self.frame += 1
        self.iteration += 1

        # Return the data in the right format for L-BFGS.
        return loss, np.array(grads).flatten().astype(np.float64)

    def run(self):
        """The main entry point for the application, runs through multiple phases at increasing resolutions.
        """
        self.frame, Xn = 0, None
        for i in range(args.phases):
            self.error = 255.0
            scale = 1.0 / 2.0 ** (args.phases - 1 - i)

            shape = self.content_img_original.shape
            print('\n{}Phase #{}: resolution {}x{}  scale {}{}'\
                    .format(ansi.BLUE_B, i, int(shape[1]*scale), int(shape[0]*scale), scale, ansi.BLUE))
            if int(shape[1]*scale) < 24:
                error("Resolutions below 24x24 are not possible.",
                      "  - Change the provided combination of `--output-size` and `--phases` to keep the resolution above 24x24.")

            # Precompute all necessary data for the various layers, put patches in place into augmented network.
            self.model.setup(layers=['sem'+l for l in self.style_layers] + ['conv'+l for l in self.content_layers])
            self.prepare_content(scale)
            self.prepare_style(scale)

            # Now setup the model with the new data, ready for the optimization loop.
            self.model.setup(layers=['sem'+l for l in self.style_layers] + ['conv'+l for l in self.used_layers])
            self.prepare_optimization()
            print('{}'.format(ansi.ENDC))

            # Setup the seed for the optimization as specified by the user.
            shape = self.content_img.shape[2:]
            if args.seed == 'content':
                Xn = self.content_img[0] + self.model.pixel_mean
            if args.seed == 'noise':
                bounds = [int(i) for i in args.seed_range.split(':')]
                Xn = np.random.uniform(bounds[0], bounds[1], shape + (3,)).astype(np.float32)
            if args.seed == 'previous':
                Xn = scipy.misc.imresize(Xn[0], shape, interp='bicubic')
                Xn = Xn.transpose((2, 0, 1))[np.newaxis]
            if os.path.exists(args.seed):
                seed_image = scipy.ndimage.imread(args.seed, mode='RGB')
                seed_image = scipy.misc.imresize(seed_image, shape, interp='bicubic')
                self.seed_image = self.model.prepare_image(seed_image)
                Xn = self.seed_image[0] + self.model.pixel_mean
            if Xn is None:
                error("Seed for optimization was not found. You can either...",
                      "  - Set the `--seed` to `content` or `noise`.", "  - Specify `--seed` as a valid filename.")

            # Optimization algorithm needs min and max bounds to prevent divergence.
            data_bounds = np.zeros((np.product(Xn.shape), 2), dtype=np.float64)
            data_bounds[:] = (0.0, 255.0)

            self.iter_time, self.iteration, interrupt = time.time(), 0, False
            try:
                Xn, Vn, info = scipy.optimize.fmin_l_bfgs_b(
                                self.evaluate,
                                Xn.astype(np.float64).flatten(),
                                bounds=data_bounds,
                                factr=0.0, pgtol=0.0,            # Disable automatic termination, set low threshold.
                                m=5,                             # Maximum correlations kept in memory by algorithm.
                                maxfun=args.iterations-1,        # Limit number of calls to evaluate().
                                iprint=-1)                       # Handle our own logging of information.
            except OverflowError:
                error("The optimization diverged and NaNs were encountered.",
                      "  - Try using a different `--device` or change the parameters.",
                      "  - Make sure libraries are updated to work around platform bugs.")
            except KeyboardInterrupt:
                interrupt = True

            args.seed = 'previous'
            resolution = self.content_img.shape
            Xn = Xn.reshape(resolution)

            output = self.model.finalize_image(Xn[0], self.content_img_original.shape)
            scipy.misc.toimage(output, cmin=0, cmax=255).save(args.output)
            if interrupt: break

        status = "finished in" if not interrupt else "interrupted at"
        print('\n{}Optimization {} {:3.1f}s, average pixel error {:3.1f}!{}\n'\
              .format(ansi.CYAN, status, time.time() - self.start_time, self.error, ansi.ENDC))


if __name__ == "__main__":
    generator = NeuralGenerator()
    generator.run()
