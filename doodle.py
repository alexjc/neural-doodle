#
# Copyright (c) 2016, Alex J. Champandard.
# 
# Research and development sponsored by the nucl.ai Conference!
#   http://events.nucl.ai/
#   July 18-20, 2016 in Vienna/Austria.
#

import os
import sys
import bz2
import pickle
import argparse
import warnings
from PIL import Image


# Configure all options first so we can custom load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument

add_arg('--content',        default=None, type=str,         help='Content image path as optimization target.')
add_arg('--content-weight', default=10.0, type=float,       help='Weight of content relative to style.')
add_arg('--content-layers', default='4_2', type=str,        help='The layer with which to match content.')
add_arg('--style',          required=True, type=str,        help='Style image path to extract patches.')
add_arg('--style-weight',   default=25.0, type=float,       help='Weight of style relative to content.')
add_arg('--style-layers',   default='3_1,4_1', type=str,    help='The layers to match style patches.')
add_arg('--semantic-ext',   default='_sem.png', type=str,   help='File extension for the semantic maps.')
add_arg('--semantic-weight', default=10.0, type=float,      help='Global weight of semantics vs. features.')
add_arg('--output',         default='output.png', type=str, help='Output image path to save once done.')
add_arg('--resolutions',    default=3, type=int,            help='Number of image scales to process.')
add_arg('--smoothness',     default=1E+0, type=float,       help='Weight of image smoothing scheme.')
add_arg('--seed',           default='noise', type=str,      help='Seed image path, "noise" or "content".')
add_arg('--iterations',     default=100, type=int,          help='Number of iterations to run each resolution.')
add_arg('--device',         default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
add_arg('--safe-mode',      default=0, action='store_true', help='Use conservative Theano setting to avoid problems.')
add_arg('--print-every',    default=10, type=int,           help='How often to log statistics to stdout.')
add_arg('--save-every',     default=10, type=int,           help='How frequently to save PNG into `frames`.')
add_arg('--size',     default=512, type=int,           help='Resolution of final image. Valid range is from 320 and above.')
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

print('{}Neural Doodle for semantic style transfer.{}'.format(ansi.CYAN_B, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
extra_flags = ',optimizer=fast_compile' if args.safe_mode else ''
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,'\
                                      'print_active_device=False{}'.format(args.device, extra_flags))

# Scientific Libraries
import numpy as np
import scipy.optimize
import skimage.transform

# Numeric Computing (GPU)
import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours

# Deep Learning Framework
with warnings.catch_warnings():
    # suppress: "downsample module has been moved to the pool module." (Temporary workaround.)
    warnings.simplefilter("ignore")
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

    def setup_model(self):
        """Use lasagne to create a network of convolution layers, first using VGG19 as the framework
        and then adding augmentations for Semantic Style Transfer.
        """

        net = {}

        # First network for the main image. These are convolution only, and stop at layer 4_2 (rest unused).
        net['img']     = InputLayer((1, 3, None, None))
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
        net['main']    = net['conv4_2']

        # Second network for the semantic layers.  This dynamically downsamples the map and concatenates it.
        net['map'] = InputLayer((1, 3, None, None))
        net['map1_1'] = PoolLayer(net['map'], 2, mode='average_exc_pad')
        net['map2_1'] = PoolLayer(net['map'], 2, mode='average_exc_pad')
        net['map3_1'] = PoolLayer(net['map'], 4, mode='average_exc_pad')
        net['map4_1'] = PoolLayer(net['map'], 8, mode='average_exc_pad')

        # Third network for the nearest neighbors; it's a default size for now, updated once we know more.
        net['nn1_1'] = ConvLayer(net['conv1_1'], 1, 3, b=None, pad=0)
        net['nn2_1'] = ConvLayer(net['conv2_1'], 1, 3, b=None, pad=0)
        net['nn3_1'] = ConvLayer(net['conv3_1'], 1, 3, b=None, pad=0)
        net['nn4_1'] = ConvLayer(net['conv4_1'], 1, 3, b=None, pad=0)

        net['mm1_1'] = ConvLayer(net['map1_1'], 1, 3, b=None, pad=0)
        net['mm2_1'] = ConvLayer(net['map2_1'], 1, 3, b=None, pad=0)
        net['mm3_1'] = ConvLayer(net['map3_1'], 1, 3, b=None, pad=0)
        net['mm4_1'] = ConvLayer(net['map4_1'], 1, 3, b=None, pad=0)

        self.network = net

    def load_data(self):
        """Open the serialized parameters from a pre-trained network, and load them into the model created.
        """

        if not os.path.exists('vgg19_conv.pkl.bz2'):
            print("{}ERROR: Model file with pre-trained convolution layers not found. Download here:{}\n"\
                  "https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2{}\n"\
            .format(ansi.RED_B, ansi.RED, ansi.ENDC))
            sys.exit(-1)

        data = pickle.load(bz2.open('vgg19_conv.pkl.bz2', 'rb'))
        params = lasagne.layers.get_all_param_values(self.network['main'])
        lasagne.layers.set_all_param_values(self.network['main'], data[:len(params)])

    def setup(self, layers):
        """Setup the inputs and outputs, knowing the layers that are required by the optimization algorithm.
        """

        self.tensor_img = T.tensor4()
        self.tensor_map = T.tensor4()
        self.tensor_inputs = {self.network['img']: self.tensor_img, self.network['map']: self.tensor_map}

        outputs = lasagne.layers.get_output([self.network[l] for l in layers], self.tensor_inputs)
        self.tensor_outputs = {k: v for k, v in zip(layers, outputs)}

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

        image = image.reshape(resolution)[::-1]
        image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
        return np.clip(image, 0, 255).astype('uint8')


#----------------------------------------------------------------------------------------------------------------------
# Semantic Style Transfer
#----------------------------------------------------------------------------------------------------------------------
class NeuralGenerator(object):
    """This is the main part of the application that generates an image using optimization and LBFGS.
    The images will be processed at increasing resolutions in the run() method.
    """

    def __init__(self):
        self.model = Model()

        self.style_layers = args.style_layers.split(',')
        self.content_layers = args.content_layers.split(',')

        if args.save_every is not None:
            os.makedirs('frames', exist_ok=True)
        if args.output is not None and os.path.isfile(args.output):
            os.remove(args.output)

        print(ansi.CYAN, end='')
        target = args.content or args.output
        self.content_img_original, self.content_map_original = self.load_images('content', target)
        self.style_img_original, self.style_map_original = self.load_images('style', args.style)
        print(ansi.ENDC, end='')

        if self.content_map_original is None and self.content_img_original is None:
            basename, _ = os.path.splitext(target)
            print("\n{}ERROR: Couldn't find either the target image or a valid semantic map.\n"\
                  "{}  - Try creating the file `{}_sem.png` with your annotations.{}\n".format(ansi.RED_B, ansi.RED, basename, ansi.ENDC))
            sys.exit(-1)

        if self.style_img_original is None:
            print("\n{}ERROR: Couldn't find style image as expected.\n"\
                  "{}  - Try making sure `{}` exists and is a valid image.{}\n".format(ansi.RED_B, ansi.RED, args.style, ansi.ENDC))
            sys.exit(-1)

        if self.content_map_original is not None and self.style_map_original is None:
            basename, _ = os.path.splitext(args.style)
            print("\n{}ERROR: Expecting a semantic map for the input style image too.\n"\
                  "{}  - Try creating the file `{}_sem.png` with your annotations.{}\n".format(ansi.RED_B, ansi.RED, basename, ansi.ENDC))
            sys.exit(-1)

        if self.style_map_original is not None and self.content_map_original is None:
            basename, _ = os.path.splitext(target)
            print("\n{}ERROR: Expecting a semantic map for the input content image too.\n"\
                  "{}  - Try creating the file `{}_sem.png` with your annotations.{}\n".format(ansi.RED_B, ansi.RED, basename, ansi.ENDC))
            sys.exit(-1)

        if self.content_map_original is None:
            self.content_map_original = np.zeros(self.content_img_original.shape[:2]+(1,))
            args.semantic_weight = 0.0

        if self.style_map_original is None:
            self.style_map_original = np.zeros(self.style_img_original.shape[:2]+(1,))
            args.semantic_weight = 0.0

        if self.content_img_original is None:
            self.content_img_original = np.zeros(self.content_map_original.shape[:2]+(3,))
            args.content_weight = 0.0

        if self.content_map_original.shape[2] != self.style_map_original.shape[2]:
            print("\n{}ERROR: Mismatch in number of channels for style and content semantic map.\n"\
                  "{}  - Make sure both images are RGB or RGBA.{}\n".format(ansi.RED_B, ansi.RED, ansi.ENDC))
            sys.exit(-1)

    def load_images(self, name, filename):
        """If the image and map files exist, load them. Otherwise they'll be set to default values later.
        """
        basename, _ = os.path.splitext(filename)
        mapname = basename + args.semantic_ext
        
        if os.path.exists(filename):
            resizedname = self.image_resize_byname(filename)
            img = scipy.ndimage.imread(resizedname, mode='RGB')
        else: img = None
        
        if os.path.exists(mapname):
            resizedname = self.image_resize_byname(mapname)
            map = scipy.ndimage.imread(resizedname)
        else: map = None
        
        if img is not None: print('  - Loading {} image data from {}.'.format(name, filename))
        if map is not None: print('  - Loading {} semantic map from {}.'.format(name, mapname))
        
        if img is not None and map is not None and img.shape[:2] != map.shape[:2]:
            print("\n{}ERROR: The {} image and its semantic map have different resolutions. Either:\n"\
                  "{}  - Resize {} to {}, or\n  - Resize {} to {}.\n"\
                  .format(ansi.RED_B, name, ansi.RED, filename,map.shape[1::-1], mapname,img.shape[1::-1], ansi.ENDC))
            sys.exit(-1)

        return img, map

    def image_resize_byname(self, path):
        image = Image.open(path)
        (w,h) = image.size
        
        sizearg = args.size
        
        if (sizearg < 320): size = 320
        
        if (w >= h):
            if (w != sizearg):
                h = int(h * sizearg / w)
                w = sizearg
                print('  - Resizing {} image data to {}x{}.'.format(path, w, h))
                image = image.resize((w, h), Image.ANTIALIAS)
        else:
            if (h != sizearg):
                w = int(w * sizearg / h)
                h = sizearg
                print('  - Resizing {} image data to {}x{}.'.format(path, w, h))
                image = image.resize((w, h), Image.ANTIALIAS)
        
        tmppathname = '/tmp/' + os.path.basename(path)
    
        image.save(tmppathname)
        
        return tmppathname

    #------------------------------------------------------------------------------------------------------------------
    # Initialization & Setup
    #------------------------------------------------------------------------------------------------------------------

    def prepare_content(self, scale=1.0):
        """Called each phase of the optimization, rescale the original content image and its map to use as inputs.
        """
        content_image = skimage.transform.rescale(self.content_img_original, scale) * 255.0
        self.content_image = self.model.prepare_image(content_image)

        content_map = skimage.transform.rescale(self.content_map_original, scale) * 255.0
        self.content_map = content_map.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

    def prepare_style(self, scale=1.0):
        """Called each phase of the optimization, process the style image according to the scale, then run it
        through the model to extract intermediate outputs (e.g. sem4_1) and turn them into patches.
        """
        style_image = skimage.transform.rescale(self.style_img_original, scale) * 255.0
        self.style_image = self.model.prepare_image(style_image)

        style_map = skimage.transform.rescale(self.style_map_original, scale) * 255.0
        self.style_map = style_map.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

        # Workaround for Issue #8. Not clear what this is caused by, NaN seems to happen in convolution node
        # on some OSX installations. https://github.com/alexjc/neural-doodle/issues/8
        if args.safe_mode:
            from theano.compile.nanguardmode import NanGuardMode
            flags = {'mode': NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)}
        else:
            flags = {}

        # Compile a function to run on the GPU to extract patches for all layers at once.
        required_layers = ['conv'+l for l in self.style_layers] + ['map'+l for l in self.style_layers]
        extractor = theano.function(
                        [self.model.tensor_img, self.model.tensor_map],
                        self.extract_patches([self.model.tensor_outputs[l] for l in required_layers]),
                        **flags)
        result = extractor(self.style_image, self.style_map)

        # For each layer, build it from set of patches and their magnitude.
        def build(layer, prefix, name, patches, norms):
            l = self.model.network[prefix+layer]
            l.N = theano.shared(norms)
            l.W.set_value(patches)
            l.num_filters = patches.shape[0]
            print('  - {} layer {}: {} patches in {:,}kb.'.format(name, layer, patches.shape[0], patches.size//1000))

        result_nn = result[:len(self.style_layers)*2]
        for layer, *data in zip(self.style_layers, result_nn[::2], result_nn[1::2]):
            build(layer, 'nn', 'Style', *data)

        result_mm = result[len(self.style_layers)*2:]
        for layer, *data in zip(self.style_layers, result_mm[::2], result_mm[1::2]):
            build(layer, 'mm', 'Semantic', *data)


    def extract_patches(self, layers, size=3, stride=1):
        """This function builds a Theano expression that will get compiled an run on the GPU. It extracts 3x3 patches
        from the intermediate outputs in the model.
        """
        results = []
        for f in layers:
            # Use a Theano helper function to extract "neighbors" of specific size, seems a bit slower than doing
            # it manually but much simpler!
            patches = theano.tensor.nnet.neighbours.images2neibs(f, (size, size), (stride, stride), mode='valid')

            # Make sure the patches are in the shape required to insert them into the model as another layer.
            patches = patches.reshape((-1, patches.shape[0] // f.shape[1], size, size)).dimshuffle((1, 0, 2, 3))

            # Calcualte the magnitude that we'll use for normalization at runtime, then store...
            norm = T.sqrt(T.sum(patches ** 2.0, axis=(1,2,3), keepdims=True))
            results.extend([patches[:,:,::-1,::-1], norm])
        return results

    def prepare_optimization(self):
        """Optimization requires a function to compute the error (aka. loss) which is done in multiple components.
        Here we compile a function to run on the GPU that returns all components separately.
        """
        
        # Build a list of Theano expressions that, once summed up, compute the total error.
        self.losses = self.content_loss() + self.style_loss() + self.total_variation_loss()
        # Let Theano automatically compute the gradient of the error, used by LBFGS to update image pixels.
        grad = T.grad(sum([l[-1] for l in self.losses]), self.model.tensor_img)
        # Create a single function that returns the gradient and the individual errors components.
        self.compute_grad_and_losses = theano.function([self.model.tensor_img, self.model.tensor_map],
                                                       [grad] + [l[-1] for l in self.losses], on_unused_input='ignore')


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
        extractor = theano.function([self.model.tensor_img],
                                    [self.model.tensor_outputs['conv'+l] for l in self.content_layers])
        result = extractor(self.content_image)

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
        result = self.extract_patches([self.model.tensor_outputs['conv'+l] for l in self.style_layers]
                                    + [self.model.tensor_outputs['map'+l] for l in self.style_layers])

        result_nn = result[:len(self.style_layers)*2]
        result_mm = result[len(self.style_layers)*2:]
        # Multiple style layers are optimized separately, usually sem3_1 and sem4_1.
        for l, patches, norms, sem_norms in zip(self.style_layers, result_nn[::2], result_nn[1::2], result_mm[1::2]):
            # Compute the result of the normalized cross-correlation, using results from the nearest-neighbor
            # layers called 'nn3_1' and 'nn4_1' (for example).
            layer = self.model.network['nn'+l]
            dist = self.model.tensor_outputs['nn'+l]
            dist = dist.reshape((dist.shape[1], -1)) / norms.reshape((1,-1)) / layer.N.reshape((-1,1))

            sem_layer = self.model.network['mm'+l]
            sem = self.model.tensor_outputs['mm'+l]
            sem = sem.reshape((sem.shape[1], -1)) / sem_norms.reshape((1,-1)) / sem_layer.N.reshape((-1,1))

            # Pick the best style patches for each patch in the current image, the result is an array of indices.
            best = (dist + args.semantic_weight * sem).argmax(axis=0)

            # Compute the mean squared error between the current patch and the best matching style patch.
            # Ignore the last channels (from semantic map) so errors returned are indicative of image only.
            loss = T.mean((patches - layer.W[best]) ** 2.0)
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

    def evaluate(self, Xn):
        """Callback for the L-BFGS optimization that computes the loss and gradients on the GPU.
        """
        
        # Adjust the representation to be compatible with the model before computing results.
        current_img = Xn.reshape(self.content_image.shape).astype(np.float32) - self.model.pixel_mean
        grads, *losses = self.compute_grad_and_losses(current_img, self.content_map)

        if np.isnan(grads).any():
            raise OverflowError("Optimization diverged; try using different device or parameters.")

        # Use gradients as an estimate for overall quality.
        self.error = self.error * 0.9 + 0.1 * np.abs(grads).max()
        loss = sum(losses)

        # Dump the image to disk if requested by the user.
        if args.save_every and self.frame % args.save_every == 0:
            resolution = self.content_image.shape[1:]
            image = scipy.misc.toimage(self.model.finalize_image(Xn, resolution), cmin=0, cmax=255)
            image.save('frames/%04d.png'%self.frame)

        # Print more information to the console every few iterations.
        if args.print_every and self.frame % args.print_every == 0:
            print('{:>3}   {}error{} {:8.2e} '.format(self.frame, ansi.BOLD, ansi.ENDC, loss / 1000.0), end='')
            category = ''
            for v, l in zip(losses, self.losses):
                if l[0] == 'smooth':
                    continue
                if l[0] != category:
                    print('  {}{}{}'.format(ansi.BOLD, l[0], ansi.ENDC), end='')
                    category = l[0]
                print(' {}{}{} {:8.2e} '.format(ansi.BOLD, l[1], ansi.ENDC, v / 1000.0), end='')

            quality = 100.0 - 100.0 * np.sqrt(self.error / 255.0)
            print('  {}quality{} {:3.1f}% '.format(ansi.BOLD, ansi.ENDC, quality, flush=True))

        # Return the data in the right format for L-BFGS.
        self.frame += 1
        return loss, np.array(grads).flatten().astype(np.float64)

    def run(self):
        """The main entry point for the application, runs through multiple phases at increasing resolutions.
        """

        self.frame = 0
        for i in range(args.resolutions):
            self.error = 255.0
            scale = 1.0 / 2.0 ** (args.resolutions - 1 - i)

            shape = self.content_img_original.shape
            print('\n{}Phase #{}: resolution {}x{}  scale {}{}'\
                    .format(ansi.BLUE_B, i, int(shape[1]*scale), int(shape[0]*scale), scale, ansi.BLUE))

            # Precompute all necessary data for the various layers, put patches in place into augmented network.
            self.model.setup(layers=['conv'+l for l in self.style_layers] +
                                    ['map'+l for l in self.style_layers] +
                                    ['conv'+l for l in self.content_layers])
            self.prepare_content(scale)
            self.prepare_style(scale)

            # Now setup the model with the new data, ready for the optimization loop.
            self.model.setup(layers=['conv'+l for l in self.style_layers] +
                                    ['map'+l for l in self.style_layers] +
                                    ['nn'+l for l in self.style_layers] +
                                    ['mm'+l for l in self.style_layers] +
                                    ['conv'+l for l in self.content_layers])
            self.prepare_optimization()
            print('{}'.format(ansi.ENDC))

            # Setup the seed for the optimization as specified by the user.
            shape = self.content_image.shape[2:]
            if args.seed == 'content':
                Xn = self.content_image[0] + self.model.pixel_mean
            if args.seed == 'noise':
                Xn = np.random.uniform(32, 224, shape + (3,)).astype(np.float32)
            if args.seed == 'previous':
                Xn = scipy.misc.imresize(Xn[0], shape)
                Xn = Xn.transpose((2, 0, 1))[np.newaxis]

            # Optimization algorithm needs min and max bounds to prevent divergence.
            data_bounds = np.zeros((np.product(Xn.shape), 2), dtype=np.float64)
            data_bounds[:] = (0.0, 255.0)

            try:
                Xn, Vn, info = scipy.optimize.fmin_l_bfgs_b(
                                self.evaluate,
                                Xn.astype(np.float64).flatten(),
                                bounds=data_bounds,
                                factr=0.0, pgtol=0.0,            # Disable automatic termination, set low threshold.
                                m=4,                             # Maximum correlations kept in memory by algorithm. 
                                maxfun=args.iterations-1,        # Limit number of calls to evaluate().
                                iprint=-1)                       # Handle our own logging of information.
            except OverflowError:
                print("{}ERROR: The optimization diverged and NaNs were encountered.{}\n"\
                      "  - Try using a different `--device` or change the parameters.\n"\
                      "  - Experiment with `--safe-mode` to work around platform bugs.{}\n".format(ansi.RED_B, ansi.RED, ansi.ENDC))
                sys.exit(-1)

            args.seed = 'previous'
            resolution = self.content_image.shape
            Xn = Xn.reshape(resolution)
            scipy.misc.toimage(self.model.finalize_image(Xn, resolution[1:]), cmin=0, cmax=255).save(args.output)


if __name__ == "__main__":
    generator = NeuralGenerator()
    generator.run()
