Neural Doodle
=============

.. image:: docs/Workflow.gif

Use a deep neural network to borrow the skills of real artists and turn your two-bit doodles into masterpieces! This project is an implementation of `Semantic Style Transfer <http://arxiv.org/abs/1603.01768>`_ (Champandard, 2016), based on the `Neural Patches <http://arxiv.org/abs/1601.04589>`_ algorithm (Li, 2016). Read more about the motivation in this `in-depth article <https://nucl.ai/blog/neural-doodles/>`_ and watch this `workflow video <https://www.youtube.com/watch?v=fu2fzx4w3mI>`_ for inspiration.

The ``doodle.py`` script generates a new image by using one, two, three or four images as inputs depending what you're trying to do: the original style and its annotation, and a target content image (optional) with its annotation (a.k.a. your doodle). The algorithm extracts annotated patches from the style image, and incrementally transfers them over to the target image based on how closely they match.

**NOTE**: Making a ``#NeuralDoodle`` is a skill. The parameters in the script were adjusted to work well by default and with the examples below. For new images, you may need to adjust values and modify on your input data too. It takes practice, but you can reach almost photo-realistic results if you iterate! (`Ask for advice here or see examples <https://github.com/alexjc/neural-doodle/issues?q=label%3Aadvice>`_.)

1. `Examples & Usage <#1-examples--usage>`_
2. `Installation <#2-installation--setup>`_
3. `Troubleshooting <#3-troubleshooting-problems>`_
4. `Frequent Questions <#4-frequent-questions>`_

**IMPORTANT**: This project is possible thanks to the `nucl.ai Conference <http://events.nucl.ai/>`_ on Creative AI, **July 18-20**. Join us in **Vienna**!

|Python Version| |License Type| |Project Stars|

----

.. image:: docs/Landscape_example.png

1. Examples & Usage
===================

The main script is called ``doodle.py``, which you can run with Python 3.4+ (see setup below).  The ``--device`` argument that lets you specify which GPU or CPU to use. For the samples above, here are the performance results:

* **GPU Rendering** — Assuming you have CUDA setup and enough on-board RAM, the process should complete in 3 to 8 minutes, even with twice the iteration count.
* **CPU Rendering** — This will take hours and hours, even up to 12h on older hardware. To match quality it'd take twice the time. Do multiple runs in parallel!

The default is to use ``cpu``, if you have NVIDIA card setup with CUDA already try ``gpu0``. On the CPU, you can also set environment variable to ``OMP_NUM_THREADS=4``, but we've found the speed improvements to be minimal.

1.a) Image Analogy
------------------

The algorithm is built for style transfer, but can also generate image analogies that we call a ``#NeuralDoodle``; use the hashtag if you post your images!  Example files are included in the ``#/samples/`` folder. Execute with these commands:

.. code:: bash

    # Synthesize a coastline as if painted by Monet. This uses "*_sem.png" masks for both images.
    python3 doodle.py --style samples/Monet.jpg --output samples/Coastline.png \
                      --variety 0.0 0.2 --layers 4_1 3_1 --iterations 4 6

    # Generate a scene around a lake in the style of a Renoir painting.
    python3 doodle.py --style samples/Renoir.jpg --output samples/Landscape.png \
                      --variety 0.5 --layers 6_1 5_1 4_1 3_1 --iterations 4

Notice the Renoir results look a little better than the Monet. Some rotational variations of the source image could improve the quality of the arch outline in particular.


1.b) Style Transfer
-------------------

If you want to transfer the style given a source style with annotations, and a target content image with annotations, you can use the following command lines.  In all cases, the semantic map is loaded and used if it's found under the ``*_sem.png`` filename that matches the input file.

.. code:: bash

    # Synthesize a portrait of Seth Johnson like a Gogh portrait. This uses "*_sem.png" masks for both images.
    python3 doodle.py --content samples/Seth.jpg --style samples/Gogh.jpg \
                        --variety 0.2 0.1 --balance 0.85 1.0 --layers 4_1 3_1 --iterations 6

    # Generate what a photo of Vincent van Gogh would look like, using Seth's portrait as reference.
    python3 doodle.py --content samples/Gogh.jpg --style samples/Seth.jpg \
                      --variety 0.0 --balance 0.7 0.8 --layers 4_1 3_1 --iterations 4

To perform regular style transfer without semantic annotations, simply delete or rename the files with the semantic maps.  The photo is originally by `Seth Johnson <http://sethjohnson.tumblr.com/post/655063019/this-was-a-project-for-an-art-history-class-turns>`_, and the concept for this style transfer by `Kyle McDonald <https://twitter.com/kcimc>`_.

.. image:: docs/Portraits_example.jpg


1.c) Texture Synthesis
----------------------

For synthesizing bitmap textures, you only need an input style without anotations and without target output.  In this case, you simply specify one input style image and the output file as follows:

.. code:: bash

    # Generate an image of stones based on the input photograph only.
    python3 doodle.py --style samples/Stones.jpg --output Stones.png \
                      --layers 5_1 4_1 3_1 --iterations 6 4 4 --variety 0.4 0.2 0.1

You can also control the output resolution using ``--output-size=512x512`` parameter—which also depends on the memory you have available. By default the size will be the same as the style image.

.. image:: docs/Textures_example.jpg


1.d) Script Parameters
----------------------

You can configure the algorithm using the following parameters. Type ``python3 doodle.py --help`` for the full list of options, or see the source code.

* ``--style-weight=50.0`` — Weight of style relative to content.
* ``--style-layers=3_1,4_1`` — The layers to match style patches.
* ``--semantic-weight=1.0`` — Global weight of semantics vs. features.
* ``--smoothness=1.0`` — Weight of image smoothing scheme.
* ``--seed=noise`` — Seed image path, "noise" or "content".
* ``--print-every=10`` — How often to log statistics to stdout.
* ``--save-every=10`` — How frequently to save PNG into `frames`.


2. Installation & Setup
=======================

.. image:: docs/DockerConsole.gif

2.a) Using Docker Image (recommended)
-------------------------------------

The easiest way to get up-and-running is to `install Docker <https://www.docker.com/>`_. Then, you should be able to downloand and run the pre-built image using the ``docker`` command line tool.  Find out more about the ``alexjc/neural-doodle`` image on its `Docker Hub <https://hub.docker.com/r/alexjc/neural-doodle/>`_ page.

The easiest way to run the script from the docker image is to setup an easy access command called `doodle`. This will automatically:

* Mount the ``frames`` folder from current directory into the instance for visualization.
* Expose the ``samples`` folder from the current directory so the script can access files!

This is how you can do it in your terminal console on OSX or Linux:

.. code:: bash

    # Setup the alias. Put this in your .bash_rc or .zshrc file so it's available at startup.
    alias doodle="docker run -v $(pwd)/samples:/nd/samples -v $(pwd)/frames:/nd/frames \
                             -it alexjc/neural-doodle"
    
    # Now run any of the examples above using this alias, without the `.py` extension.
    doodle --help

If you want to run on your NVIDIA GPU, you can instead use the image ``alexjc/neural-doodle:gpu`` which comes with CUDA and CUDNN pre-installed in the image.  See the scripts in ``docker/*.sh`` for how to setup your host machine. (advanced)


2.b) Manual Installation (optional)
-----------------------------------

This project requires Python 3.4+ and you'll also need ``numpy`` and ``scipy`` (numerical computing libraries) as well as ``python3-dev`` installed system-wide.  If you want more detailed instructions, follow these:

1. `Linux Installation of Lasagne <https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04>`_ **(intermediate)**
2. `Mac OSX Installation of Lasagne <http://deeplearning.net/software/theano/install.html#mac-os>`_ **(advanced)**
3. `Windows Installation of Lasagne <https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Windows-7-%2864-bit%29>`_ **(expert)**

Afterward fetching the repository, you can run the following commands from your terminal to setup a local environment:

.. code:: bash

    # Create a local environment for Python 3.x to install dependencies here.
    python3 -m venv pyvenv --system-site-packages

    # If you're using bash, make this the active version of Python.
    source pyvenv/bin/activate

    # Setup the required dependencies simply using the PIP module.
    python3 -m pip install --ignore-installed -r requirements.txt

After this, you should have ``scikit-image``, ``theano`` and ``lasagne`` installed in your virtual environment.  You'll also need to download this `pre-trained neural network <https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2>`_ (VGG19, 80Mb) and put it in the same folder as the script to run. Once you're done you can just delete the ``#/pyvenv/`` folder.

.. image:: docs/Coastline_example.png


3. Troubleshooting Problems
===========================

It's running out of GPU Ram, throwing ``MemoryError``. Help!
------------------------------------------------------------

You'll need a good NVIDIA card with CUDA to run this software on GPU, ideally 2Gb / 4Gb or better still, 8Gb to 12Gb for larger resolutions.  The code does work on CPU by default, so use that as fallback since you likely have more system RAM!

To improve memory consumption, you can also install NVIDIA's ``cudnn`` library version 3.0 or 4.0. This allows convolutional neural networks to run faster and save space in GPU RAM.

**FIX:** Use ``--device=cpu`` to use main system memory.


How much GPU is being used? It doesn't seem very fast...
--------------------------------------------------------

First make sure CUDA is installed correctly and environment variables are set, then reinstall ``theano``.  If everything is setup correctly, the GPU should be used regularly as the gradient calculations are offloaded. If you run NVIDIA's monitoring tool it looks something like this:

.. code:: bash
    # gpu   pwr  temp    sm   mem   enc   dec  mclk  pclk
    # Idx     W     C     %     %     %     %   MHz   MHz
        0    88    63    50    25     0     0  3004  1252
        0    60    63     0     1     0     0  3004  1252
        0    75    63    19     9     0     0  3004  1252
        0    59    63     0     1     0     0  3004  1240
        0    62    63    16     3     0     0  3004  1240
        0    63    64     2     1     0     0  3004  1252
        0    66    63    26     4     0     0  3004  1252 

The third column is the utilitazition of compute resources, and the fourth column is the use of memory.  If memory is under-used you can increase resolution!  If compute is under allocated too you can try running multiple scripts in parallel!

**FIX:** Run ``nvidia-smi dmon`` and check the ``sm`` column.


Can't install or Unable to find pgen, not compiling formal grammar.
-------------------------------------------------------------------

There's a Python extension compiler called Cython, and it's missing or inproperly installed. Try getting it directly from the system package manager rather than PIP.

*FIX:* ``sudo apt-get install cython3``


NotImplementedError: AbstractConv2d theano optimization failed.
---------------------------------------------------------------

This happens when you're running without a GPU, and the CPU libraries were not found (e.g. ``libblas``).  The neural network expressions cannot be evaluated by Theano and it's raising an exception.

**FIX:** ``sudo apt-get install libblas-dev libopenblas-dev``


TypeError: max_pool_2d() got an unexpected keyword argument 'mode'
------------------------------------------------------------------

You need to install Lasagne and Theano directly from the versions specified in ``requirements.txt``, rather than from the PIP versions.  These alternatives are older and don't have the required features.

**FIX:** ``python3 -m pip install -r requirements.txt``


ValueError: unknown locale: UTF-8
---------------------------------

It seems your terminal is misconfigured and not compatible with the way Python treats locales. You may need to change this in your ``.bash_rc`` or other startup script. Alternatively, this command will fix it once for this shell instance.

**FIX:** ``export LC_ALL=en_US.UTF-8``


ERROR: The optimization diverged and NaNs were encountered.
-----------------------------------------------------------

It's possible there's a platform bug in the underlying libraries or compiler, which has been reported on MacOS El Capitan.  It's not clear how to fix it, but you can try to disable optimizations to prevent the bug. (See `Issue #8 <https://github.com/alexjc/neural-doodle/issues/8>`_.)

**FIX:** Use ``--safe-mode`` flag to disable optimizations.


4. Frequent Questions
=====================

Q: When will this be possible in realtime? I want it as filter!
---------------------------------------------------------------

Related algorithms have shown this is possible in realtime—if you're willing to accept slightly lower quality:

* `Texture Networks: Feed-forward Synthesis of Textures and Stylized Images <http://arxiv.org/abs/1603.03417>`_
* `Perceptual Losses for Real-Time Style Transfer and Super-Resolution <http://arxiv.org/abs/1603.08155>`_
* `Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks <http://arxiv.org/abs/1604.04382>`_

This project is not designed for real-time use, the focus is on quality.  The code in this repository is ideal for training realtime capable networks. 

Q: Is there an application for this? I want to download it!
-----------------------------------------------------------

There are many online services that provide basic style transfer with neural networks. We run `@DeepForger <https://deepforger.com/>`_, a Twitter & Facebook bot with web interface, that can take your requests too.  It takes time to make forgeries, so there's a queue... be patient!


----

|Python Version| |License Type| |Project Stars|

.. |Python Version| image:: http://aigamedev.github.io/scikit-neuralnetwork/badge_python.svg
    :target: https://www.python.org/

.. |License Type| image:: https://img.shields.io/badge/license-AGPL-blue.svg
    :target: https://github.com/alexjc/neural-doodle/blob/master/LICENSE

.. |Project Stars| image:: https://img.shields.io/github/stars/alexjc/neural-doodle.svg?style=flat
    :target: https://github.com/alexjc/neural-doodle/stargazers
