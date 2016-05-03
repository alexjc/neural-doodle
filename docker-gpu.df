FROM nvidia/cuda:7.5-cudnn4-devel

# Install dependencies

RUN apt-get update               && \
    apt-get install --assume-yes    \
        "module-init-tools"         \
        "build-essential"           \
        "cmake"                     \
        "git"                       \
        "wget"                      \
        "libopenblas-dev"           \
        "liblapack-dev"             \
        "libjpeg-dev"               \
        "libtiff5-dev"              \
        "zlib1g-dev"                \
        "libfreetype6-dev"          \
        "liblcms2-dev"              \
        "libwebp-dev"               \
        "tcl8.6-dev"                \
        "tk8.6-dev"                 \
        "gfortran"                  \
        "python3"                   \
        "python3-dev"               \
        "python3-pip"               \
        "python3-numpy"             \
        "python3-scipy"             \
        "python3-matplotlib"        \
        "python3-six"               \
        "python3-networkx"          \
        "python3-tk"             && \
    python3 -m pip install "cython"

# Install OpenJPEG

RUN cd "/tmp"                                                                                 && \
    wget "http://iweb.dl.sourceforge.net/project/openjpeg.mirror/2.1.0/openjpeg-2.1.0.tar.gz" && \
    tar --extract --verbose --gzip --file openjpeg*.tar.gz                                    && \
    rm openjpeg*.tar.gz                                                                       && \
    cd openjpeg*                                                                              && \
    cmake -D "CMAKE_INSTALL_PREFIX=/usr" .                                                    && \
    make                                                                                      && \
    make install                                                                              && \
    rm --recursive --force *

# Download and install Neural Doodle with Theano and Lasagne

RUN git clone "https://github.com/alexjc/neural-doodle.git" "/neural-doodle"
WORKDIR /neural-doodle

RUN python3 -m pip install --ignore-installed -r "requirements.txt"

# Copy Theano's configuration

COPY .theanorc /root/

# Get a pre-trained neural network (VGG19)

RUN wget "https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2"

# Set an entrypoint to the main doodle.py script

ENTRYPOINT ["python3", "doodle.py", "--device=gpu"]
