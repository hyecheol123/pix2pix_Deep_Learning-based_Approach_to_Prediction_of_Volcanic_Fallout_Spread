## Training/test Tips
#### Training/test options
Please see `options/train_options.py` and `options/base_options.py` for the training flags; see `options/test_options.py` and `options/base_options.py` for the test flags. There are some model-specific flags as well, which are added in the model files, such as `--lambda_A` option in `model/cycle_gan_model.py`. The default values of these options are also adjusted in the model files.
#### CPU/GPU (default `--gpu_ids 0`)
Please set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g., `--batch_size 32`) to benefit from multiple GPUs.

#### Visualization
During training, the current results can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will appear on a local graphics web server launched by [visdom](https://github.com/facebookresearch/visdom). To do this, you should have `visdom` installed and a server running by the command `python -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the `visdom` server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id -1`. Second, the intermediate results are saved to `[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid this, set `--no_html`.

#### Preprocessing
 Images can be resized and cropped in different ways using `--preprocess` option. The default option `'resize_and_crop'` resizes the image to be of size `(opt.load_size, opt.load_size)` and does a random crop of size `(opt.crop_size, opt.crop_size)`. `'crop'` skips the resizing step and only performs random cropping. `'scale_width'` resizes the image to have width `opt.crop_size` while keeping the aspect ratio. `'scale_width_and_crop'` first resizes the image to have width `opt.load_size` and then does random cropping of size `(opt.crop_size, opt.crop_size)`. `'none'` tries to skip all these preprocessing steps. However, if the image size is not a multiple of some number depending on the number of downsamplings of the generator, you will get an error because the size of the output image may be different from the size of the input image. Therefore, `'none'` option still tries to adjust the image size to be a multiple of 4. You might need a bigger adjustment if you change the generator architecture. Please see `data/base_datset.py` do see how all these were implemented.

#### Fine-tuning/resume training
To fine-tune a pre-trained model, or resume the previous training, use the `--continue_train` flag. The program will then load the model based on `epoch`. By default, the program will initialize the epoch count as 1. Set `--epoch_count <int>` to specify a different starting epoch count.

#### Prepare your own datasets for pix2pix
Pix2pix's training requires paired data. We provide a python script to generate training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene. For example, these might be pairs {label map, photo} or {bw image, color image}. Then we can learn to translate A to B or B to A:

Create folder `/path/to/data` with subdirectories `A` and `B`. `A` and `B` should each have their own subdirectories `train`, `val`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.


#### About image size
 Since the generator architecture in CycleGAN involves a series of downsampling / upsampling operations, the size of the input and output image may not match if the input image size is not a multiple of 4. As a result, you may get a runtime error because the L1 identity loss cannot be enforced with images of different size. Therefore, we slightly resize the image to become multiples of 4 even with `--preprocess none` option. For the same reason, `--crop_size` needs to be a multiple of 4.

#### Training/Testing with high res images
CycleGAN is quite memory-intensive as four networks (two generators and two discriminators) need to be loaded on one GPU, so a large image cannot be entirely loaded. In this case, we recommend training with cropped images. For example, to generate 1024px results, you can train with `--preprocess scale_width_and_crop --load_size 1024 --crop_size 360`, and test with `--preprocess scale_width --load_size 1024`. This way makes sure the training and test will be at the same scale. At test time, you can afford higher resolution because you don’t need to load all networks.

#### Training/Testing with rectangular images
Both pix2pix and CycleGAN can work for rectangular images. To make them work, you need to use different preprocessing flags. Let's say that you are working with `360x256` images. During training, you can specify `--preprocess crop` and `--crop_size 256`. This will allow your model to be trained on randomly cropped `256x256` images during training time. During test time, you can apply the model on `360x256` images with the flag `--preprocess none`.

There are practical restrictions regarding image sizes for each generator architecture. For `unet256`, it only supports images whose width and height are divisible by 256. For `unet128`, the width and height need to be divisible by 128. For `resnet_6blocks` and `resnet_9blocks`, the width and height need to be divisible by 4.

#### About loss curve
Unfortunately, the loss curve does not reveal much information in training GANs, and CycleGAN is no exception. To check whether the training has converged or not, we recommend periodically generating a few samples and looking at them.

#### About batch size
For all experiments in the paper, we set the batch size to be 1. If there is room for memory, you can use higher batch size with batch norm or instance norm. (Note that the default batchnorm does not work well with multi-GPU training. You may consider using [synchronized batchnorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) instead). But please be aware that it can impact the training. In particular, even with Instance Normalization, different batch sizes can lead to different results. Moreover, increasing `--crop_size` may be a good alternative to increasing the batch size.
