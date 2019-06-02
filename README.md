# Lung image segmentation
## Overview
### Data
Data is loaded and processed by methods in helper.py. The method get_dataset_pathes() is used to get pathes image with masks from provided dataset. The images are resized to 512 x 512 and scaled to [0, 1] interval. The batch generator can augment images, if you set flag augmentation=True in generator(). The dataset, which I used to train the model, contains 704 pairs of images of lung X-rays and masks. The train dataset has 563 samples and the test dataset has 141 samples
### Model
The provided model has an [U-Net](https://arxiv.org/abs/1505.04597) like architecture type. As encoder was chosen pretrained [VGG-16](https://keras.io/applications/#vgg16). This deep neural network is implemented with Keras functional API. Output from the network is a 512x512x1, which represents mask that should be learned. 
### Training
The model is trained for 20 epochs. After 20 epochs, calculated loss funcition is ~0.11 and Mean IoU is ~0.93%. Loss function for the training is a binary crossentropy + log(Dice). For Dice set smooth factor = 1. The weights are updated by Adam optimizer, with a 1e-4 learning rate. 
During training, model's weights are saved in HDF5 format.
![Graphs](https://user-images.githubusercontent.com/42701384/58761544-3a166400-854e-11e9-94a0-73a06ec2ecc1.png)
![Results](https://user-images.githubusercontent.com/42701384/58761453-38986c00-854d-11e9-8e21-bfcdd577de5b.png)
## How to use

### Run training
```python
unet = model.UNet(loss=losses.bce_logdice_loss, metrics=[metrics.mean_iou], lr=lr)
model = unet.get_model(INPUT_SHAPE)
train_gen = helper.generator(BATCH_SIZE, TRAIN_DATASET_DIR, augmentation=True)
test_gen = helper.generator(BATCH_SIZE, TEST_DATASET_DIR)
history = model.fit_generator(train_gen, epochs=EPOCHS, 
                              steps_per_epoch=int(N_TRAIN / BATCH_SIZE), 
                              validation_data=test_gen, 
                              validation_steps=int(N_TEST / BATCH_SIZE),  
                              use_multiprocessing=True)
```
Or you can use a provided google colab [notebook](https://colab.research.google.com/drive/1H5gjJbEivlrK-CnbL9CO1tP1nnItt6wn).
### Get mask for your image
You can download [model's weights](https://drive.google.com/file/d/1UB0QL08Na48ASvP8z7QT0BKMTh5STWwC/view?usp=sharing).
```python
image = cv2.imread(path_to_image)
image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
image = image / 255
model.load_weights(path_to_model)
x = np.array([image])
prediction = model.predict(x, verbose=1)
```
## Further improving
* In the provided dataset we have a bunch of images without a mask. In case we have a lot of images of masks it may be relevant to try to generate masks with GAN as proposed in paper [«Deep Multi-Class Segmentation Without Ground-Truth Labels»](https://openreview.net/forum?id=S11Xr-3iM). Also, we may try [unsupervised segmentation quality assessment](https://arxiv.org/abs/1903.08773) for image segmentation without masks. Straightforward approach: make the masks themselves. It is suitable for cases when dataset without masks is small, but it isn’t suitable for cases with a large dataset.
* The resulting masks are somewhat blurry, we can try to find a loss that produces “sharp” boundaries of the mask. For example, loss Dice produces “sharp masks”, but at the same time with it, you get seamless superfluous pieces of the mask, so at the moment I stopped at bce + log(Dice). But there is still something to try.
* This model has signs of tendency to overfitting. We need to use regulizaton methods, for example, we can use Batch Normalization or Dropout.
* This model stops improving pretty early. We need to try different learning rate reduce methods or another optimizer.
