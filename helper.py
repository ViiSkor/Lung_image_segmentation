from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

ABBR = "CHNCXR"

def get_dataset_pathes(img_paths, mask_paths):
    without_mask = []
    dataset_images = []
    dataset_masks = []
      
    for image_dir in img_paths:
        mask_dir = image_dir.replace("images", "masks")
        if ABBR in mask_dir:
            mask_dir = '_'.join([mask_dir[:-4], "mask.png"])
        
        if mask_dir in mask_paths:        
            dataset_images.append(image_dir)
            dataset_masks.append(mask_dir)
        else:
            without_mask.append(image_dir)
                    
    return without_mask, np.array(dataset_images), np.array(dataset_masks)

def split_test_train(dataset_image_paths, dataset_mask_paths, ratio=0.8, seed=42):
  idx_lst = [i for i in range(len(dataset_image_paths))]
  random.seed(seed)
  random.shuffle(idx_lst)
  cut_idx = int(len(dataset_image_paths)*ratio)
  train_image_paths = dataset_image_paths[idx_lst[:cut_idx]]
  train_mask_paths = dataset_mask_paths[idx_lst[:cut_idx]]
  test_image_paths = dataset_image_paths[idx_lst[cut_idx:]]
  test_mask_paths = dataset_mask_paths[idx_lst[cut_idx:]]
  
  train_image_paths = train_image_paths.tolist()
  train_mask_paths = train_mask_paths.tolist()
  test_image_paths = test_image_paths.tolist()
  test_mask_paths = test_mask_paths.tolist()
  
  return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths

def creat_dataset_dir(image_dir, masks_dir, image_paths, mask_paths):
  os.makedirs(image_dir)
  os.makedirs(masks_dir)
  
  for i, (image, mask) in enumerate(zip(image_paths, mask_paths)):
    os.rename(image, image_dir+str(i)+".png")
    os.rename(mask, masks_dir+str(i)+".png")
    
def preprocess(image, mask):
    image = image / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return image, mask

def generator(batch_size, path, target_size=(512, 512), augmentation=False, seed=42):
  dg_args = {}
  if augmentation:
    dg_args = dict(rotation_range=0.2,
                     width_shift_range=0.08,
                     height_shift_range=0.08,
                     shear_range=0.05,
                     zoom_range = 0.05,
                     horizontal_flip=True,
                     vertical_flip = False,
                     fill_mode='nearest')
  image_datagen = ImageDataGenerator(**dg_args)
  mask_datagen = ImageDataGenerator(**dg_args)
    
  image_generator = image_datagen.flow_from_directory(
      path,
      classes = ["images"],
      target_size = target_size,
      batch_size = batch_size,
      seed = seed)

  mask_generator = mask_datagen.flow_from_directory(
      path,
      classes = ["masks"],
      color_mode = "grayscale",
      target_size = target_size,
      batch_size = batch_size,
      seed = seed)

  for image, mask in zip(image_generator, mask_generator):
    image, mask = preprocess(image[0], mask[0])
    yield (image, mask)
    
def plot_data(dataset_images, dataset_masks, generator):
  fig = plt.figure(figsize=(24, 10))

  for i in range(16):
    image, mask = next(generator)
    plt.subplot(4, 8, i*2+1)
    plt.imshow(image[0, :, :, :])
    plt.axis('off')
    plt.subplot(4, 8, i*2+2)
    plt.imshow(mask[0, :, :, 0], cmap='gray')
    plt.axis('off')

  plt.show()
  
def plot_result(X, y, prediction):
  fig = plt.figure(figsize=(15, 12))
    
  for i in range(3):
    j = np.random.randint(len(X))
    plt.subplot(3, 4, i*4+1)
    plt.imshow(X[j, :, :, :])
    plt.axis('off')
      
    plt.subplot(3, 4, i*4+2)
    plt.gca().set_title('Origin mask')
    plt.imshow(y[j, :, :, 0], cmap='gray')
    plt.axis('off')
      
    plt.subplot(3, 4, i*4+3)
    plt.gca().set_title('Predicted mask')
    plt.imshow(prediction[j, :, :, 0], cmap='gray')
    plt.axis('off')
      
    overlay = cv2.addWeighted(y[j, :, :, 0], 0.5, prediction[j, :, :, 0], 0.5, 0)
    plt.subplot(3, 4, i*4+4)
    plt.gca().set_title('Overlay origin and predicted mask')
    plt.imshow(overlay)
    plt.axis('off')
    
  plt.show()