import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 30

def get_image_and_mask_paths(data_dir):
    image_paths = []
    mask_paths = []
    
    for class_name in ['normal', 'debut', 'grave']:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for file in os.listdir(class_dir):
            # Only process raw images (not masks)
            if file.endswith('.png') and '_mask' not in file:
                img_path = os.path.join(class_dir, file)
                
                # For each image, there might be 1 or multiple masks
                base_name = file.replace('.png', '')
                mask_files = [f for f in os.listdir(class_dir) if f.startswith(base_name + '_mask')]
                
                # Store the base name to load masks dynamically later or just pass the directory
                image_paths.append(img_path)
                mask_paths.append([os.path.join(class_dir, m) for m in mask_files])
                
    return image_paths, mask_paths

def read_image_and_mask(img_path, mask_paths):
    # Read image
    img_path = img_path.decode('utf-8') if isinstance(img_path, bytes) else img_path
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize
    
    # Read and combine masks
    combined_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    
    # Extract string paths
    if isinstance(mask_paths, bytes):
        paths_str = mask_paths.decode('utf-8')
        paths_list = paths_str.split('|') if paths_str else []
    else:
        paths_list = mask_paths.split('|') if mask_paths else []
        
    for m_path in paths_list:
        if not m_path: continue
        m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            m = cv2.resize(m, (IMG_WIDTH, IMG_HEIGHT))
            m = m / 255.0
            combined_mask = np.maximum(combined_mask, m)
            
    # Binary threshold
    combined_mask = (combined_mask > 0.5).astype(np.float32)
    combined_mask = np.expand_dims(combined_mask, axis=-1)
    
    return img.astype(np.float32), combined_mask

def tf_parse(img_path, mask_paths_str):
    img, mask = tf.numpy_function(read_image_and_mask, [img_path, mask_paths_str], [tf.float32, tf.float32])
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    mask.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    return img, mask

def get_dataset(data_dir):
    img_paths, mask_paths_list = get_image_and_mask_paths(data_dir)
    # Join mask paths with | for tf dataset processing
    mask_paths_str = ['|'.join(p) for p in mask_paths_list]
    
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths_str))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- U-Net Model ---
def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Dice Loss function
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

if __name__ == '__main__':
    print("Preparing Datasets...")
    train_dataset = get_dataset('datasets_split/train')
    val_dataset = get_dataset('datasets_split/val')
    
    print("Building U-Net Model...")
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss=dice_loss, 
                  metrics=['accuracy', dice_coef])
                  
    # Callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('models/unet_segmentation.keras', save_best_only=True, monitor='val_dice_coef', mode='max'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    print("Training U-Net...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print("Training Complete. Model saved to models/unet_segmentation.keras")
