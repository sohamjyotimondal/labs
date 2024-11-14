import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
from visiontransformer import build_vision_transformer  # Import your ViT model
from tensorflow.keras.utils import to_categorical
from patchify import patchify
import numpy as np
import cv2

""" Vision Transformer Config """
cf = {}
cf["image_size"] = 256
cf["num_layers"] = 12
cf["hidden_dim"] = 128
cf["mlp_dim"] = 32
cf["num_heads"] = 6
cf["dropout_rate"] = 0.1
cf["patch_size"] = 16
cf["num_channels"] = 3
cf["num_patches"] = (cf["image_size"] // cf["patch_size"]) ** 2
cf["num_classes"] = 440  # Set number of classes for whale IDs
cf["num_experts"] = 5  # Number of experts
cf["top_k"] = 3  # Top-k gating (not used in expert choice routing)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(csv_path, img_dir):
    """ Load CSV and prepare dataset """
    df = pd.read_csv(csv_path)
    
    images = df["Image"].values
    labels = df["whaleID"].values

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, num_classes=cf["num_classes"])
    #save the label encoder in csv
    le_df = pd.DataFrame(le.classes_, columns=["whaleID"])
    le_df.to_csv("vision_transformer/label_encoders.csv", index=False)
    #save as npy file
    np.save("vision_transformer/label_encoder.npy", le.classes_)

    # Prepare file paths for images
    image_paths = [os.path.join(img_dir, img_name) for img_name in images]

    # Split data
    train_x, test_x, train_y, test_y = train_test_split(image_paths, labels, test_size=0.1, random_state=42)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y), le

def read_image(path):
    """ Reads and preprocesses image for Vision Transformer and splits into patches """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image = image / 255.0  # Normalize pixel values
    
    """ Create patches """
    patch_size = cf["patch_size"]
    patches = patchify(image, (patch_size, patch_size, cf["num_channels"]), step=patch_size)
    
    # Reshape patches into a flat array (num_patches, patch_size * patch_size * num_channels)
    patches = patches.reshape(-1, patch_size * patch_size * cf["num_channels"])
    
    return patches.astype(np.float32)

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x.decode())  # Read and process image
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])
    y.set_shape([cf["num_classes"]])
    return x, y


def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(lambda x, y: tf_parse(x, y)).batch(batch).prefetch(10)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    filedir = "vision_transformer"
    create_dir(filedir)

    """ Hyperparameters """
    batch_size = 4
    lr = 0.01
    num_epochs = 60
    model_path = os.path.join(filedir, "model.h5")
    csv_path = os.path.join(filedir, "log.csv")
    dataset_csv = "whales/train.csv"
    img_dir = "whales/imgs/"

    """ Load Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y), label_encoder = load_dataset(dataset_csv, img_dir)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_vision_transformer(cf)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr), metrics=["accuracy"])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    """ Train Model """
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    """ Save Label Encoder """
    np.save(os.path.join(filedir, "label_encoder.npy"), label_encoder.classes_)
