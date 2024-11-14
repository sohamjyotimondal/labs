import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from visiontransformer import build_vision_transformer  # Import your ViT model
from patchify import patchify
from sklearn.preprocessing import LabelEncoder

""" Vision Transformer Config (same as training) """
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
cf["num_classes"] = 440  # Number of whale IDs

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

def load_sample_data(sample_csv, img_dir):
    """ Load sample submission and corresponding image paths """
    df = pd.read_csv(sample_csv)
    images = df["ID"].values
    image_paths = [os.path.join(img_dir, img_name) for img_name in images]
    return image_paths, df

def predict_and_save(model, image_paths, df, output_csv, label_encoder):
    """ Predict on images and save results """
    predictions = []
    for img_path in image_paths:
        image = read_image(img_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        pred = model.predict(image)
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])  # Get the predicted whale ID
        predictions.append(pred_label[0])

    # Save predictions to CSV
    df["whaleID"] = predictions
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    """ Load Model """
    model_path = "vision_transformer/model.h5"
    model = build_vision_transformer(cf)
    model.load_weights(model_path)

    """ Load Label Encoder """
    label_encoder_classes = np.load("vision_transformer/label_encoder.npy", allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
    sample_csv = "whales/sample_submission.csv"
    img_dir = "whales/imgs/"
    image_paths, sample_df = load_sample_data(sample_csv, img_dir)

    """ Predict and Save """
    output_csv = "whales/sample_predictions.csv"
    predict_and_save(model, image_paths, sample_df, output_csv, label_encoder)
