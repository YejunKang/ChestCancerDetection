#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to the image
        target_size: Size to resize the image to
        
    Returns:
        preprocessed_img: Preprocessed image ready for model input
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img_resized = cv2.resize(img, target_size)
    
    img_normalized = img_resized / 255.0
    
    preprocessed_img = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    preprocessed_img = np.expand_dims(preprocessed_img, axis=-1)  # Add channel dimension
    
    return preprocessed_img

def get_class_names():
    """
    Get class names from the text file or use defaults
    
    Returns:
        class_names: List of class names
    """
    try:
        if os.path.exists('class_names.txt'):
            with open('class_names.txt', 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            return class_names
        else:
            return ['normal', 'large_cell_carcinoma', 'squamous_cell']
    except Exception as e:
        print(f"Error reading class names: {e}")
        return ['normal', 'large_cell_carcinoma', 'squamous_cell']

def generate_heatmap(model, img, predicted_class_idx, image_path):
    """
    Generate Grad-CAM heatmap highlighting areas of interest
    
    Args:
        model: Trained model
        img: Preprocessed image
        predicted_class_idx: Index of the predicted class
        image_path: Path to original image for visualization
        
    Returns:
        superimposed_img: Image with heatmap overlay
    """
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.output, last_conv_layer.output]
        )
        
        with tf.GradientTape() as tape:
            preds, conv_outputs = grad_model(img)
            class_channel = preds[:, predicted_class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        original_img = cv2.imread(image_path)
        height, width, _ = original_img.shape
        heatmap_resized = cv2.resize(heatmap, (width, height))
        
        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
        heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
        
        alpha = 0.4  
        superimposed_img = heatmap_rgb * alpha + original_img * (1 - alpha)
        superimposed_img = superimposed_img.astype(np.uint8)
        
        return superimposed_img
    
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

def display_cancer_type_info(predicted_class):
    """
    Display information about the predicted cancer type
    
    Args:
        predicted_class: Name of the predicted class
    """
    if predicted_class == 'normal':
        print("\nInformation: The CT scan appears normal without significant findings.")
        print("Recommendation: Regular screening as recommended by healthcare guidelines.")
    
    elif predicted_class == 'large_cell_carcinoma':
        print("\nWarning: The CT scan shows signs consistent with Large Cell Carcinoma.")
        print("\nLarge Cell Carcinoma Information:")
        print("- A type of non-small cell lung cancer (NSCLC)")
        print("- Characterized by large, abnormal cells that grow and divide rapidly")
        print("- Typically aggressive and can spread quickly")
        print("- Often diagnosed at later stages due to lack of early symptoms")
        print("\nRecommendation: Immediate consultation with an oncologist is strongly advised.")
    
    elif predicted_class == 'squamous_cell':
        print("\nWarning: The CT scan shows signs consistent with Squamous Cell Carcinoma.")
        print("\nSquamous Cell Carcinoma Information:")
        print("- A type of non-small cell lung cancer (NSCLC)")
        print("- Develops from the flat cells that line the bronchial tubes")
        print("- Often correlated with smoking history")
        print("- Typically grows in the central part of the lungs")
        print("\nRecommendation: Prompt evaluation by a lung specialist is strongly advised.")
    
    else:
        print(f"\nPredicted class: {predicted_class}")
        print("Recommendation: Follow up with a healthcare provider for proper evaluation.")

def main():
    """
    Main function for lung cancer detection from the command line
    """
    print("Lung Cancer Detection from CT Scans")
    print("===================================\n")
    
    model_path = 'lung_cancer_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first using the training script.")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    class_names = get_class_names()
    print(f"Classes: {', '.join(class_names)}")
    
    while True:
        image_path = input("\nEnter the path to the CT scan image (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found.")
            continue
        
        try:
            preprocessed_img = preprocess_image(image_path)
            
            predictions = model.predict(preprocessed_img)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = class_names[predicted_class_idx]
            
            heatmap_img = generate_heatmap(model, preprocessed_img, predicted_class_idx, image_path)
            
            print("\nPrediction Results:")
            print("-----------------")
            print(f"Diagnosis: {predicted_class.upper()}")
            print(f"Confidence: {confidence:.2%}")
            
            display_cancer_type_info(predicted_class)
            
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.title("Original CT Scan")
            plt.imshow(original_img, cmap='gray')
            plt.axis('off')
            
            if heatmap_img is not None:
                plt.subplot(1, 2, 2)
                plt.title("Areas of Interest")
                plt.imshow(heatmap_img)
                plt.axis('off')
                
                result_filename = f"analysis_{os.path.basename(image_path)}.png"
                plt.savefig(result_filename)
                print(f"\nVisualization saved as '{result_filename}'")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()