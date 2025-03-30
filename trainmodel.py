import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(data_dir, img_size=(224, 224)):
    """
    Load and preprocess images from the dataset
    
    Args:
        data_dir: Directory containing class folders with images
        img_size: Target size for image resizing
        
    Returns:
        X: Preprocessed images
        y: Labels
        class_names: List of class names
    """
    X = []
    y = []
    class_names = ['normal', 'large_cell_carcinoma', 'squamous_cell']
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory {class_dir} not found. Creating empty directory.")
            os.makedirs(class_dir, exist_ok=True)
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                img = cv2.resize(img, img_size)
                img = img / 255.0
                
                img = np.expand_dims(img, axis=-1)
                
                X.append(img)
                y.append(i)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(X), np.array(y), class_names

def create_model(input_shape, num_classes):
    """
    Create a CNN model for lung cancer detection
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes to predict
        
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Train the model with data augmentation
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size: Batch size for training
        epochs: Number of epochs to train
        
    Returns:
        history: Training history
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    datagen.fit(X_train)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )
    
    return history

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the model and print metrics
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data and labels
        class_names: List of class names
        
    Returns:
        test_loss, test_acc: Test loss and accuracy
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return test_loss, test_acc

def visualize_training_history(history):
    """
    Visualize training history
    
    Args:
        history: Training history
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def save_model(model, model_path, class_names):
    """
    Save the model and class names
    
    Args:
        model: Trained Keras model
        model_path: Path to save the model
        class_names: List of class names
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    with open('model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print("Class names saved to class_names.txt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Lung Cancer Detection from CT Scans')
    parser.add_argument('--data_dir', type=str, default='chest_ct_data',
                        help='Directory containing class folders with images')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--model_path', type=str, default='lung_cancer_model.h5',
                        help='Path to save the trained model')
    args = parser.parse_args()
    
    # Set parameters
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    epochs = args.epochs
    
    print("Loading and preprocessing data...")
    X, y, class_names = load_and_preprocess_data(args.data_dir, img_size)
    
    if len(X) == 0:
        print("Error: No images found in the specified directories.")
        print(f"Please ensure images are placed in the following folders within {args.data_dir}:")
        for class_name in ['normal', 'large_cell_carcinoma', 'squamous_cell']:
            print(f"  - {class_name}")
        exit(1)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Classes: {class_names}")
    
    print("Creating model...")
    input_shape = X_train.shape[1:]
    num_classes = len(class_names)
    model = create_model(input_shape, num_classes)
    model.summary()
    
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs)
    
    visualize_training_history(history)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, class_names)
    
    save_model(model, args.model_path, class_names)
    
    print("Training and evaluation complete!")