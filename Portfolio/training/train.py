import tensorflow as tf
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.tensorflow import TensorFlow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the layers of ResNet50
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Define paths to S3
    bucket = 'lntav'
    prefix = 'road sign/'
    s3_train_path = f's3://{bucket}/{prefix}'
    
    # Create data generators
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Load data from S3 directly
    train_generator = datagen.flow_from_directory(
        s3_train_path,
        subset='training',
        batch_size=32,
        target_size=(224, 224),
        class_mode='categorical'
    )
    
    validation_generator = datagen.flow_from_directory(
        s3_train_path,
        subset='validation',
        batch_size=32,
        target_size=(224, 224),
        class_mode='categorical'
    )
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    
    # Create and train model
    model = create_model(num_classes)
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator
    )
    
    # Save the model
    model.save('/opt/ml/model/road_sign_model.h5')

if __name__ == '__main__':
    main()
