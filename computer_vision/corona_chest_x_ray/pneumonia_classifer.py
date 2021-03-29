import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    # step: first create normal/pneu classifier (then virus/non-virus, then covid,non-covid)

    # remember: parameters and model are adapted from the book (Deep Learninig with Python by Francois Chollet)

    normpneu_model = models.Sequential()
    normpneu_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
    normpneu_model.add(layers.MaxPooling2D(2, 2))
    normpneu_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    normpneu_model.add(layers.MaxPooling2D(2, 2))
    normpneu_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    normpneu_model.add(layers.Flatten())
    normpneu_model.add(layers.Dense(512, activation='relu'))
    normpneu_model.add(layers.Dense(1, activation='sigmoid'))
    normpneu_model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

    train_norm_pneu_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             )
    valid_norm_pneu_gen = ImageDataGenerator(rescale=1. / 255)

    norm_pneu_train_dir = 'Coronahack-Chest-XRay-Dataset/train/norm_pneu'
    norm_pneu_valid_dir = 'Coronahack-Chest-XRay-Dataset/valid/norm_pneu'

    # remember: x-ray data uses grayscale
    batch_size = 20
    train_norm_pneu_generator = train_norm_pneu_gen.flow_from_directory(
        norm_pneu_train_dir, target_size=(150, 150), batch_size=batch_size,
        class_mode='binary', color_mode='grayscale', )
    valid_norm_pneu_generator = valid_norm_pneu_gen.flow_from_directory(
        norm_pneu_valid_dir,
        target_size=(150, 150), batch_size=batch_size, class_mode='binary', color_mode='grayscale')

    # step: fit
    # remember: steps_per_epoch * epochs <= num images
    history = normpneu_model.fit(
        train_norm_pneu_generator,
        steps_per_epoch=int(train_norm_pneu_generator.n / batch_size),
        epochs=30,
        validation_data=valid_norm_pneu_generator,
        validation_steps=int(valid_norm_pneu_generator.n / batch_size)
    )

    # # step: save
    normpneu_model.save('pneumonia_model_data/normal_and_pneumonia_small_1.h5')

    ####################################################################################################################

    plt.plot(history.history['acc'], label='Training acc', linestyle=':', linewidth=5)
    plt.plot(history.history['val_acc'], label='Validation acc', linewidth=3)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training and validation accuracy')
    plt.savefig('pneumonia_model_data/norm_pneu_train_val_acc.pdf')

    plt.clf()
    plt.plot(history.history['loss'], label='Training loss', linestyle=':', linewidth=5)
    plt.plot(history.history['val_loss'], label='Validation loss', linewidth=3)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.savefig('pneumonia_model_data/norm_pneu_train_val_loss.pdf')

    # plt.show()

    # Comments: classifying between healthy and unhealthy appears quite possible.
