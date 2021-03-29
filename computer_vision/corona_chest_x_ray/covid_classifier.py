import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    # step: first create normal/pneu classifier (then virus/non-virus, then covid,non-covid)

    # remember: parameters and model are adapted from the book (Deep Learninig with Python by Francois Chollet)

    covid_model = models.Sequential()
    covid_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
    covid_model.add(layers.MaxPooling2D(2, 2))
    covid_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    covid_model.add(layers.MaxPooling2D(2, 2))
    covid_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    covid_model.add(layers.Flatten())
    covid_model.add(layers.Dropout(0.5))
    covid_model.add(layers.Dense(512, activation='relu'))
    covid_model.add(layers.Dense(1, activation='sigmoid'))
    covid_model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

    train_covid_gen = ImageDataGenerator(rescale=1. / 255,
                                         rotation_range=40,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         )

    valid_covid_gen = ImageDataGenerator(rescale=1. / 255)

    covid_train_dir = 'Coronahack-Chest-XRay-Dataset/train/covid_noncovid'
    covid_valid_dir = 'Coronahack-Chest-XRay-Dataset/valid/covid_noncovid'

    # remember: x-ray data uses grayscale
    batch_size = 5
    train_covid_generator = train_covid_gen.flow_from_directory(
        covid_train_dir, target_size=(150, 150), batch_size=batch_size,
        class_mode='binary', color_mode='grayscale', )
    valid_covid_generator = valid_covid_gen.flow_from_directory(
        covid_valid_dir,
        target_size=(150, 150), batch_size=batch_size, class_mode='binary', color_mode='grayscale')

    # step: fit
    # remember: steps_per_epoch * epochs <= num images
    history = covid_model.fit(
        train_covid_generator,
        steps_per_epoch=int(train_covid_generator.n / batch_size),
        epochs=30,
        validation_data=valid_covid_generator,
        validation_steps=int(valid_covid_generator.n / batch_size),
    )

    #  step: save
    covid_model.save('covid_model_data/covid_small_1.h5')

    ####################################################################################################################

    plt.plot(history.history['acc'], label='Training acc', linestyle=':', linewidth=5)
    plt.plot(history.history['val_acc'], label='Validation acc', linewidth=3)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training and validation accuracy')
    plt.savefig('covid_model_data/covid_train_val_acc.pdf')

    plt.clf()
    plt.plot(history.history['loss'], label='Training loss', linestyle=':', linewidth=5)
    plt.plot(history.history['val_loss'], label='Validation loss', linewidth=3)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.savefig('covid_model_data/covid_train_val_loss.pdf')

    # plt.show()

    # Comments: Whilst high accuracy is obtained, it is unclear if the results are reliable due to the small dataset.

