from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    norm_pneu_test_generator = test_datagen.flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/norm_pneu',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        color_mode='grayscale'
    )

    norm_pneu_model = load_model('pneumonia_model_data/normal_and_pneumonia_small_1.h5')

    test_loss, test_acc = norm_pneu_model.evaluate(norm_pneu_test_generator,
                                                   steps=int(
                                                       norm_pneu_test_generator.n / norm_pneu_test_generator.batch_size))
    print("Norm/Pneu loss,acc:", test_loss, test_acc)

    virus_bac_test_generator = test_datagen.flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/virus_bacteria',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        color_mode='grayscale'
    )
    virus_bac_model = load_model('virus_model_data/virus_small_1.h5')
    test_loss, test_acc = virus_bac_model.evaluate(virus_bac_test_generator,
                                                   steps=int(
                                                       virus_bac_test_generator.n / virus_bac_test_generator.batch_size))
    print("Virus/Bac loss,acc:", test_loss, test_acc)

    covid_test_generator = test_datagen.flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/covid_noncovid',
        target_size=(150, 150),
        batch_size=5,
        class_mode='binary',
        color_mode='grayscale'
    )

    covid_model = load_model('covid_model_data/covid_small_1.h5')
    test_loss, test_acc = covid_model.evaluate(covid_test_generator,
                                               steps=int(
                                                   covid_test_generator.n / covid_test_generator.batch_size))
    print("Covid/Non-covid loss,acc:", test_loss, test_acc)
