import matplotlib.pyplot as plt
import numpy as np
from keras import models
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def create_visualisations(model, class_one_img_generator, class_two_img_generator, activation_model, num_layers,
                          class_names,
                          num_cols=16, num_samples=100, digitize=False, num_bins=7, main_cmap='viridis',
                          diff_cmap='seismic'):
    layer_names = [layer.name for layer in model.layers[:num_layers]]

    activations_one = activation_model.predict(class_one_img_generator.next()[0])
    activations_two = activation_model.predict(class_two_img_generator.next()[0])

    for _ in range(1, num_samples):
        temp_class1_acts = activation_model.predict(class_one_img_generator.next()[0])
        temo_class2_acts = activation_model.predict(class_two_img_generator.next()[0])

        for l in range(num_layers):
            activations_one[l] += temp_class1_acts[l]
            activations_two[l] += temo_class2_acts[l]

    for l in range(num_layers):
        activations_one[l] /= (num_samples + 1)
        activations_two[l] /= (num_samples + 1)

    for i in range(num_layers):

        layer_name, class1_activation_layer, class2_activation_layer = layer_names[i], activations_one[i], \
                                                                       activations_two[i]

        if 'conv' not in layer_name:
            continue # ignores max-pool layers ect

        num_filters = class1_activation_layer.shape[-1]
        size = class1_activation_layer.shape[1]

        num_rows = num_filters // num_cols  # integer division
        display_grid_one = np.zeros((num_rows * size, num_cols * size))
        display_grid_two = display_grid_one.copy()

        for row in range(num_rows):
            for col in range(num_cols):
                # update display grid image with filter
                filt_idx = (row * num_cols) + col
                if num_layers >= 2:
                    class1_img = class1_activation_layer[0, ..., filt_idx]
                    class2_img = class2_activation_layer[0, ..., filt_idx]
                else:
                    class1_img = class1_activation_layer[..., filt_idx]
                    class2_img = class2_activation_layer[..., filt_idx]

                class1_img = post_process_image(class1_img)
                class2_img = post_process_image(class2_img)

                display_grid_one[row * size: (row + 1) * size, col * size: (col + 1) * size] = class1_img
                display_grid_two[row * size: (row + 1) * size, col * size: (col + 1) * size] = class2_img

        if digitize:
            bins = np.linspace(0, 255, num_bins, endpoint=True)
            # bins = np.linspace(np.min(np.r_[norm_display_grid, pneu_display_grid]),
            #                    np.max(np.r_[norm_display_grid, pneu_display_grid]), num_bins)
            norm_idxs = np.digitize(display_grid_one,
                                    bins=bins, right=True)
            pneu_idxs = np.digitize(display_grid_two,
                                    bins=bins, right=True)

            display_grid_one = bins[norm_idxs].reshape(num_rows * size, num_cols * size)
            display_grid_two = bins[pneu_idxs].reshape(num_rows * size, num_cols * size)

        fig, axs = plt.subplots(ncols=1, nrows=4)

        axs[0].set_title(layer_name + ' : ' + class_names[0])
        axs[0].imshow(display_grid_one, aspect='auto', cmap=main_cmap)

        axs[1].set_title(layer_name + ' : ' + class_names[1])
        axs[1].imshow(display_grid_two, aspect='auto', cmap=main_cmap)

        avg_diff = display_grid_two - display_grid_one
        axs[2].set_title(layer_name + ' : diff = ' + class_names[0] + ' - ' + class_names[1])
        axs[2].imshow(avg_diff, aspect='auto', cmap=diff_cmap)

        avg_diff_alp = (avg_diff / np.max(np.abs(avg_diff))) * 4
        avg_diff = np.clip(avg_diff, 0, 255).astype(int)

        axs[3].set_title(layer_name + ' : diff added to ' + class_names[0])
        axs[3].imshow(display_grid_one, aspect='auto', alpha=1 - avg_diff_alp, cmap=main_cmap)
        axs[3].imshow(avg_diff, aspect='auto', alpha=avg_diff_alp, cmap='inferno')

        [axs[i].axis('off') for i in range(len(axs))]
        [axs[i].grid(False) for i in range(len(axs))]


def post_process_image(chan_img):
    # chan_img /= np.max(chan_img)
    # chan_img *= 255
    # remember: standardize image to have image position at centre of colormap, with std = 64 = (256/4)
    #  Hence, 128 +- (2 std) = 256 (from Deep Learning with Python book)
    chan_img -= chan_img.mean()
    chan_img = (chan_img / chan_img.std()) * 64 + 128
    return np.clip(chan_img, 0, 255).astype(int)


if __name__ == '__main__':
    # step: study of normal/pneumonia model

    norm_pneu_model = load_model('pneumonia_model_data/normal_and_pneumonia_small_1.h5')

    class_one_img_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/norm_pneu/',
        target_size=(150, 150), batch_size=1, class_mode='binary', color_mode='grayscale', classes=['normal'])

    class_two_img_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/norm_pneu/',
        target_size=(150, 150), batch_size=1, class_mode='binary', color_mode='grayscale', classes=['pneumonia'])

    num_layers = 3

    layer_outputs = [layer.output for layer in norm_pneu_model.layers[:num_layers]]
    activation_model = models.Model(inputs=norm_pneu_model.input, outputs=layer_outputs)
    class_names = ['normal', 'pneumonia']

    create_visualisations(norm_pneu_model, class_one_img_generator, class_two_img_generator, activation_model,
                          num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=50, digitize=False)
    plt.show()
    create_visualisations(norm_pneu_model, class_one_img_generator, class_two_img_generator, activation_model,
                          num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=50, digitize=True, num_bins=7)
    plt.show()
    create_visualisations(norm_pneu_model, class_one_img_generator, class_two_img_generator, activation_model,
                          num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=50, digitize=False, main_cmap='Paired')
    plt.show()

    ####################################################################################################################

    # substep: see if visualisations are possible from the original dataset

    num_samp = 30
    img_one = class_one_img_generator.next()[0][0, ..., 0]
    img_two = class_two_img_generator.next()[0][0, ..., 0]
    for _ in range(1, num_samp):
        img_one += class_one_img_generator.next()[0][0, ..., 0]
        img_two += class_two_img_generator.next()[0][0, ..., 0]

    img_one /= num_samp
    img_two /= num_samp

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(img_one, label=class_names[0])
    axs[1].imshow(img_two, label=class_names[1])

    [axs[i].legend() for i in range(len(axs))]
    plt.show()

    ####################################################################################################################

    # step: study of virus/bacteria model

    virus_bac_model = load_model('virus_model_data/virus_small_1.h5')

    class_one_img_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/virus_bacteria/',
        target_size=(150, 150), batch_size=1, class_mode='binary', color_mode='grayscale', classes=['virus'])

    class_two_img_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/virus_bacteria/',
        target_size=(150, 150), batch_size=1, class_mode='binary', color_mode='grayscale', classes=['bacteria'])

    num_layers = 2

    layer_outputs = [layer.output for layer in virus_bac_model.layers[:num_layers]]
    activation_model = models.Model(inputs=virus_bac_model.input, outputs=layer_outputs)

    class_names = ['virus', 'bacteria']
    create_visualisations(virus_bac_model, class_one_img_generator, class_two_img_generator, activation_model,
                          num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=10, digitize=False, num_bins=7)
    plt.show()
    create_visualisations(virus_bac_model, class_one_img_generator, class_two_img_generator, activation_model,
                          num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=10, digitize=True, num_bins=7)
    plt.show()
    create_visualisations(virus_bac_model, class_one_img_generator, class_two_img_generator, activation_model,
                          num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=10, digitize=False, num_bins=7, main_cmap='Paired')
    plt.show()

    ####################################################################################################################

    # substep: see if visualisations are possible from the original dataset

    num_samp = 30
    img_one = class_one_img_generator.next()[0][0, ..., 0]
    img_two = class_two_img_generator.next()[0][0, ..., 0]
    for _ in range(1, num_samp):
        img_one += class_one_img_generator.next()[0][0, ..., 0]
        img_two += class_two_img_generator.next()[0][0, ..., 0]

    img_one /= num_samp
    img_two /= num_samp

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(img_one, label=class_names[0])
    axs[1].imshow(img_two, label=class_names[1])

    [axs[i].legend() for i in range(len(axs))]
    plt.show()

    ####################################################################################################################
    # step: study of covid/non-covid (virus) model

    covid_model = load_model('covid_model_data/covid_small_1.h5')

    class_one_img_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/covid_noncovid/',
        target_size=(150, 150), batch_size=1, class_mode='binary', color_mode='grayscale', classes=['covid'])

    class_two_img_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'Coronahack-Chest-XRay-Dataset/test/covid_noncovid/',
        target_size=(150, 150), batch_size=1, class_mode='binary', color_mode='grayscale', classes=['non-covid'])

    num_layers = 1
    layer_outputs = [layer.output for layer in covid_model.layers[:num_layers]]
    activation_model = models.Model(inputs=covid_model.input, outputs=layer_outputs)

    class_names = ['covid', 'non-covid (virus)']
    create_visualisations(covid_model, class_one_img_generator, class_two_img_generator, activation_model, num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=30, digitize=False, num_bins=7)
    plt.show()
    create_visualisations(covid_model, class_one_img_generator, class_two_img_generator, activation_model, num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=30, digitize=True, num_bins=7)
    plt.show()
    create_visualisations(covid_model, class_one_img_generator, class_two_img_generator, activation_model, num_layers,
                          class_names=class_names,
                          num_cols=16, num_samples=30, digitize=False, num_bins=7, main_cmap='Paired')
    plt.show()

    ####################################################################################################################

    # substep: see if visualisations are possible from the original dataset

    num_samp = 30
    img_one = class_one_img_generator.next()[0][0,...,0]
    img_two = class_two_img_generator.next()[0][0,...,0]
    for _ in range(1, num_samp):
        img_one += class_one_img_generator.next()[0][0,...,0]
        img_two += class_two_img_generator.next()[0][0,...,0]

    img_one /= num_samp
    img_two /= num_samp

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(img_one, label=class_names[0])
    axs[1].imshow(img_two, label=class_names[1])

    [axs[i].legend() for i in range(len(axs))]
    plt.show()

    ####################################################################################################################