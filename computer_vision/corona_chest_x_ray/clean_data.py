import pandas as pd
import os, shutil
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('Coronahack-Chest-XRay-Dataset/Chest_xray_Corona_Metadata.csv')
    df: pd.DataFrame = df.sample(frac=1).reset_index(drop=True)  # remember: shuffle data

    print("df len:", len(df))

    data_dir = 'Coronahack-Chest-XRay-Dataset/data'
    dataset_dir = 'Coronahack-Chest-XRay-Dataset/'

    # create train/test/valid directories
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

    # step: remove directories if already created, allowing for re-creation.
    shutil.rmtree(train_dir)
    shutil.rmtree(valid_dir)
    shutil.rmtree(test_dir)

    os.mkdir(train_dir)
    os.mkdir(valid_dir)
    os.mkdir(test_dir)

    dirs = [train_dir, valid_dir, test_dir]

    # step: for each of train,test,dir create problem directories
    norm_pneu_dirs = [os.path.join(i, 'norm_pneu') for i in dirs]

    for i in range(3):
        os.mkdir(norm_pneu_dirs[i])
        os.mkdir(os.path.join(norm_pneu_dirs[i], 'normal'))
        os.mkdir(os.path.join(norm_pneu_dirs[i], 'pneumonia'))

    vir_nonvir_dirs = [os.path.join(i, 'virus_bacteria') for i in dirs]
    for i in range(3):
        os.mkdir(vir_nonvir_dirs[i])
        os.mkdir(os.path.join(vir_nonvir_dirs[i], 'virus'))
        os.mkdir(os.path.join(vir_nonvir_dirs[i], 'bacteria'))

    cov_noncov_dirs = [os.path.join(i, 'covid_noncovid') for i in dirs]
    for i in range(3):
        os.mkdir(cov_noncov_dirs[i])
        os.mkdir(os.path.join(cov_noncov_dirs[i], 'covid'))
        os.mkdir(os.path.join(cov_noncov_dirs[i], 'non-covid'))
    #################################################################################################################
    # step: determine length num samples for each problem (starting with covid), then split 50,25,25

    # step: create covid problem

    covid_mask = df['Label_2_Virus_category'] == 'COVID-19'
    covid_names = df[covid_mask]['X_ray_image_name'].to_numpy()
    non_covid_names = df[~covid_mask]['X_ray_image_name'].to_numpy()

    N = min(len(covid_names), len(non_covid_names))

    trainN, testN, = int(0.5 * N), int(0.25 * N)
    valid_N = N - (trainN + testN)
    print("covid/non-covid total:", N * 2)
    print("covid split (each):", trainN, testN, valid_N)

    # TRAIN:
    for c in covid_names[:trainN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(cov_noncov_dirs[0], 'covid', c)
        shutil.copy(img_src, img_des)

    for c in non_covid_names[:trainN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(cov_noncov_dirs[0], 'non-covid', c)
        shutil.copy(img_src, img_des)

    # TEST:
    for c in covid_names[trainN:trainN + testN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(cov_noncov_dirs[1], 'covid', c)
        shutil.copy(img_src, img_des)

    for c in non_covid_names[trainN:trainN + testN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(cov_noncov_dirs[1], 'non-covid', c)
        shutil.copy(img_src, img_des)

    # VALID:
    for c in covid_names[trainN + testN:N]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(cov_noncov_dirs[2], 'covid', c)
        shutil.copy(img_src, img_des)

    for c in non_covid_names[trainN + testN:N]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(cov_noncov_dirs[2], 'non-covid', c)
        shutil.copy(img_src, img_des)

    # step: drop covid/non-covid testset
    # remember: this will allow us to test the whole model together if we do this.
    covid_idxs = np.flatnonzero(covid_mask)[trainN + testN:N]
    noncovid_idxs = np.flatnonzero(~covid_mask)[trainN + testN:N]
    df = df.drop(np.r_[covid_idxs, noncovid_idxs]).reset_index(drop=True)
    print("remaining df len after covid set:", len(df))

    ###################################################################################################################

    # step: create norm,pneu problem
    normal_mask = df['Label'] == 'Normal'
    normal_names = df[normal_mask]['X_ray_image_name'].to_numpy()
    pneu_names = df[~normal_mask]['X_ray_image_name'].to_numpy()

    N = min(len(normal_names), len(pneu_names))

    trainN, testN, = int(0.5 * N), int(0.25 * N)
    valid_N = N - (trainN + testN)
    print("normal/pneu total:", N * 2)
    print("normal/pneu split (each):", trainN, testN, valid_N)

    # TRAIN:
    for c in normal_names[:trainN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(norm_pneu_dirs[0], 'normal', c)
        shutil.copy(img_src, img_des)

    for c in pneu_names[:trainN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(norm_pneu_dirs[0], 'pneumonia', c)
        shutil.copy(img_src, img_des)

    # # TEST:
    for c in normal_names[trainN:trainN + testN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(norm_pneu_dirs[1], 'normal', c)
        shutil.copy(img_src, img_des)

    for c in pneu_names[trainN:trainN + testN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(norm_pneu_dirs[1], 'pneumonia', c)
        shutil.copy(img_src, img_des)

    # # VALID:
    for c in normal_names[trainN + testN:N]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(norm_pneu_dirs[2], 'normal', c)
        shutil.copy(img_src, img_des)

    for c in pneu_names[trainN + testN:N]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(norm_pneu_dirs[2], 'pneumonia', c)
        shutil.copy(img_src, img_des)

    # step: drop all normals, but also just the pneumonia testset
    normal_idxs = np.flatnonzero(normal_mask)
    pneu_idxs = np.flatnonzero(~normal_mask)[trainN + testN:N]
    df = df.drop(np.r_[normal_idxs, pneu_idxs]).reset_index(drop=True)
    print("remaining df len after normal/pneu set:", len(df))
    ###################################################################################################################

    # step: create virus,non_virus problem
    virus_mask = df['Label_1_Virus_category'] == 'Virus'
    bacteria_mask = df['Label_1_Virus_category'] == 'Bacteria'
    virus_names = df[virus_mask]['X_ray_image_name'].to_numpy()
    bacteria_names = df[bacteria_mask]['X_ray_image_name'].to_numpy()

    N = min(len(virus_names), len(bacteria_names))

    trainN, testN, = int(0.5 * N), int(0.25 * N)
    valid_N = N - (trainN + testN)
    print("virus/bacteria total:", N * 2)
    print("virus/bacteria split (each):", trainN, testN, valid_N)

    # TRAIN:
    for c in virus_names[:trainN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(vir_nonvir_dirs[0], 'virus', c)
        shutil.copy(img_src, img_des)

    for c in bacteria_names[:trainN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(vir_nonvir_dirs[0], 'bacteria', c)
        shutil.copy(img_src, img_des)

    # # TEST:
    for c in virus_names[trainN:trainN + testN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(vir_nonvir_dirs[1], 'virus', c)
        shutil.copy(img_src, img_des)

    for c in bacteria_names[trainN:trainN + testN]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(vir_nonvir_dirs[1], 'bacteria', c)
        shutil.copy(img_src, img_des)

    # # VALID:
    for c in virus_names[trainN + testN:N]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(vir_nonvir_dirs[2], 'virus', c)
        shutil.copy(img_src, img_des)

    for c in bacteria_names[trainN + testN:N]:
        img_src = os.path.join(data_dir, c)
        img_des = os.path.join(vir_nonvir_dirs[2], 'bacteria', c)
        shutil.copy(img_src, img_des)

    # step: drop virus/non-virus rows (of the test portion)
    virus_idxs = np.flatnonzero(virus_mask)[trainN + testN:N]
    bacteria_idxs = np.flatnonzero(bacteria_mask)[trainN + testN:N]
    df = df.drop(np.r_[virus_idxs, bacteria_idxs]).reset_index(drop=True)
    print("remaining df len after virus set:", len(df))
