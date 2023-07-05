import cfl
from espirit import espirit, espirit_proj, ifft, fft

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass




if __name__ == '__main__':
    # Configurations
    is_load_data_temp = False
    data_temp_path = 'fastmri_result/tmp_crop'
    # data_temp_path = 'fastmri_result/tmp'

    # Load data
    data_path = 'data/fastMRI/file1000267.h5'
    with h5py.File(data_path) as hf:
        kspace_data = hf['kspace'][()]  # (slices, coils, h, w) or (slices, h, w)

    num_coils = kspace_data.shape[1]

    # Transpose to (slices, h, w, coils)
    kspace_data = kspace_data.transpose(0, 2, 3, 1)

    # Get slice
    X = kspace_data[6:7, ...]
    x = ifft(X, (1, 2))

    # image space center crop
    # H, W --> 320, 320
    h = x.shape[1]
    w = x.shape[2]
    x = x[:, int(h / 2 - 160):int(h / 2 + 160), int(w / 2 - 160):int(w / 2 + 160), :]
    X = fft(x, (1, 2))

    if not is_load_data_temp:
        # Derive ESPIRiT operator
        esp = espirit(X=X, k=6, r=24, t=0.01, c=0.9925)

        # Do projections
        ip, proj, null = espirit_proj(x, esp)

        # Save temp data
        mkdir(data_temp_path)
        np.save(os.path.join(data_temp_path, 'esp.npy'), esp)
        np.save(os.path.join(data_temp_path, 'x.npy'), x)
        np.save(os.path.join(data_temp_path, 'ip.npy'), ip)
        np.save(os.path.join(data_temp_path, 'proj.npy'), proj)
        np.save(os.path.join(data_temp_path, 'null.npy'), null)

    else:
        esp = np.load(os.path.join(data_temp_path, 'esp.npy'))
        x = np.load(os.path.join(data_temp_path, 'x.npy'))
        ip = np.load(os.path.join(data_temp_path, 'ip.npy'))
        proj = np.load(os.path.join(data_temp_path, 'proj.npy'))
        null = np.load(os.path.join(data_temp_path, 'null.npy'))


    # Figure code
    esp = np.squeeze(esp)  # (H, W, coils, coils)
    x = np.squeeze(x)  # (H, W, coils)
    ip = np.squeeze(ip)  # (H, W, coils)
    proj = np.squeeze(proj)  # (H, W, coils)
    null = np.squeeze(null)  # (H, W, coils)

    print("Close figures to continue execution...")



    for idx in range(num_coils):
        plt.imsave(os.path.join(data_temp_path, 'esp_{}.png'.format(idx)), np.abs(esp[:, :, idx, 0]), cmap='gray')
        plt.imsave(os.path.join(data_temp_path, 'x_{}.png'.format(idx)), np.abs(x[:, :, idx]), cmap='gray')
        plt.imsave(os.path.join(data_temp_path, 'ip_{}.png'.format(idx)), np.abs(ip[:, :, idx]), cmap='gray')
        plt.imsave(os.path.join(data_temp_path, 'proj_{}.png'.format(idx)), np.abs(proj[:, :, idx]), cmap='gray')
        plt.imsave(os.path.join(data_temp_path, 'null_{}.png'.format(idx)), np.abs(null[:, :, idx]), cmap='gray')














    # # Display ESPIRiT operator
    # for idx in range(num_coils):
    #     for jdx in range(num_coils):
    #         plt.subplot(num_coils, num_coils, (idx * num_coils + jdx) + 1)
    #         plt.imshow(np.abs(esp[:, :, idx, jdx]), cmap='gray',)
    #         plt.axis('off')
    #
    # plt.show(dpi=1000)
    #
    # dspx = np.power(np.abs(np.concatenate((x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3], x[:, :, 4], x[:, :, 5], x[:, :, 6], x[:, :, 7]), axis=0)), 1 / 3)
    # dspip = np.power(np.abs(np.concatenate((ip[:, :, 0], ip[:, :, 1], ip[:, :, 2], ip[:, :, 3], ip[:, :, 4], ip[:, :, 5], ip[:, :, 6], ip[:, :, 7]), axis=0)), 1 / 3)
    # dspproj = np.power(np.abs(np.concatenate((proj[:, :, 0], proj[:, :, 1], proj[:, :, 2], proj[:, :, 3], proj[:, :, 4], proj[:, :, 5], proj[:, :, 6], proj[:, :, 7]), axis=0)), 1 / 3)
    # dspnull = np.power(np.abs(np.concatenate((null[:, :, 0], null[:, :, 1], null[:, :, 2], null[:, :, 3], null[:, :, 4], null[:, :, 5], null[:, :, 6], null[:, :, 7]), axis=0)), 1 / 3)
    #
    # print("NOTE: Contrast has been changed")
    #
    # # Display ESPIRiT projection results
    # plt.subplot(1, 4, 1)
    # plt.imshow(dspx, cmap='gray')
    # plt.title('Data')
    # plt.axis('off')
    # plt.subplot(1, 4, 2)
    # plt.imshow(dspip, cmap='gray')
    # plt.title('Inner product')
    # plt.axis('off')
    # plt.subplot(1, 4, 3)
    # plt.imshow(dspproj, cmap='gray')
    # plt.title('Projection')
    # plt.axis('off')
    # plt.subplot(1, 4, 4)
    # plt.imshow(dspnull, cmap='gray')
    # plt.title('Null Projection')
    # plt.axis('off')
    # plt.show(dpi=1000)
    #
    #

