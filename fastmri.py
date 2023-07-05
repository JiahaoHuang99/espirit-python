import os
import h5py
import cv2
from glob import glob
import numpy as np
import math
import imageio
from scipy.fftpack import *


def process_h5(data_path,
               save_master_path,
               track,
               data_type='kspace',
               is_write=False,
               is_crop_to_320=False,
               is_normalise=False,
               mask=None,):

    # get basic information
    data_name = get_file_name_from_path(data_path)

    if data_type == 'kspace':

        # read data
        with h5py.File(data_path) as hf:
            # print('Keys:', list(hf.keys()))
            # print('Attrs:', dict(hf.attrs))
            # print(dict(hf.attrs)['acquisition'])
            kspace_data = hf['kspace'][()]  # (slices, coils, h, w) or (slices, h, w)
            reconstruction_rss = hf['reconstruction_rss'][()]  # (slices, h, w)
            ismrmrd_header = hf["ismrmrd_header"][()]
            acquisition = dict(hf.attrs)['acquisition']

            if track == 'singlecoil':
                slices, h, w = kspace_data.shape
            elif track == 'multicoil':
                slices, coils, h, w = kspace_data.shape
            else:
                raise ValueError

        # slice selection
        kspace_data = kspace_data[slices // 2 - 10: slices // 2 + 10, ...]
        reconstruction_rss = reconstruction_rss[slices // 2 - 10: slices // 2 + 10, ...]

        # update slices
        slices = kspace_data.shape[0]

        under_kspace_list = []
        img_rss_list = []
        # loop for slice
        for slice_idx in range(slices):

            kspace_slice = kspace_data[slice_idx, ...]
            reconstruction_rss_slice = reconstruction_rss[slice_idx, ...]

            # reminder: when singlecoil track, img_slice here is NOT equal to reconstruction_rss_slice!
            img_slice = fftshift(ifftn(ifftshift(kspace_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

            # normalisation
            # In this version, No normalisation is applied here.

            # crop
            if is_crop_to_320:
                img_slice = img_slice[..., (h // 2 - 160):(h // 2 + 160), (w // 2 - 160):(w // 2 + 160)]
                assert img_slice.shape[-1] == 320 and img_slice.shape[-2] == 320
            else:
                img_slice = img_slice[..., (h // 2 - 320):(h // 2 + 320), (w // 2 - 160):(w // 2 + 160)]
                assert img_slice.shape[-1] == 320 and img_slice.shape[-2] == 640

            assert reconstruction_rss_slice.shape[-1] == 320 and reconstruction_rss_slice.shape[-2] == 320

            if is_normalise:
                img_slice = preprocess_normalisation(img_slice, type='complex')

            # undersampled k-space
            kspace_slice = fftshift(fftn(ifftshift(img_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

            under_kspace_slice = kspace_slice * mask[np.newaxis, ...]
            under_kspace_list.append(under_kspace_slice)

            img_slice_norm = preprocess_normalisation(img_slice, type='complex')
            img_slice_rss = root_sum_of_squares(abs(img_slice_norm), dim=0)
            img_rss_list.append(img_slice_rss)

        under_kspace = np.stack(under_kspace_list, axis=0)  # (slices, coils, h, w)
        img_rss = np.stack(img_rss_list, axis=0)  # (slices, h, w)

        if is_write:
            mkdir(os.path.join(save_master_path, 'h5raw',))
            with h5py.File(os.path.join(save_master_path, 'h5raw', '{}.h5'.format(data_name)), 'w') as f:
                f['kspace'] = under_kspace  # (slices, coils, h, w)
                f['mask'] = mask[0, :]  # (h,)
                f['reconstruction_rss'] = img_rss  # (slices, h, w)
                f['ismrmrd_header'] = ismrmrd_header
                # f.attrs['max'] = 1.0
                f.attrs['max'] = abs(img_rss).max()

    elif data_type == 'image_rss':
        raise NotImplementedError
    elif data_type == 'image_esc':
        raise NotImplementedError
    else:
        raise ValueError

if __name__ == '__main__':