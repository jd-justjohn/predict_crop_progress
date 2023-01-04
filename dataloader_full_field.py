import os
import tensorflow as tf
import tensorflow_io as tfio
from functools import partial
import tensorflow_probability as tfp

parquet_dict = {'seeding_date': tf.int32,
                'harvest_date': tf.int32,
                'L14_RSum_harvest': tf.float64,
                # 'hex_index_L12': tf.string,
                # 'n_epochs_seeding': tf.int32,
                # 'n_epochs_harvest': tf.int32,
                'mean_YieldVolumePerArea_bu_per_ha': tf.float64,
                # 'mean_HarvestMoisture_prcnt': tf.float64,
                'num_images': tf.int64,
                'bands': tf.string,
                'scene_ids': tf.string,
                # 'tiles': tf.string,
                # 'img_dates_int': tf.string
                'img_dates': tf.string,
                'scl_vals': tf.string
                }

# batch_size=5
# num_bands=12
# shuffle_buffer=200000
# crop_season_only=False
# predict_on_N_imgs=50
# perc_field_fill=0.8
# min_num_hexes_per_field=50
# parametric=False

def get_dataset(dir_paths, batch_size, num_bands=12, shuffle_buffer=200000, crop_season_only=False, predict_on_N_imgs=5, perc_field_fill=0.8,
                min_num_hexes_per_field=50, parametric=True):
    '''
    :param dir_path: path of the field op guid partition
    :param max_imgs: Must match the model input size
    :param num_bands: sentinel 2 = 12 bands
    :param flatten_output: True for dense models
    :param get_full_timseries_output: only for investigations
    :param predict_on_N_imgs: number of images to use for prediction.  Like max_imgs, it must match the model training
    :return:
    '''

    def parse(row, fop):
        tf.assert_greater(row['L14_RSum_harvest'], tf.constant(0.7, tf.float64))

        split_scenes = tf.strings.split(tf.reshape(row['scene_ids'], [-1]), ',')[0]
        # if tf.strings.length(split_scenes[-1]) != 60: split_scenes = split_scenes[:-1]

        ################### remove duplicate scenes by randomly choosing one of the duplicates #########################
        _, _, scene_counts = tf.unique_with_counts(split_scenes)
        unique_idxs = tf.concat([[0], tf.cumsum(scene_counts)[:-1]], axis=0)

        def rand_sample(maxval):
            return tf.random.uniform(shape=(), minval=0, maxval=maxval, dtype=tf.int32)

        unique_idxs = unique_idxs + tf.map_fn(fn=rand_sample, elems=scene_counts, fn_output_signature=tf.int32)

        ###################### constrain to crop season only ###########################################################
        def convert_bytes_dates(idx):
            # img_date_bytes = tf.strings.substr(row['img_dates_int'],idx*2,2)  # %band_lengths guarantees idx is in range
            img_date_bytes = tf.strings.substr(row['img_dates'], idx * 2, 2)  # %band_lengths guarantees idx is in range
            img_date_ints = tf.io.decode_raw(img_date_bytes, out_type=tf.uint16, little_endian=False, fixed_length=2)
            return tf.cast(img_date_ints, tf.int32)

        img_dates = tf.map_fn(fn=convert_bytes_dates, elems=unique_idxs, fn_output_signature=tf.int32)
        row['seeding_date'] = tf.cast(tf.cast(row['seeding_date'], tf.float32) / 24., tf.int32)
        row['harvest_date'] = tf.cast(tf.cast(row['harvest_date'], tf.float32) / 24., tf.int32)
        if crop_season_only:
            crop_season_mask = tf.reshape(tf.math.logical_and(img_dates < row['harvest_date'], img_dates > row['seeding_date']), shape=(-1,))
            unique_idxs = tf.boolean_mask(unique_idxs, crop_season_mask)

        tf.Assert(len(unique_idxs) > 0, [1])
        r_samp_idxs = unique_idxs
        ## exclude snow-covered pixels
        scl_vals = tf.gather(tf.io.decode_raw(row['scl_vals'], out_type=tf.uint8), r_samp_idxs)
        r_samp_idxs = tf.gather(r_samp_idxs, tf.squeeze(tf.where(scl_vals != 11)))
        scl_vals = tf.gather(scl_vals, r_samp_idxs)

        img_dates = tf.map_fn(fn=convert_bytes_dates, elems=r_samp_idxs, fn_output_signature=tf.int32)

        def convert_bytes(idx):
            random_img_bytes = tf.strings.substr(row['bands'], idx * num_bands * 2, num_bands * 2)  # %band_lengths guarantees idx is in range
            random_img_ints = tf.io.decode_raw(random_img_bytes, out_type=tf.uint16, little_endian=False, fixed_length=24)
            return tf.cast(random_img_ints, tf.float32)

        band_vals = tf.map_fn(fn=convert_bytes, elems=r_samp_idxs, fn_output_signature=tf.float32)

        def convert_bytes_NDVI(idx):
            RED = tf.cast(tf.io.decode_raw(tf.strings.substr(row['bands'], idx * num_bands * 2 + 6, 2), out_type=tf.uint16,
                                           little_endian=False, fixed_length=2), tf.float32)
            NIR = tf.cast(tf.io.decode_raw(tf.strings.substr(row['bands'], idx * num_bands * 2 + 14, 2), out_type=tf.uint16,
                                 little_endian=False, fixed_length=2), tf.float32)
            return (NIR - RED) / (NIR + RED)

        NDVI = tf.map_fn(fn=convert_bytes_NDVI, elems=r_samp_idxs, fn_output_signature=tf.float32)
        return band_vals, tf.cast(row['mean_YieldVolumePerArea_bu_per_ha'], tf.float32), tf.squeeze(NDVI), tf.squeeze(
            img_dates), row['seeding_date'], row['harvest_date'], fop

    def image_selector_proc_fn(*input_val):
        band_vals, y, NDVI_vals, img_dates, seeding_dates, harvest_dates, fop = input_val
        tf.Assert(tf.shape(band_vals)[0] > min_num_hexes_per_field, [1])
        unique_cnts = tf.unique_with_counts(tf.reshape(img_dates, [-1]))
        unique_dates = tf.sort(tf.boolean_mask(unique_cnts.y, tf.logical_and(unique_cnts.y > 0, unique_cnts.count / tf.shape(img_dates)[0] > perc_field_fill)))
        harvest_date = tf.unique_with_counts(tf.reshape(harvest_dates, [-1]))
        harvest_date = harvest_date.y[tf.argmax(harvest_date.count)]
        seeding_date = tf.unique_with_counts(tf.reshape(seeding_dates, [-1]))
        seeding_date = seeding_date.y[tf.argmax(seeding_date.count)]

        ### tf.map_fn each "column" (date) of NDVI values to get R**2 from linear_reg (only for in-season dates -- out-of-season label as zero)
        def linear_reg(date_val):
            NDVI = tf.reshape(tf.gather_nd(NDVI_vals, tf.where(img_dates == date_val)), [-1, 1])
            y_ = tf.reshape(tf.gather_nd(y, tf.where(img_dates == date_val)[:, 0:1]), [-1, 1])
            lst_sq_fit = tf.linalg.lstsq(tf.concat([tf.ones_like(NDVI), NDVI], axis=1), y_)
            y_pred_lst_sq_corrected = tf.matmul(tf.concat([tf.ones_like(NDVI), NDVI], axis=1), lst_sq_fit)
            unexplained_error = tf.reduce_sum(tf.square(y_ - y_pred_lst_sq_corrected))
            total_error = tf.reduce_sum(tf.square(y_ - tf.reduce_mean(y_)))
            R_squared = 1. - unexplained_error / total_error
            band_vals_on_date = tf.gather_nd(band_vals, tf.where(img_dates == date_val))
            if parametric:
                return R_squared, tf.concat([tf.reduce_mean(band_vals_on_date, axis=0), tf.math.reduce_std(band_vals_on_date, axis=0)], axis=0)
            else:
                # quants = tfp.stats.percentile(band_vals_on_date, [25, 50, 75], axis=0)
                # return R_squared, (tf.concat([quants[1,:], quants[-1,:] - quants[0,:]], axis=0) - non_par_means)/non_par_stdevs
                return R_squared, tf.reshape(tfp.stats.percentile(band_vals_on_date, [5, 25, 50, 75, 95], axis=0), [-1])

        Rsq_vals, band_vals_ret_map = tf.map_fn(linear_reg, unique_dates, fn_output_signature=(tf.float32,tf.float32))

        Rsq_vals = tf.where(tf.math.is_nan(Rsq_vals), tf.zeros_like(Rsq_vals), Rsq_vals)
        rsq_max = tf.reduce_max(Rsq_vals)

        r_samp_len = tf.minimum(predict_on_N_imgs, len(unique_dates))
        r_samp_dates = tf.sort(tf.random.shuffle(unique_dates)[:r_samp_len])
        r_samp_idxs = tf.where(unique_dates == tf.reshape(r_samp_dates, [-1, 1]))[:, -1]
        gt_rsq = tf.gather(Rsq_vals, tf.where(unique_dates == tf.reshape(r_samp_dates, [-1, 1]))[:, -1])

        band_vals_ret_map_out = tf.gather_nd(band_vals_ret_map, tf.reshape(r_samp_idxs, [-1, 1]))

        return band_vals_ret_map_out, gt_rsq, rsq_max, (tf.cast(harvest_date, tf.float32) - tf.squeeze(tf.cast(r_samp_dates, tf.float32)))/(tf.cast(harvest_date, tf.float32) - tf.cast(seeding_date, tf.float32))

    def parquet_ds(field_op_dir):
        ds = tf.data.Dataset.list_files(field_op_dir + '/*.parquet')
        ds = ds.interleave(lambda x: tfio.IODataset.from_parquet(x, parquet_dict), num_parallel_calls=None, deterministic=False)
        ds = ds.map(partial(parse, fop=tf.strings.split(field_op_dir, 'FIELD_OPERATION_GUID=')[-1]), num_parallel_calls=None)
        return ds.apply(tf.data.experimental.ignore_errors()).padded_batch(10000, ([None, num_bands], [], [None], [None], [], [], []))

    ds = tf.data.Dataset.from_tensor_slices(dir_paths).shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.interleave(parquet_ds, num_parallel_calls=2, deterministic=False)
    ds = ds.map(image_selector_proc_fn, num_parallel_calls=int(os.cpu_count()/4)+1).apply(tf.data.experimental.ignore_errors())
    if parametric:
        ds = ds.padded_batch(batch_size, padded_shapes=([None, 2*num_bands], [None], [], [None]), padding_values=-1.)
    else:
        ds = ds.padded_batch(batch_size, padded_shapes=([None, 5*num_bands], [None], [], [None]), padding_values=-1.)

    return ds
