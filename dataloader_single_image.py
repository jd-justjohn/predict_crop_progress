import tensorflow as tf
import tensorflow_io as tfio


def get_dataset(files_in, shuffle_buffer=1000, num_bands=12, shuffle_files=True, min_season_length=60, max_season_length=360):

    means_ = tf.constant([6.0327573, 6.290839, 6.691273, 6.63656, 7.116921, 7.788275,
       7.977958, 8.02139, 8.070433, 8.088675, 7.84131, 7.389418], tf.float32)

    stdevs_ = tf.constant([0.81428736, 0.63173527, 0.47528076, 0.7584638, 0.48651397,
       0.35625926, 0.41182348, 0.38492793, 0.37940097, 0.3752155,
       0.3995582, 0.5896098], tf.float32)

    ############################################ read from s3 ##########################################################
    # filenames = tf.data.Dataset.list_files('s3://jd-us01-isg-analytics-data-bricks/user/jj99886/time_series_paaw_imgs_joined_parquet/year=2019/quad_key=023111101/FieldOperationGuid_seeding=0236483a-79bc-421e-8bb5-5247a133e393/*.parquet', shuffle=shuffle_files)
    #################### read from local/EBS (recommended!  Much faster than s3) #######################################
    # filenames = tf.data.Dataset.list_files('/home/ubuntu/Document/stime_series_paaw_imgs_joined_parquet/**/*.parquet')
    ############################################ glob local files #################################################################
    # files_in = glob.glob('/home/ubuntu/Document/stime_series_paaw_imgs_joined_parquet/**/*.parquet', recursive=True)
    if shuffle_files:
        filenames = tf.data.Dataset.from_tensor_slices(files_in).shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    else:
        filenames = tf.data.Dataset.from_tensor_slices(files_in)
    def parquet_ds(file):
        ds = tfio.IODataset.from_parquet(file,
                                         {'seeding_date': tf.int32,
                                          'harvest_date': tf.int32,
                                          'L14_RSum_harvest': tf.float64,
                                          'hex_index_L12': tf.string,
                                          # 'n_epochs_seeding': tf.int32,
                                          # 'n_epochs_harvest': tf.int32,
                                          # 'mean_YieldVolumePerArea_bu_per_ha': tf.float64,
                                          'mean_HarvestMoisture_prcnt': tf.float64,
                                          'num_images': tf.int64,
                                          'bands': tf.string,
                                          'scene_ids': tf.string,
                                          # 'tiles': tf.string,
                                          # 'img_dates_int': tf.string
                                          'img_dates': tf.string,
                                          'scl_vals': tf.string
                                          })
        return ds

    ds = filenames.interleave(parquet_ds,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              deterministic=False)

    def parse(row):
        tf.assert_greater(row['L14_RSum_harvest'], tf.constant(0.7, tf.float64))
        row['seeding_date'] = tf.cast(tf.cast(row['seeding_date'], tf.float32)/24., tf.int32)
        row['harvest_date'] = tf.cast(tf.cast(row['harvest_date'], tf.float32)/24., tf.int32)
        tf.assert_greater(row['harvest_date'] - row['seeding_date'], tf.constant(min_season_length, tf.int32))
        tf.assert_less(row['harvest_date'] - row['seeding_date'], tf.constant(max_season_length, tf.int32))

        split_scenes = tf.strings.split(tf.reshape(row['scene_ids'],[-1]),',')[0]

        ################### remove duplicate scenes by randomly choosing one of the duplicates #########################
        _, _, scene_counts = tf.unique_with_counts(split_scenes)
        unique_idxs = tf.concat([[0],tf.cumsum(scene_counts)[:-1]],axis=0)

        def rand_sample(maxval):
            return tf.random.uniform(shape=(), minval=0, maxval=maxval, dtype=tf.int32)

        unique_idxs = unique_idxs + tf.map_fn(fn=rand_sample, elems=scene_counts, fn_output_signature=tf.int32)

        ###################### constrain to crop season only ###########################################################
        def convert_bytes_dates(idx):
            # img_date_bytes = tf.strings.substr(row['img_dates_int'],idx*2,2)  # %band_lengths guarantees idx is in range
            img_date_bytes = tf.strings.substr(row['img_dates'],idx*2,2)  # %band_lengths guarantees idx is in range
            img_date_ints = tf.io.decode_raw(img_date_bytes, out_type=tf.uint16, little_endian=False, fixed_length=2)
            return tf.cast(img_date_ints, tf.int32)

        # eliminate snow pixels
        scl_vals = tf.gather(tf.io.decode_raw(row['scl_vals'], out_type=tf.uint8), unique_idxs)
        unique_idxs = tf.gather(unique_idxs, tf.squeeze(tf.where(scl_vals != 11)))

        img_dates = tf.map_fn(fn=convert_bytes_dates, elems=unique_idxs, fn_output_signature=tf.int32)
        crop_season_mask = tf.reshape(tf.math.logical_and(img_dates < row['harvest_date'], img_dates > row['seeding_date']), shape=(-1,))
        unique_idxs = tf.boolean_mask(unique_idxs, crop_season_mask)
        img_dates = tf.boolean_mask(img_dates, crop_season_mask)
        tf.Assert(len(unique_idxs) > 0, [1])

        #TODO get date of the randomly selected image and calculate %progress through crop season to see if certain periods are more accurate than others by crop

        def convert_bytes(idx):
            random_img_bytes = tf.strings.substr(row['bands'],idx*num_bands*2,num_bands*2)  # %band_lengths guarantees idx is in range
            random_img_ints = tf.io.decode_raw(random_img_bytes, out_type=tf.uint16, little_endian=False, fixed_length=24)
            tf.assert_greater(random_img_ints, tf.constant(0, tf.uint16))
            return (tf.math.log(tf.cast(random_img_ints, tf.float32)) - means_) / stdevs_

        # chosen_idx = tf.random.shuffle(unique_idxs)[:1]
        chosen_idx = tf.random.uniform((1,), minval=0, maxval=len(unique_idxs), dtype=tf.int32)[0]
        band_vals = tf.cast(tf.reshape(convert_bytes(unique_idxs[chosen_idx]), [-1]), tf.float32)
        # band_vals = tf.map_fn(fn=convert_bytes, elems=chosen_idx, fn_output_signature=tf.float32)

        # band_vals = tf.concat([tf.reshape(band_vals, [-1]), tf.cast(row['harvest_date'] - img_dates[-1], tf.float32)], axis=0)

        # return band_vals, (tf.cast(img_dates[-1], tf.float32) - tf.cast(row['seeding_date'], tf.float32))/(tf.cast(row['harvest_date'], tf.float32) - tf.cast(row['seeding_date'], tf.float32))#, tf.cast(img_dates[-1] - row['seeding_date'], tf.float32)#, tf.cast(row['mean_HarvestMoisture_prcnt'], tf.float32)
        # return band_vals, tf.cast(row['mean_HarvestMoisture_prcnt'], tf.float32), tf.cast(row['harvest_date'] - img_dates[-1], tf.float32)
        return tf.squeeze(band_vals), (tf.cast(row['harvest_date'] - tf.squeeze(img_dates[chosen_idx]), tf.float32))/(tf.cast(row['harvest_date'], tf.float32) - tf.cast(row['seeding_date'], tf.float32))

    ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).apply(tf.data.experimental.ignore_errors())

    return ds