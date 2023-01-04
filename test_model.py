from dataloader_full_field import get_dataset
import os
import tensorflow as tf
import normalizing_functions
import numpy as np

crop = ['corn', 'soy', 'winter_wheat']

lowest_dirs_corn = list()
top_level_dir_path = '/home/ubuntu/Downloads/corn_hexes_and_imgery_latest_small_files'
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs_corn.append(root)

lowest_dirs_soy = list()
top_level_dir_path = '/home/ubuntu/Downloads/soy_hexes_and_imgery_latest_small_files'
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs_soy.append(root)

lowest_dirs_wheat = list()
top_level_dir_path = '/home/ubuntu/Downloads/wheat_hexes_and_imagery'
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs_wheat.append(root)

lowest_dirs_list = [lowest_dirs_corn, lowest_dirs_soy, lowest_dirs_wheat]

train_files = [[x for x in yy if ('seeding_date_min_by_fop=2020' in x or 'seeding_date_min_by_fop=2019' in x)] for yy in lowest_dirs_list]
val_files = [[x for x in yy if 'seeding_date_min_by_fop=2021' in x] for yy in lowest_dirs_list]

num_bands=12
shuffle_buffer=200000
batch_size = 35
crop_season_only = True
predict_on_N_imgs = 20
perc_field_fill=0.8
min_num_hexes_per_field=50
parametric=False

train_ds_list = [get_dataset(yy, shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, predict_on_N_imgs=predict_on_N_imgs,
                       num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field, parametric=parametric).prefetch(buffer_size=tf.data.AUTOTUNE) for yy in train_files]
val_ds_list = [get_dataset(yy,shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, predict_on_N_imgs=100,
                     num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field, parametric=parametric).prefetch(buffer_size=tf.data.AUTOTUNE) for yy in val_files]

train_ds = tf.data.Dataset.zip(tuple(train_ds_list))
val_ds = tf.data.Dataset.zip(tuple(val_ds_list))

def prep_batch(x_in, y_in):
    y_in = tf.reshape(y_in, shape=[-1, 1])
    x_in = tf.reshape(x_in, shape=[-1, *tf.shape(x_in)[2:]])
    y_not_zero = tf.not_equal(y_in, -1)
    y_in = tf.gather(y_in, tf.where(y_not_zero)[:, 0], axis=0).numpy()
    x_in = tf.gather(x_in, tf.where(y_not_zero)[:, 0], axis=0).numpy()
    y_in = y_in[np.all(x_in > 0, axis=1), :]
    x_in = x_in[np.all(x_in > 0, axis=1), :]
    x_in, y = normalizing_functions.normalize(x_in, y_in, 'all_crops')
    return x_in, tf.convert_to_tensor(y_in)

model = tf.keras.models.load_model('/home/ubuntu/Documents/keras_model_training/tensorboard/regression/days_to_harvest/20230103-023726/my_model')

tt = val_ds.as_numpy_iterator().next()
batch_prepped = [prep_batch(xx[0], xx[-1]) for xx in tt]
b_lens = np.cumsum([0] + [len(xx[0]) for xx in batch_prepped])
x = tf.concat([xx[0] for xx in batch_prepped], axis=0)
y = tf.concat([xx[-1] for xx in batch_prepped], axis=0)