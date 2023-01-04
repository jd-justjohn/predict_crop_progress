from dataloader_single_image import get_dataset
import glob
import tensorflow as tf
import datetime
import numpy as np
from sklearn.metrics import r2_score
import os
import collections

crop = ['corn', 'soy', 'winter_wheat']

files_wheat = glob.glob('/home/ubuntu/Downloads/wheat_hexes_and_imagery/**/*.parquet', recursive=True)
files_corn = glob.glob('/home/ubuntu/Downloads/corn_hexes_and_imgery_latest_small_files/**/*.parquet', recursive=True)
files_soy = glob.glob('/home/ubuntu/Downloads/soy_hexes_and_imgery_latest_small_files/**/*.parquet', recursive=True)

files_list = [(files_corn, 100, 225), (files_soy, 90, 225), (files_wheat, 175, 360)]

train_files = [([x for x in yy[0] if ('seeding_date_min_by_fop=2020' in x or 'seeding_date_min_by_fop=2019' in x)], yy[1], yy[2]) for yy in files_list]
val_files = [([x for x in yy[0] if 'seeding_date_min_by_fop=2021' in x], yy[1], yy[2]) for yy in files_list]

num_bands=12
train_ds = [get_dataset(yy[0],shuffle_buffer=10000000, num_bands=num_bands, min_season_length=yy[1], max_season_length=yy[2]).batch(1000).prefetch(buffer_size=tf.data.AUTOTUNE) for yy in train_files]
val_ds = [get_dataset(yy[0],shuffle_buffer=10000000, num_bands=num_bands, min_season_length=yy[1], max_season_length=yy[2]).batch(3500).prefetch(buffer_size=tf.data.AUTOTUNE) for yy in val_files]


train_ds = tf.data.Dataset.zip(tuple(train_ds))
val_ds = tf.data.Dataset.zip(tuple(val_ds))

inputs = tf.keras.Input(shape=((num_bands),))
dense = tf.keras.layers.Dense(6, activation="swish")(inputs) #selu, mish
outputs = tf.keras.layers.Dense(1)(dense)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="out")

optimizer = tf.keras.optimizers.Adam(0.02)

##################################### tensorboard ####################################
##################################### tensorboard ####################################
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = '/home/ubuntu/Documents/keras_model_training/tensorboard/regression/best_img_r2/' + current_time + '/train'
train_log_dir_list = ['/home/ubuntu/Documents/keras_model_training/tensorboard/regression/days_to_harvest/' + current_time + '/train_' + s for s in crop]
# val_log_dir = '/home/ubuntu/Documents/keras_model_training/tensorboard/regression/best_img_r2/' + current_time + '/val'
val_log_dir_list = ['/home/ubuntu/Documents/keras_model_training/tensorboard/regression/days_to_harvest/' + current_time + '/val_' + s for s in crop]

train_summary_writer_list = [tf.summary.create_file_writer(s) for s in train_log_dir_list]
val_summary_writer_list = [tf.summary.create_file_writer(s) for s in val_log_dir_list]
######################################################################################
######################################################################################

loss_scores_queue = collections.deque(10*[1000], 10)
best_loss = .22
for epoch in range(500):
    for tt in train_ds:
        batch_prepped = [(xx[0], xx[1]) for xx in tt]
        b_lens = np.cumsum([0] + [len(xx[0]) for xx in batch_prepped])
        x = tf.concat([xx[0] for xx in batch_prepped], axis=0)
        y = tf.reshape(tf.concat([xx[1] for xx in batch_prepped], axis=0), [-1,1])

        with tf.GradientTape() as tape:
            y_pred = model(tf.concat(x, axis=0))
            loss = tf.math.sqrt(tf.reduce_mean((y - y_pred) ** 2))
            # loss = tf.reduce_mean(tf.math.abs(y - y_pred))

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        for i, sw in enumerate(train_summary_writer_list):
            with sw.as_default():
                tf.summary.scalar('loss', tf.math.sqrt(tf.reduce_mean((y[b_lens[i]:b_lens[i+1],:] - y_pred[b_lens[i]:b_lens[i+1], :]) ** 2)), step=optimizer.iterations)
                tf.summary.scalar('r2', r2_score(y[b_lens[i]:b_lens[i+1],:].numpy(), y_pred[b_lens[i]:b_lens[i+1],:].numpy()), step=optimizer.iterations)

        if optimizer.iterations % 100 == 0:
            ## validation
            tt = val_ds.as_numpy_iterator().next()
            batch_prepped = [(xx[0], xx[1]) for xx in tt]
            b_lens = np.cumsum([0] + [len(xx[0]) for xx in batch_prepped])
            x = tf.concat([xx[0] for xx in batch_prepped], axis=0)
            y = tf.reshape(tf.concat([xx[1] for xx in batch_prepped], axis=0), [-1,1])

            y_pred = model(tf.concat(x, axis=0))
            loss = tf.math.sqrt(tf.reduce_mean((y - y_pred) ** 2))
            # loss = tf.reduce_mean(tf.math.abs(y - y_pred))

            loss_scores_queue.appendleft(loss.numpy())
            if np.mean(loss_scores_queue) < best_loss:
                model.save(os.path.split(train_log_dir_list[0])[0] + '/my_model')
                best_loss = np.mean(loss_scores_queue)
            r2_score_val = r2_score(y.numpy(), y_pred.numpy())
            for i, sw in enumerate(val_summary_writer_list):
                with sw.as_default():
                    tf.summary.scalar('loss', tf.math.sqrt(tf.reduce_mean((y[b_lens[i]:b_lens[i + 1], :] - y_pred[b_lens[i]:b_lens[i + 1], :]) ** 2)),step=optimizer.iterations)
                    tf.summary.scalar('r2', r2_score(y[b_lens[i]:b_lens[i + 1], :].numpy(),y_pred[b_lens[i]:b_lens[i + 1], :].numpy()),step=optimizer.iterations)