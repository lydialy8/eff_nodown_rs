import tensorflow as tf
import os
import model
import transforms_mod as transforms

batch_size = 8
epochs = 20
num_classes = 2
bands = 4
length = 6000
height = 512
width = 512
resume = True;
onlyTest = True;
onlyTrain = False;
train_dataset_path = [
    'D:/detection_experiments/datasets/australia_pristine/val/*/*']  # /media/hdddati1/labady/dataset/detection/all_13bands_train.tfrecords
# val_dataset_path = ['/media/hdddati1/labady/dataset/detection/all_13bands_val.tfrecords']
val_dataset_path = ['D:/detection_experiments/datasets/china_13bands/val/*/*']
val_dataset_path = ['D:/detection_experiments/datasets/china_4bands/val/val/*/*']
#val_dataset_path = ['D:/detection_experiments/datasets/australia_pristine/val/*/*']



# val_dataset_path = ['G:\\scand_13bands_train.tfrecords']


model_name = 'last_models/refined/efficientb4_model_lc_4bands.h5'
model_name_noext = 'efficientb4_model_all_13bands'

data_augmentation = transforms.Compose([
    transforms.GaussianBlur(kernel_size=9, p=0.2),
    transforms.RandomCrop(32),
    transforms.RandomFlip(p=0.3),
    transforms.RandomShift(max_percent=0.1, p=0.3),
    transforms.RandomRotation(30, p=0.2),
    transforms.ToTensor()
])

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    if parts[-2] == 'fake':
        return 0
    return 1


def parse_path(record):
    label = get_label(record)
    # load the raw data from the file as a string
    img = tf.io.read_file(record)
    data = tf.io.decode_raw(img, tf.uint16)[858:]  # 1653 858  144
    img = data / tf.reduce_max(data, axis=None)
    reshaped_img = tf.reshape(img, (512, 512, 4))
    #reshaped_img = tf.gather(reshaped_img, [1, 2, 3, 7], axis=2)
    return reshaped_img, label


def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
    data = tf.io.decode_raw(features['data'], tf.uint16)
    img = data / tf.reduce_max(data, axis=None)
    return tf.reshape(img, features['shape']), features['label']


def data_generator(files):
    dataset = tf.data.Dataset.list_files(files, shuffle=True)  # tf.data.TFRecordDataset(files)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    autotune = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(parse_path,
                          num_parallel_calls=autotune)  # dataset.map(parse_tfrecord_tf, num_parallel_calls=32)
    dataset = dataset.map(lambda x, y: (tf.py_function(data_augmentation, inp=[x], Tout=tf.float32), y), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(autotune)
    return dataset


def data_generator_val(files):
    dataset = tf.data.Dataset.list_files(files, shuffle=False)  # tf.data.TFRecordDataset(files)
    autotune = tf.data.experimental.AUTOTUNE
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(parse_path,
                          num_parallel_calls=autotune)  # dataset.map(parse_tfrecord_tf, num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(autotune)
    return dataset


shape = height, width, bands
tf.keras.backend.clear_session()
model = model.EfficientNetB4(include_top=True,
                             input_tensor=None,
                             input_shape=shape,
                             pooling=None,
                             weights=None,
                             classes=2,
                             classifier_activation="softmax")

model.summary()

starter_learning_rate = 1e-2
end_learning_rate = 1e-5
decay_steps = 80000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.8)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.metrics.SparseCategoricalAccuracy()])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_name,
                                       save_best_only=False,
                                       save_weights_only=False,
                                       save_freq="epoch",
                                       monitor='loss')]
if not onlyTest:
    train_dataset = data_generator(train_dataset_path)
    if resume:
        model.load_weights(model_name)
        print("weights loaded")

    model.fit(train_dataset,
              epochs=epochs, steps_per_epoch=length // batch_size,
              callbacks=callbacks)

    model.save(model_name_noext)
if not onlyTrain:
    val_dataset = data_generator_val(val_dataset_path)
    model.load_weights(model_name)
    print("Evaluate on test data")
    results = model.evaluate(val_dataset)
    print("test loss, test acc:", results)