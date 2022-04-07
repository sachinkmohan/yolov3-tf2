'''
Modified by: Sachin K Mohan
Use the below to create a pruned_model
python train_without_flags.py --batch_size 2 --dataset ./data/voc2012_val.tfrecord --val_dataset ./data/voc2012_val.tfrecord --epochs 1 --transfer fine_tune --weights ./checkpoints/yolov3.tf
'''
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

# Pruning imports
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')


def setup_model():
    model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    model.load_weights(FLAGS.weights)
    darknet = model.get_layer('yolo_darknet')
    freeze_all(darknet)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(FLAGS.mode == 'eager_fit'))

    return model, optimizer, loss, anchors, anchor_masks


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    # Setup
    if FLAGS.multi_gpu:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        FLAGS.batch_size = BATCH_SIZE

        with strategy.scope():
            model, optimizer, loss, anchors, anchor_masks = setup_model()
    else:
        model, optimizer, loss, anchors, anchor_masks = setup_model()

    model.summary()

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    else:
        train_dataset = dataset.load_fake_dataset()
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset()
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    '''
    Introducing pruning here
    '''
    num_images = 5717
    end_step = np.ceil(num_images / FLAGS.batch_size).astype(np.int32) * FLAGS.epochs

    '''
    # commented out to test layer wise pruning
    # Define model for pruning.
    # full model pruning
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    '''

    ### Layer pruning begins here

    def apply_pruning_to_conv2d(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.sparsity.keras.prune_low_magnitude(layer,
                                                            pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0))
        return layer

    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_conv2d,
    )

    model_for_pruning.compile(optimizer=optimizer, loss=loss)
    model_for_pruning.summary()

    print('I am inside last model fit')
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    '''
    Introducing Pruning parameters
    '''
    csv_logger = CSVLogger(filename='ssd7_training_quant_pruned_50p_2702.csv',
                       separator=',',
                       append=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

    logdir = 'pruning_summaries'
    checkpoint = ModelCheckpoint('checkpoints/yolov3_pruned_{epoch}.tf',
                        verbose=1, save_weights_only=True)
    callbacks_pruning = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch = 100000000, histogram_freq=0, batch_size=8, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir, update_freq='epoch'),
  checkpoint,
  csv_logger,
  early_stopping,
  reduce_learning_rate
]
    model_for_pruning.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks_pruning,
                        validation_data=val_dataset)
    '''
    -> commenting out the normal training
    start_time = time.time()
    history = model.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset)
    end_time = time.time() - start_time
    print(f'Total Training Time: {end_time}')
    '''

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
