import numpy as np
import tensorflow as tf
import pathlib

from datetime import datetime


def load_images(root_path):
    root_path = pathlib.Path(root_path)
    image_paths = list(root_path.glob('*'))
    image_paths = [str(path) for path in image_paths]

    n_images = len(image_paths)
    print(n_images, 'images imported from', root_path)

    return image_paths


def split_train_test(dataset_list, train_set_ratio):
    np.random.shuffle(dataset_list)
    train_length = int(len(dataset_list) * train_set_ratio)
    train_set = dataset_list[:train_length]
    test_set = dataset_list[train_length:]
    return train_set, test_set

    
def preprocess_image(image):
    image = tf.read_file(image)
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image, channels=3),
        lambda: tf.image.decode_bmp(image, channels=3))
    
    grayscaled = tf.image.rgb_to_grayscale(image)

    ground_truth = tf.image.random_crop(grayscaled, [32, 32, 1])
    ground_truth = tf.cast(ground_truth, tf.float32)
    ground_truth /= 255.0

    downsampled = tf.image.resize_images(ground_truth, [16, 16])
    lowres = tf.image.resize_images(downsampled, [32, 32], method=tf.image.ResizeMethod.BICUBIC)
    #lowres /= 255.0

    return (lowres, ground_truth)


def input_dataset(dataset_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
    dataset = dataset.map(preprocess_image, num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size=len(dataset_list))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()

    (lowres, ground_truth) = iterator.get_next()
    init_op = iterator.initializer

    inputs = {
        'lowres': lowres,
        'ground_truth': ground_truth,
        'iterator_init_op': init_op
    }

    return inputs


def model(inputs, training):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        lowres = inputs['lowres']
        ground_truth = inputs['ground_truth']

        # Build a 3-layered network for SR task(2x) with 3x3 conv & ReLU.
        conv1 = tf.layers.conv2d(inputs=lowres, 
                                filters=64, 
                                kernel_size=[3,3], 
                                padding='SAME', 
                                activation=tf.nn.relu, 
                                name='conv1')
        
        conv2 = tf.layers.conv2d(inputs=conv1, 
                                filters=64, 
                                kernel_size=[3,3], 
                                padding='SAME', 
                                activation=tf.nn.relu, 
                                name='conv2')

        conv3 = tf.layers.conv2d(inputs=conv2, 
                                filters=1, 
                                kernel_size=[3,3], 
                                padding='SAME', 
                                activation=None, 
                                name='conv3')
        
        output = conv3
        loss = tf.losses.mean_squared_error(labels=ground_truth, predictions=output)
        psnr = tf.reduce_mean(tf.image.psnr(output, ground_truth, 1.0))

        if training == True:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            training_op = optimizer.minimize(loss)

        # For tensorboard summary
        loss_summary = tf.summary.scalar('loss', loss)
        psnr_summary = tf.summary.scalar('PSNR', psnr)
        
        img_gt_summary = tf.summary.image("ground truth", ground_truth, max_outputs=1)
        img_output_summary = tf.summary.image("SR result", output, max_outputs=1)
        img_input_summary = tf.summary.image("lowres input", lowres, max_outputs=1)

        # Save the model specification
        spec = inputs
        spec['loss'] = loss
        spec['accuracy'] = psnr
        spec['summary_op'] = tf.summary.merge_all()

        if training == True:
            spec['train_op'] = training_op

        return spec


def train(train_spec, test_spec, n_epoch):
    # Set logging directory with timestamps
    # logging code is from "Hands on ML" book
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    init = tf.global_variables_initializer()
    #saver = tf.train.Saver()
    
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(n_epoch):
            sess.run(train_spec['iterator_init_op'])
            
            if epoch % 5 == 0:
                _, loss, accuracy, summary = sess.run([train_spec['train_op'], train_spec['loss'], train_spec['accuracy'], train_spec['summary_op']])
                file_writer.add_summary(summary, global_step=epoch)
            else:
                _, loss, accuracy = sess.run([train_spec['train_op'], train_spec['loss'], train_spec['accuracy']])
            print('epoch #{} PSNR:{}'.format(epoch, accuracy))
        
        sess.run(test_spec['iterator_init_op'])
        test_loss, test_accuracy = sess.run([test_spec['loss'], test_spec['accuracy']])
        print('TEST PSNR: {}'.format(test_accuracy))

    file_writer.close()


if __name__ == '__main__':
    n_epoch = 100
    batch_size = 128
    
    image_paths = load_images('SR_dataset/291')

    # Create two dataset (input data pipeline with image paths)
    train_paths, test_paths = split_train_test(image_paths, 0.8)

    # Create two iterators over the two datasets
    train_inputs = input_dataset(train_paths, batch_size)
    test_inputs = input_dataset(test_paths, len(test_paths))

    # Define the model and save two model specifications for train and test
    train_spec = model(train_inputs, training=True)
    test_spec = model(test_inputs, training=False)

    # Train the model
    train(train_spec, test_spec, n_epoch)