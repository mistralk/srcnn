import numpy as np
import tensorflow as tf
import pathlib
import os
import imageio
from datetime import datetime

# Ignore warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    image = tf.image.decode_image(image, channels=3)

    grayscaled = tf.image.rgb_to_grayscale(image)

    ground_truth = tf.image.random_crop(grayscaled, [32, 32, 1])
    ground_truth = tf.cast(ground_truth, tf.float32)
    ground_truth /= 255.0

    downsampled = tf.image.resize_images(ground_truth, [16, 16], method=tf.image.ResizeMethod.AREA)
    lowres = tf.image.resize_images(downsampled, [32, 32], method=tf.image.ResizeMethod.BICUBIC)

    ground_truth = tf.clip_by_value(ground_truth, 0.0, 1.0)
    lowres = tf.clip_by_value(lowres, 0.0, 1.0)

    return (lowres, ground_truth)


def input_dataset(dataset_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
    dataset_length = len(dataset_list)

    dataset = dataset.map(preprocess_image, num_parallel_calls=12)
    dataset = dataset.shuffle(buffer_size=dataset_length)
    dataset = dataset.repeat(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()

    (lowres, ground_truth) = iterator.get_next()
    init_op = iterator.initializer

    inputs = {
        'lowres': lowres,
        'ground_truth': ground_truth,
        'dataset_length': dataset_length,
        'iterator_init_op': init_op
    }

    return inputs


def model(inputs, training=False):
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
        output = tf.clip_by_value(output, 0.0, 1.0)

        loss = tf.losses.mean_squared_error(labels=ground_truth, predictions=output)
        psnr = tf.reduce_mean(tf.image.psnr(output, ground_truth, 1.0))

        if training == True:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            training_op = optimizer.minimize(loss)

            loss_summary = tf.summary.scalar('loss', loss)
            psnr_summary = tf.summary.scalar('PSNR', psnr)
            
            img_gt_summary = tf.summary.image("ground truth", ground_truth, max_outputs=3)
            img_output_summary = tf.summary.image("SR result", output, max_outputs=3)
            img_input_summary = tf.summary.image("lowres input", lowres, max_outputs=3)

        # Save the model specification
        spec = inputs
        spec['output'] = output
        spec['loss_op'] = loss
        spec['psnr_op'] = psnr

        if training == True:
            spec['train_op'] = training_op
            spec['summary_op'] = tf.summary.merge_all()

        return spec


def train_and_test(train_spec, test_spec, n_epoch):
    # Set logging directory with timestamps
    # logging code is from "Hands on ML" book
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    iterator_init_op = train_spec['iterator_init_op']
    train_op = train_spec['train_op']
    loss_op = train_spec['loss_op']
    psnr_op = train_spec['psnr_op']
    summary_op = train_spec['summary_op']

    with tf.Session() as sess:
        sess.run(init)
        
        best_accuracy = -99999.0

        # Training
        for epoch in range(n_epoch):
            sess.run(train_spec['iterator_init_op'])

            if epoch % 10 == 0:
                _, loss, accuracy, summary = sess.run([train_op, loss_op, psnr_op, summary_op])
                file_writer.add_summary(summary, global_step=epoch)
    
            else:
                _, loss, accuracy = sess.run([train_op, loss_op, psnr_op])
            
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                save_path = saver.save(sess, 'tmp/model.ckpt')

            print('epoch #{} PSNR:{} LOSS:{}'.format(epoch, accuracy, loss))

        # Test
        saver.restore(sess, 'tmp/model.ckpt')

        # Get parameter from tf variables
        default_graph = tf.get_default_graph()
        conv1_kernel = default_graph.get_tensor_by_name('model/conv1/kernel:0').eval()
        conv1_bias = default_graph.get_tensor_by_name('model/conv1/bias:0').eval()
        conv2_kernel = default_graph.get_tensor_by_name('model/conv2/kernel:0').eval()
        conv2_bias = default_graph.get_tensor_by_name('model/conv2/bias:0').eval()
        conv3_kernel = default_graph.get_tensor_by_name('model/conv3/kernel:0').eval()
        conv3_bias = default_graph.get_tensor_by_name('model/conv3/bias:0').eval()

        # Save parameters as numpy array
        np.save("conv1_kernel", conv1_kernel)
        np.save("conv1_bias", conv1_bias)
        np.save("conv2_kernel", conv2_kernel)
        np.save("conv2_bias", conv2_bias)
        np.save("conv3_kernel", conv3_kernel)
        np.save("conv3_bias", conv3_bias)
        
        sess.run(test_spec['iterator_init_op'])
        test_loss, test_accuracy = sess.run([test_spec['loss_op'], test_spec['psnr_op']])
        print('------------------------------------')
        print('Test set loss: {}'.format(test_loss))
        print('Test set PSNR: {}'.format(test_accuracy))
        print('------------------------------------')

    file_writer.close()


def reuse_model(model_path, model_spec):
    # Logging path
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/valid-{}/".format(root_logdir, now)

    # Initialize
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    lw_images = model_spec['lowres']
    gt_images = model_spec['ground_truth']
    output_images = model_spec['output']
    loss_op = model_spec['loss_op']
    psnr_op = model_spec['psnr_op']

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        sess.run(model_spec['iterator_init_op'])
               
        lw, gt, output, loss, accuracy = sess.run([lw_images, gt_images, output_images, loss_op, psnr_op])

        # Save result images
        for i in range(model_spec['dataset_length']):
            imageio.imwrite(str(i) + '_lowres.png', lw[i])
            imageio.imwrite(str(i) + '_gt.png', gt[i])
            imageio.imwrite(str(i) + '_output.png', output[i])

        print('------------------------------------')
        print('Set5 loss: {}'.format(loss))
        print('Set5 PSNR: {}'.format(accuracy))
        print('------------------------------------')

    file_writer.close()


def test_Set5():
    set5_paths = load_images('SR_dataset/Set5')
    set5_inputs = input_dataset(set5_paths, len(set5_paths))
    set5_spec = model(set5_inputs, training=False)
    reuse_model('tmp/model.ckpt', set5_spec)


if __name__ == '__main__':
    # TODO: command line interface

    n_epoch = 10000
    batch_size = 128
    
    # Create two dataset (input data pipeline with image paths)
    image_paths = load_images('SR_dataset/291')
    train_paths, test_paths = split_train_test(image_paths, 0.8)

    # Timer start
    time_fmt = '%H:%M:%S'
    present = datetime.now().strftime(time_fmt)
    
    # Create two iterators over the two datasets
    train_inputs = input_dataset(train_paths, batch_size)
    test_inputs = input_dataset(test_paths, len(test_paths))

    # Define the model and save two model specifications for train and test
    train_spec = model(train_inputs, training=True)
    test_spec = model(test_inputs, training=False)

    # Train the model
    train_and_test(train_spec, test_spec, n_epoch)

    # Timer end
    now = datetime.now().strftime(time_fmt)
    elapsed_time = datetime.strptime(now, time_fmt) - datetime.strptime(present, time_fmt)
    print('Elapsed time for training: ', elapsed_time)
    print()

    # Final validation by Set5
    test_Set5()