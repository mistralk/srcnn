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

    
def preprocess_image_train(image):
    image = tf.read_file(image)
    image = tf.image.decode_image(image, channels=3)

    grayscaled = tf.image.rgb_to_grayscale(image)

    ground_truth = tf.image.random_crop(grayscaled, [32, 32, 1])
    ground_truth = tf.cast(ground_truth, tf.float32)
    ground_truth /= 255.0

    downsampled = tf.image.resize_images(ground_truth, [16, 16], method=tf.image.ResizeMethod.AREA)
    lowres = tf.image.resize_images(downsampled, [32, 32], method=tf.image.ResizeMethod.BICUBIC)

    return (lowres, ground_truth)

'''
def preprocess_image(image):
    image = tf.read_file(image)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])

    grayscaled = tf.image.rgb_to_grayscale(image)

    ground_truth = tf.cast(grayscaled, tf.float32)
    ground_truth /= 255.0

    img_shape = tf.shape(image)
    height, width = img_shape[0], img_shape[1]
    down_height = tf.to_int32(height/2)
    down_width = tf.to_int32(width/2)

    downsampled = tf.image.resize_images(ground_truth, [down_height, down_width], method=tf.image.ResizeMethod.AREA)
    lowres = tf.image.resize_images(downsampled, [height, width], method=tf.image.ResizeMethod.BICUBIC)

    return (lowres, ground_truth)
'''

def input_dataset(dataset_list, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)

    if training == True:
        dataset = dataset.map(preprocess_image_train, num_parallel_calls=12)
    '''
    else:
        dataset = dataset.map(preprocess_image, num_parallel_calls=12)
    '''
    dataset = dataset.shuffle(buffer_size=len(dataset_list))
    dataset = dataset.repeat(batch_size)
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
        loss = tf.losses.mean_squared_error(labels=ground_truth, predictions=output)
        psnr = tf.reduce_mean(tf.image.psnr(output, ground_truth, 1.0))

        if training == True:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            global_step = tf.train.get_or_create_global_step()
            training_op = optimizer.minimize(loss, global_step=global_step)

        # For tensorboard summary
        loss_summary = tf.summary.scalar('loss', loss)
        psnr_summary = tf.summary.scalar('PSNR', psnr)
        
        img_gt_summary = tf.summary.image("ground truth", ground_truth, max_outputs=3)
        img_output_summary = tf.summary.image("SR result", output, max_outputs=3)
        img_input_summary = tf.summary.image("lowres input", lowres, max_outputs=3)

        # Save the model specification
        spec = inputs
        spec['output'] = output
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
    '''
    default_graph = tf.get_default_graph()
    conv1_kernel = default_graph.get_tensor_by_name('model/conv1/kernel:0')
    conv1_bias = default_graph.get_tensor_by_name('model/conv1/bias:0')
    conv2_kernel = default_graph.get_tensor_by_name('model/conv2/kernel:0')
    conv2_bias = default_graph.get_tensor_by_name('model/conv2/bias:0')
    conv3_kernel = default_graph.get_tensor_by_name('model/conv3/kernel:0')
    conv3_bias = default_graph.get_tensor_by_name('model/conv3/bias:0')
    saver = tf.train.Saver({'conv1_kernel': conv1_kernel,
                            'conv1_bias': conv1_bias,
                            'conv2_kernel': conv2_kernel,
                            'conv2_bias': conv2_bias,
                            'conv3_kernel': conv3_kernel,
                            'conv3_bias': conv3_bias})
    '''
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        
        best_accuracy = -99999.0

        # Training
        for epoch in range(n_epoch):
            sess.run(train_spec['iterator_init_op'])

            if epoch % 1 == 0:
                
                _, loss, accuracy, summary, global_step = sess.run([train_spec['train_op'], train_spec['loss'], train_spec['accuracy'], train_spec['summary_op'], tf.train.get_global_step()])
                file_writer.add_summary(summary, global_step=global_step)
    
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    save_path = saver.save(sess, 'tmp/model.ckpt')
            else:
                _, loss, accuracy = sess.run([train_spec['train_op'], train_spec['loss'], train_spec['accuracy']])
            print('epoch #{} PSNR:{}'.format(epoch, accuracy))

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
        test_loss, test_accuracy = sess.run([test_spec['loss'], test_spec['accuracy']])
        print('------')
        print('Result')
        print('------')
        print('Test set loss: {}'.format(test_loss))
        print('Test set PSNR: {}'.format(test_accuracy))

    file_writer.close()


def reuse_model(model_path, model_spec):
    # Logging path
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/valid-{}/".format(root_logdir, now)

    # Initialize
    #init = tf.global_variables_initializer()
    #saver = tf.train.Saver()
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    
    with tf.Session() as sess:
        #sess.run(init)
        saver.restore(sess, model_path)
        sess.run(model_spec['iterator_init_op'])
        cnt = 0
        while True:
            try:
                lw, gt, output, loss, accuracy = sess.run([model_spec['lowres'], model_spec['ground_truth'], model_spec['output'], model_spec['loss'], model_spec['accuracy']])
                print(accuracy)

                imageio.imwrite(str(cnt) + '_lowres.png', lw[0])
                imageio.imwrite(str(cnt) + '_gt.png', gt[0])
                imageio.imwrite(str(cnt) + '_output.png', output[0])
                cnt = cnt + 1

            except tf.errors.OutOfRangeError:
                break

        '''
        파악된 문제점
        1. 과거 저장된 트레이닝 set (약 230개)을 그대로 input으로 이용하고 있음 (예: 우주비행사 머리)
        '''

        #file_writer.add_summary(summary, global_step=epoch)

        print('------')
        print('Result')
        print('------')
        print('Set5 loss: {}'.format(loss))
        print('Set5 PSNR: {}'.format(accuracy))

    file_writer.close()


if __name__ == '__main__':
    # TODO: command line interface
    # python srcnn --train --dataset="" --epoch=""
    # python srcnn --eval --dataset="" --model=""

    n_epoch = 10
    batch_size = 128
    
    image_paths = load_images('SR_dataset/291')

    # Create two dataset (input data pipeline with image paths)
    train_paths, test_paths = split_train_test(image_paths, 0.8)

    time_fmt = '%H:%M:%S'
    present = datetime.now().strftime(time_fmt)
    
    # Create two iterators over the two datasets
    train_inputs = input_dataset(train_paths, batch_size, training=True)
    test_inputs = input_dataset(test_paths, len(test_paths), training=True)

    # Define the model and save two model specifications for train and test
    train_spec = model(train_inputs, training=True)
    test_spec = model(test_inputs, training=False)

    # Train the model
    train(train_spec, test_spec, n_epoch)
    
    # Final validation by set5
    set5_paths = load_images('SR_dataset/Set5')
    set5_inputs = input_dataset(set5_paths, len(set5_paths), training=True)
    set5_spec = model(set5_inputs, training=False)
    reuse_model('tmp/model100.ckpt', set5_spec)
    
    now = datetime.now().strftime(time_fmt)

    elapsed_time = datetime.strptime(now, time_fmt) - datetime.strptime(present, time_fmt)
    print('Elapsed time: ', elapsed_time)