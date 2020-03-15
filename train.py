from Data_segment import data_seg_class
import ctc_utils
import model
import tensorflow as tf
import os

data_dirpath = "C:/Users/jackg/Desktop/mini-project/Dataset/package_ab"
data_filepath = "train.txt"
dictionary_path = "vocabulary_semantic.txt"
save_model = "./trained_semantic_model"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.reset_default_graph()
sess = tf.InteractiveSession(config=config)

data = data_seg_class(data_dirpath, data_filepath, dictionary_path, val_split=0.1)

img_height = 128
params = model.default_model_params(img_height,data.vocabulary_size)
max_epochs = 64000
dropout = 0.5

inputs, seq_len, targets, decoded, loss, rnn_keep_prob = model.ctc_crnn(params)
train_opt = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver(max_to_keep=None)
sess.run(tf.global_variables_initializer())


# Training loop
for epoch in range(max_epochs):
    batch = data.nextBatch(params)

    _, loss_value = sess.run([train_opt, loss],
                             feed_dict={
                                inputs: batch['inputs'],
                                seq_len: batch['seq_lengths'],
                                targets: ctc_utils.sparse_tuple_from(batch['targets']),
                                rnn_keep_prob: dropout,
                            })

    if epoch % 1000 == 0:
        # VALIDATION
        print ('Loss value at epoch ' + str(epoch) + ':' + str(loss_value))
        print ('Validating...')

        validation_batch, validation_size = data.validation_batch(params)
        
        val_idx = 0
        
        val_ed = 0
        val_len = 0
        val_count = 0
            
        while val_idx < validation_size:
            mini_batch_feed_dict = {
                inputs: validation_batch['inputs'][val_idx:val_idx+params['batch_size']],
                seq_len: validation_batch['seq_lengths'][val_idx:val_idx+params['batch_size']],
                rnn_keep_prob: 1.0            
            }            
                        
            
            prediction = sess.run(decoded,
                                  mini_batch_feed_dict)
    
            str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    

            for i in range(len(str_predictions)):
                ed = ctc_utils.edit_distance(str_predictions[i], validation_batch['targets'][val_idx+i])
                val_ed = val_ed + ed
                val_len = val_len + len(validation_batch['targets'][val_idx+i])
                val_count = val_count + 1
                
            val_idx = val_idx + params['batch_size']
    
        print ('[Epoch ' + str(epoch) + '] ' + str(1. * val_ed / val_count) + ' (' + str(100. * val_ed / val_len) + ' SER) from ' + str(val_count) + ' samples.')        
        print ('Saving the model...')
        saver.save(sess,args.save_model,global_step=epoch)
        print ('------------------------------')
