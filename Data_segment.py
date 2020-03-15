import random
import cv2
import numpy as np
import ctc_utils

class data_seg_class:

    PAD_COLUMN = 0
    conv_pool = [ [2,2], [2,2], [2,2], [2,2] ]



    def __init__(self, data_dirpath, data_filepath, dictionary_path, val_split=0.0):

        self.curr_idx = 0
        self.data_dirpath = data_dirpath

        
        data_file = open(data_filepath, 'r')
        data_list = data_file.read().splitlines()
        data_list = data_list[:50]
        data_file.close()


        self.word2int = {}
        self.int2word = {}

        dict_file = open(dictionary_path, 'r')
        dict_list = dict_file.read().splitlines()
        for i in dict_list:
            if not i in self.word2int:
                idx = len(self.word2int)
                self.word2int[i] = idx
                self.int2word[idx] = i
        dict_file.close()

        self.vocabulary_size = len(self.word2int)

        random.shuffle(data_list)
        split_idx = int(len(data_list) * val_split)
        self.training_data = data_list[split_idx:]
        self.testing_data = data_list[:split_idx]

        print("Training Data = " + str(len(self.training_data)) + "Testing Data = " + str(len(self.testing_data)))



    def nextBatch(self, params):
            images = []
            labels = []

            
            for i in range(16):
                temp_filepath = self.training_data[self.curr_idx]
                full_path = self.data_dirpath + '/' + temp_filepath + '/' + temp_filepath

               
           
             
                sample_img = cv2.imread(full_path + '.png', False)
                height = 128
                sample_img = ctc_utils.resize(sample_img,height)
                images.append(ctc_utils.normalize(sample_img))

                
                
                sample_full_filepath = full_path + '.semantic'
                
                gt_file = open(sample_full_filepath, 'r')
                gt_list = gt_file.readline().rstrip().split(ctc_utils.word_separator())
                gt_file.close()

                labels.append([self.word2int[lab] for lab in gt_list])

                self.curr_idx = (self.curr_idx + 1) % len( self.training_data )


            
            image_widths = [img.shape[1] for img in images]
            max_width = max(image_widths)

            batch_images = np.ones(shape=[16,
                                           128,
                                           max_width,
                                           1], dtype=np.float32)*self.PAD_COLUMN

            for i, img in enumerate(images):
                batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img

            
            width_reduction = 1
            for i in range(4):
                width_reduction = width_reduction * conv_pool[i][1]

            lengths = [ batch_images.shape[2] / width_reduction ] * batch_images.shape[0]

            return {
                'inputs': batch_images,
                'seq_lengths': np.asarray(lengths),
                'targets': labels,
            }






    def validation_batch(self, params):
            images = []
            labels = []

            
            for i in range(16):
                temp_filepath = self.testing_data[self.curr_idx]
                full_path = self.corpus_dirpath + '/' + temp_filepath + '/' + temp_filepath

               
           
             
                sample_img = cv2.imread(full_path + '.png', False)
                height = 128
                sample_img = ctc_utils.resize(sample_img,height)
                images.append(ctc_utils.normalize(sample_img))

                
                
                sample_full_filepath = full_path + '.semantic'
                
                gt_file = open(sample_full_filepath, 'r')
                gt_list = gt_file.readline().rstrip().split(ctc_utils.word_separator())
                gt_file.close()

                labels.append([self.word2int[lab] for lab in gt_list])

                self.curr_idx = (self.curr_idx + 1) % len( self.testing_data )


            
            image_widths = [img.shape[1] for img in images]
            max_width = max(image_widths)

            batch_images = np.ones(shape=[16,
                                           128,
                                           max_width,
                                           1], dtype=np.float32)*self.PAD_COLUMN

            for i, img in enumerate(images):
                batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img

            
            width_reduction = 1
            for i in range(4):
                width_reduction = width_reduction * conv_pool[i][1]

            lengths = [ batch_images.shape[2] / width_reduction ] * batch_images.shape[0]

            return {
                'inputs': batch_images,
                'seq_lengths': np.asarray(lengths),
                'targets': labels,
            }