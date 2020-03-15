import tensorflow as tf
import ctc_utils
import cv2
import os
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request




def pred(name):


	image=name
	img=name

	model="semantic_model.meta"
	voc_file="vocabulary_semantic.txt"
	font = cv2.FONT_HERSHEY_SIMPLEX







	tf.reset_default_graph()
	sess = tf.InteractiveSession()


	dict_file = open(voc_file,'r')
	dict_list = dict_file.read().splitlines()
	int2word = dict()
	for word in dict_list:
	    word_idx = len(int2word)
	    int2word[word_idx] = word
	dict_file.close()


	saver = tf.train.import_meta_graph(model)
	saver.restore(sess,model[:-5])

	graph = tf.get_default_graph()

	input = graph.get_tensor_by_name("model_input:0")
	seq_len = graph.get_tensor_by_name("seq_lengths:0")
	rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
	height_tensor = graph.get_tensor_by_name("input_height:0")
	width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
	logits = tf.get_collection("logits")[0]


	WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

	decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

	image = cv2.imread(image,False)
	img=cv2.imread(img,False)

	image = ctc_utils.resize(image, HEIGHT)
	image = ctc_utils.normalize(image)
	image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)





	seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

	prediction = sess.run(decoded,
	                      feed_dict={
	                          input: image,
	                          seq_len: seq_lengths,
	                          rnn_keep_prob: 1.0,
	                      })

	str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)

	temp=[]
	temp2 = []
	final_str = []
	for w in str_predictions[0]:

		if(int2word[w].find('note-') != -1):
			temp = int2word[w].split('note-')
			arr = temp[-1]
			temp2 = arr.split("_")
			final_str.append(temp2[0])
			#print(temp2[0])
			#

	print(final_str)
	j=0
		

	old_im = Image.open(name)
	old_size = old_im.size

	new_size = (1500, 300)
	new_im = Image.new("RGB", new_size,"white")   ## luckily, this is already black!
	new_im.paste(old_im, ((new_size[0]-old_size[0])//2,
	                      (new_size[1]-old_size[1])//2))

	new_im.save('final.png')

	finalim="final.png"
	fin = cv2.imread(finalim,False)





	for i in final_str:
		
		cv2.putText(fin, i ,(582+j,230), font, 0.5,(0,0,255),1,cv2.LINE_AA)
		j=j+70



	#fin.save('transcribed.png')
	#os.remove('static/transcribed.png')
	cv2.imwrite('static/transcribed.png', fin)

	#cv2.imshow('a',fin)
	#cv2.waitKey(0)

app = Flask(__name__)

@app.route("/")


def home():
	return render_template("cafe.html")

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['photo']  
        temp = "static/"+f.filename
        f.save(temp)  
        pred(temp)
        return render_template("success.html", name = temp)



if __name__ == "__main__":
    app.run(debug=True)








