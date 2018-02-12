import tensorflow as tf

def makeQueue(fileList):
	filename_queue = tf.train.string_input_producer([fileList])
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [['']]*14
	columns = tf.decode_csv(value,record_defaults=record_defaults)
	
	flNames = tf.stack(columns)
	return flNames

def convertToInt(inp) :
	return tf.string_to_number(inp,out_type = tf.int32)

def read_query(flNames, width=70, height=70):
	imgList = tf.unstack(flNames)
	query = tf.read_file(imgList[0])
	query_img = tf.image.decode_png(query)
	query_img = tf.cast(query_img, tf.float32)
	query_img = tf.image.resize_image_with_crop_or_pad(query_img, target_height=height, target_width=width)
	query_img = tf.reshape(query_img,[height,width,3])
	
	text = tf.stack([convertToInt(imgList[i]) for i in range(1,9)])
	labels = tf.stack([convertToInt(imgList[i]) for i in range(9,len(imgList))])
	return query_img, text,labels
	# return query_img,tf.constant(0)

def prepareExampleListQueue(fileList,img_width=70,img_height=70,batch_size=1,num_threads=1) :
	flNames = makeQueue(fileList)
	example_list = [read_query(flNames,img_width,img_height) for i in range(num_threads)]
	example_batch,text_batch,labels_batch = tf.train.batch_join(example_list,batch_size=batch_size,capacity=3*batch_size+1,allow_smaller_final_batch=False)
	# example_batch,text_batch,labels_batch = read_query(flNames,img_width,img_height)
	
	return example_batch,text_batch,labels_batch	
