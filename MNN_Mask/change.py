import caffe

net = caffe.Net('caffe_model_change/face_mask_detection.prototxt','caffe_model_change/face_mask_detection.caffemodel',caffe.TEST)

net.save('after-modify.caffemodel')