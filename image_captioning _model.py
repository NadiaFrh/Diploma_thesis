#%%
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import re
import numpy as np
import os
import json
from glob import glob
import pickle
import copy  
import datetime


print("Loading data")
total_train_image_name = pickle.load( open( 'train_images.pickle', "rb" ) )
total_val_image_name = pickle.load( open( 'val_images.pickle', "rb" ) )
attention_regions = pickle.load( open( 'my_total_regions.pickle', "rb" ) )
#incept_rois_train = pickle.load( open( 'incept_rois_train.pickle', "rb" ) )
#incept_rois_val = pickle.load( open( 'incept_rois_val.pickle', "rb" ) )

       
#%%
#incept_rois_val = pickle.load( open( 'incept_rois_val.pickle', "rb" ) )
annotation_file_train = 'annotations/captions_train2014.json'
with open(annotation_file_train, 'r') as f:
    annotations_train = json.load(f)



annotation_file_val = 'annotations/captions_val2014.json'
with open(annotation_file_val, 'r') as f:
    annotations_val = json.load(f)
print("Loading data completed")

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []
cap_validation = []
img_name_validation = []
print("Process Captions")
i=0
training_data_dict = defaultdict(list)
validation_data_dict = defaultdict(list)
for annot in annotations_train['annotations']:

    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = 'train2014/COCO_train2014_' + '%012d.jpg' % (image_id)
    #if (full_coco_image_path not in all_img_name_vector) and ((full_coco_image_path in total_train_image_name)): #or (full_coco_image_path in rcnn_train_rois2)):
    if ((full_coco_image_path in total_train_image_name)):
        print(i)
        i+=1
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)
        training_data_dict[full_coco_image_path].append(caption)


for annot in annotations_val['annotations']:

    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = 'val2014/COCO_val2014_' + '%012d.jpg' % (image_id)
    #if (full_coco_image_path not in img_name_validation) and (full_coco_image_path in total_val_image_name):
    if (full_coco_image_path in total_val_image_name):
        img_name_validation.append(full_coco_image_path)
        cap_validation.append(caption)
        validation_data_dict[full_coco_image_path].append(caption)
        
# Shuffle captions and image_names together
# Set a random state
#%%
cap_train, img_name_train = shuffle(all_captions,
                                    all_img_name_vector,
                                    random_state=1)


#%%
# Select the first 30000 captions from the shuffled set
print("Process Captions Completed")  
print("Starting Tokenization")
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

#top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer.fit_on_texts(cap_train)
init_dict = tokenizer.word_docs
frequency = copy.deepcopy(tokenizer.word_docs)
i=0
for token in frequency:
    freq = frequency[token]
    if freq < 2:

        #del tokenizer.word_docs[token]
        #del tokenizer.index_word[tokenizer.word_index[token]]
        del tokenizer.word_index[token]   
#dir1 = tokenizer.word_docs  
#tokenizer.word_docs = sorted(tokenizer.word_docs.items(), key=lambda x: x[1], reverse=True)
#dir1 = tokenizer.word_docs
frequency = copy.deepcopy(tokenizer.word_index)
tokenizer.word_index = defaultdict(list)
tokenizer.index_word = defaultdict(list)
i=0
for token in frequency :
    i+=1
    tokenizer.word_index[token] = i
    tokenizer.index_word[i] = token
    #print(tokenizer.index_word)
dict1 = tokenizer.word_index  
dict2 = tokenizer.index_word 
#print(dict1)      
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
#train_seqs = tokenizer.texts_to_sequences(cap_train)
train_seqs = tokenizer.texts_to_sequences(cap_train)
        
# Create the tokenized vectors
#train_seqs = tokenizer.texts_to_sequences(train_captions)
# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
max_length = calc_max_length(train_seqs)
cap_train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs,max_length, padding='post')

#cap_train_vector_5 = [cap_train_vector[n:n+5] for n in range(0, len(cap_train_vector),5)]


val_seqs = tokenizer.texts_to_sequences(cap_validation)
cap_val_vector = tf.keras.preprocessing.sequence.pad_sequences(val_seqs,max_length, padding='post')

# Calculates the max_length, which is used to store the attention weights
print("Tokenization Completed")
#%%

img_name_val, image_name_test, cap_val, cap_test= train_test_split(img_name_validation,
                                                                    cap_val_vector,
                                                                    test_size=0.5,
                                                                    random_state=0)



#%%
import tensorflow as tf  
from numpy import asarray,zeros 
 

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc1 = tf.keras.layers.Dense(embedding_dim)
        #self.fc2 = tf.keras.layers.Dense(embedding_dim)
        #self.BatchNormalization = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, incept):
        #x = tf.concat([incept, rcnn], axis=-2)
        #x = self.BatchNormalization(incept)
        x = self.fc1(incept)
        #incept = self.fc2(incept)
        #features = self.fc2(features)
        #x = tf.concat([incept, rcnn], axis=-2)
        x = tf.nn.relu(x)
        x = self.dropout(x)
        return x


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                               embedding_dim) 
                                               #embeddings_initializer=tf.keras.initializers.Constant(value=embedding_matrix))
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.dropout = tf.keras.layers.Dropout(0.3)
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    #hidden = tf.concat([hidden1, hidden2, hidden3], axis=-1)
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state1 = self.gru(x)
    #output, state2 = self.gru2(output)
    #output, state3 = self.gru3(output)
    output = self.dropout(output)
    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)
    x = tf.nn.relu(x)
    #x = tf.nn.relu(x)
    #x = self.dropout(x)
    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)
    x = tf.nn.softmax(x)


    return x, state1, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights




#%%
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import copy
# Read the json file

         
                                                                                                                               
BATCH_SIZE = 64                     
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512                 
vocab_size = len(tokenizer.word_index) + 1                                                                                                         
num_steps = len(img_name_train) // BATCH_SIZE
num_train_steps = len(img_name_val) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 5
EPOCHS = 20
# Load the numpy files
#%%
total_max_value = 0
i=0
for image_path in total_val_image_name:
    print(i)
    box_features = np.load('box_features/'+image_path+'.npy')
    for j in range(4):
        max_value = max(box_features[j])
        if max_value > total_max_value:
            total_max_value = max_value
    i+=1



#%%
from sklearn.preprocessing import normalize
i=0
total_max_value = 27.2314
for image_path in total_val_image_name:
    print(i)
    '''
    if i == 1:
        break
    '''
    i+=1

    reg_rcnn = []
    reshaped_rcnn = []
    box_features = np.load('box_features/'+image_path+'.npy')
    for rcnn in box_features:
        reshaped_rcnn.append([rcnn]*2)
    reshaped_rcnn = np.reshape(reshaped_rcnn,(4, 2048))

    for j in range(4):
        if max(reshaped_rcnn[j]) == 0 :
            norm = reshaped_rcnn[j]
        else:
            norm = reshaped_rcnn[j]/total_max_value
        reg_rcnn.append(norm)
    #reg_rcnn_sum = sum(reg_rcnn[3])
    img_incept = np.load(image_path+'.npy')[0]
    img_incept = np.reshape(img_incept,(1, 2048))
    total_features = np.concatenate((img_incept, reshaped_rcnn), axis=-2)
    total_features = np.reshape(total_features,(5, 2048))
    np.save('total_features/'+ image_path, total_features)
#%%
img_name = 'val2014/COCO_val2014_000000115626.jpg'
img_incept = np.load(img_name+'.npy')

#%%

def map_func(img_name, cap):

  #img_tensor = np.load('rois_' + img_name.decode('utf-8')+'.npy')
  #print("img_tensor=",img_tensor)
  #img_incept = incept_rois_train_padded[img_name.decode('utf-8')]
  #print(img_incept)
  #img_tensor = tf.reshape(img_tensor,(attention_features_shape, 2048))
  #img_incept = rcnn_train_incept2[img_name.decode('utf-8')]
  #img_incept = tf.reshape(img_incept,(1,2048))
  '''
  img_rcnn = tf.dtypes.cast(np.load('box_features/'+img_name.decode('utf-8')+'.npy'), dtype=tf.float32)
  for incept in img_rcnn:
    reshaped_rcnn.append([incept]*2)
  reshaped_rcnn = np.reshape(reshaped_rcnn,(4, 2048))

  img_incept = tf.dtypes.cast(np.load(img_name.decode('utf-8')+'.npy'), dtype=tf.float32)
  #img_incept = tf.reshape(img_incept,(1, 2048))
  #img_incept = tf.reshape(img_incept,(attention_features_shape, 2048))
  '''
  #features = tf.dtypes.cast(np.load('total_features/'+img_name.decode('utf-8')+'.npy'), dtype=tf.float32)
  #features = tf.reshape(features,(5, 2048))
  img_incept = tf.dtypes.cast(np.load(img_name.decode('utf-8')+'.npy'), dtype=tf.float32)
  img_incept = tf.reshape(img_incept,(5, 2048))
  #img_incept = tf.dtypes.cast(tf.reshape(img_incept,(attention_features_shape,2048)), dtype=tf.float32)
  #img_tensor = list(np.float_(img_tensor))
  return img_incept, cap





print("Creating Dataset")
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train_vector))
# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print("Creating Dataset Completed")


print("Creating Models")
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

#encoder.load_weights('encoder_weights')
#decoder.load_weights('decoder_weights')


initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                 initial_learning_rate,
                 decay_steps=10000,
                 decay_rate=0.9,
                 staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')

accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
accuracy_object_val = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

def accuracy_function(real, pred):
  pred = tf.nn.softmax(pred)
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracy_ = accuracy_object(real, pred)
  #mask = tf.cast(mask, dtype=accuracy_.dtype)
  #accuracy_ *= mask
  return accuracy_ #tf.reduce_mean(accuracy_)
    

print("Creating Models Completed")


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           #combine_models=combine_models,
                           optimizer = optimizer)
'''
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
'''  
acc_val_plot = []  
acc_plot = [] 
loss_val_plot = []  
loss_plot = []  
train_accuracy_results = []
val_accuracy_results = []
total_acc = []
iteration_loss = []
iteration_loss_val = []
@tf.function
def train_step(img_tensor, target):
  loss = 0
  acc = 0
  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden1 = decoder.reset_state(batch_size=target.shape[0])
  #hidden2 = decoder.reset_state(batch_size=target.shape[0])
  #hidden3 = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

  with tf.GradientTape() as tape:

      features = encoder(img_tensor)
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden1, _ = decoder(dec_input, features, hidden1)#, hidden2, hidden3)
          loss += loss_function(target[:, i], predictions)
          dec_input = tf.expand_dims(target[:, i], 1)
          
  total_loss = (loss / int(target.shape[1]))
  total_acc = (acc / int(target.shape[1]))
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables 
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss, total_acc
  

@tf.function
def validation_step(img_tensor, target):
  loss = 0
  acc = 0
  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden1 = decoder.reset_state(batch_size=BATCH_SIZE)
  #hidden2 = decoder.reset_state(batch_size=BATCH_SIZE)
  #hidden3 = decoder.reset_state(batch_size=BATCH_SIZE)

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']]*BATCH_SIZE , 1)
  features = encoder(img_tensor)
  for i in range(1, target.shape[1]):
          # passing the features through the decoder
    predictions, hidden1, _ = decoder(dec_input, features, hidden1)#, hidden2, hidden3)

    loss += loss_function(target[:, i], predictions)
    #acc += accuracy_object_val(target[:, i], predictions)
    # using teacher forcing
    dec_input = tf.expand_dims(target[:, i], 1)
  total_loss = (loss / int(target.shape[1]))
  total_acc = (acc / int(target.shape[1])) 
  return loss, total_loss, total_acc


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/val'
lr_rate_log_dir = 'logs/gradient_tape/' + current_time + '/learning_rate'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
lr_rate_summary_writer = tf.summary.create_file_writer(lr_rate_log_dir)

  
  


print("Starting Training")
for epoch in range(0, EPOCHS):
    start = time.time()
    total_loss = 0
    total_accuracy = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss, t_acc = train_step(img_tensor, target)
        iteration_loss.append(batch_loss.numpy()/ int(target.shape[1]))
        #print("total_acc", t_acc)
        total_loss += t_loss
        total_accuracy += t_acc

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', total_loss / num_steps, step=epoch)

        #tf.summary.scalar('accuracy', total_accuracy / num_steps, step=epoch)

    '''
    if epoch % 5 == 0:
      ckpt_manager.save()
    '''
    '''
    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    '''
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start)) 

    total_val_loss = 0  
    total_val_acc = 0

    for img_tensor, target in dataset_val:
        batch_loss, t_loss, t_val_acc= validation_step(img_tensor, target)
        iteration_loss_val.append(batch_loss.numpy()/ int(target.shape[1]))
        total_val_loss += t_loss
        total_val_acc += t_val_acc

    loss_val_plot.append(total_val_loss / num_train_steps)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', total_val_loss / num_train_steps, step=epoch)
        #tf.summary.scalar('accuracy', total_val_acc / num_train_steps, step=epoch)
        

    #template = 'Epoch {}, Loss: {}, Accuracy: {} Validation Loss: {}, Validation Accuracy: {}\n'
    template = 'Epoch {}, Loss: {}, Validation Loss: {}\n'
    print(template.format(epoch+1,
                          total_loss/num_steps,
                          total_val_loss / num_train_steps))
    if epoch+1==10:
        encoder.save_weights('encoder_weights_rcnn_10', save_format='tf')   
        decoder.save_weights('decoder_weights_rcnn_10', save_format='tf')
    if epoch+1==20:
        encoder.save_weights('encoder_weights_rcnn_20', save_format='tf')   
        decoder.save_weights('decoder_weights_rcnn_20', save_format='tf') 
 
    #accuracy_object.reset_states()
    #accuracy_object_val.reset_states()


#encoder.save_weights('encoder_weights_30', save_format='tf')   
#decoder.save_weights('decoder_weights_30', save_format='tf') 

#%%
plt.plot(loss_plot)
plt.plot(loss_val_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('model loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(iteration_loss)
plt.plot(iteration_loss_val)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print("Training Ended") 
 

#%%
from collections import defaultdict 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

  

# Select the first 30000 captions from the shuffled set
num_examples = 20
test_captions = cap_val[:num_examples]
test_img_name_vector = img_name_val[:num_examples] 
 

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

encoder.load_weights('encoder_weights_25')
decoder.load_weights('decoder_weights_25') 

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def evaluate(image):
    result = []
    
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)
    #hidden2 = decoder.reset_state(batch_size=1)
    #hidden3 = decoder.reset_state(batch_size=1)


    img_incept = tf.dtypes.cast(np.load(image+'.npy'),  dtype=tf.float32)
    #img_incept = tf.reshape(img_incept,(64,2048))
    #img_tensor_val = image_features_extract_model(temp_input)
    #img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_incept)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result.append('<start>')

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        #print(attention_weights)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        #print(attention_plot[i])
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = Image.open(image)
    regions = attention_regions[image]
    #print(attention_plot)
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    for l in range(len_result):
        if l == 52:
            break
        #print("\n")
        i=0
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        plt.axis('off')
        ax.set_title(result[l])
        #print("attention_plot", attention_plot[l][0])
        ax.imshow(temp_image, cmap='gray', alpha=1-attention_plot[l][0]/2)
        for r in regions[0]: 
            try:
                a = attention_plot[l][i+1]*2
            except Exception as e: 
                print(e)
                continue
            if attention_plot[l][i+1]*2<1:
                a = attention_plot[l][i+1]*2
            else:
                a = 1
            rect = patches.Rectangle((r[1], r[0]), abs(r[1]-r[3]), abs(r[0]-r[2]),linewidth=1,edgecolor='none',facecolor='white', alpha=attention_plot[l][i+1])
            ax.add_patch(rect)
            i+=1

            #print("\n")


    plt.tight_layout()
    plt.show()
    



real_prediction = defaultdict(list)
total_caption = defaultdict(list)
total_attention_plot = defaultdict(list)
rid = 0
i=0
path = 'eikones/eikones/'
for image_name in image_paths:
    '''
    if i==10:
        break
    i+=1
    '''
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]])
    rid+=1
    result, attention_plot = evaluate(image_name)
    

    print(image_name)
    print(real_caption)
    plot_attention(image_name, result, attention_plot)
    real_prediction[image_name].append(real_caption)
    total_caption[image_name].append(result)
    total_attention_plot[image_name].append(attention_plot)
 
    
    
    
#%%
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu, SmoothingFunction
from collections import defaultdict 
from nltk.translate.meteor_score import meteor_score

smoothie = SmoothingFunction().method4
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
num_examples = 10
encoder.load_weights('encoder_weights2')
decoder.load_weights('decoder_weights2')
test_img_name_vector = img_name_val[:num_examples] 

encode_test = sorted(set(image_name_test))
def BLUE_scores():
    total_BLEU_1 = []
    total_BLEU_2 = []
    total_BLEU_3 = []
    total_BLEU_4 = []
    total_captions = defaultdict(list)
    total_meteor = []
    rid = 0
    for image_name in encode_test:
        print(rid)
        reference = []
        candidate, _ = evaluate(image_name)
        candidate_join =' '.join(candidate)
        #print(candidate)
        #reference = [[tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]]]
        #reference =' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
        r = validation_data_dict[image_name]
        for ref in r:
            reference.append(ref.split())
        #print(reference,'\n')
        rid+=1
        BLEU_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        BLEU_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        BLEU_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        BLEU_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        meteor = meteor_score(r, candidate_join)
        total_meteor.append(meteor)
        total_BLEU_1.append(BLEU_1)
        total_BLEU_2.append(BLEU_2)
        total_BLEU_3.append(BLEU_3)
        total_BLEU_4.append(BLEU_4)
        total_captions[image_name].append([candidate])
    return  total_BLEU_1, total_BLEU_2, total_BLEU_3, total_BLEU_4, total_meteor 
 
def average(score):
    return sum(score)/len(score)      
    


if __name__=="__main__":     
    total_BLEU_1, total_BLEU_2, total_BLEU_3, total_BLEU_4, total_meteor = BLUE_scores()
    average_BLEU_1 = average(total_BLEU_1)
    average_BLEU_2 = average(total_BLEU_2)
    average_BLEU_3 = average(total_BLEU_3)
    average_BLEU_4 = average(total_BLEU_4)   
    average_meteor = average(total_meteor)
#%%
from nltk.translate.meteor_score import meteor_score
import nltk
#nltk.download('wordnet')

def METEOR_scores():
    total_meteor = []
    rid = 0
    for image_name in encode_test[:10]:
        print(rid)
        if rid==10:
            break
        
        reference = []
        candidate, _ = evaluate(image_name)
        candidate =' '.join(candidate)
        print(candidate)
        r = validation_data_dict[image_name]
        for ref in r:
            reference.append(ref.split())
        print(r,'\n')
        rid+=1
        meteor = meteor_score(r, candidate)
        total_meteor.append(meteor)
    return  total_meteor
 
def average(score):
    return sum(score)/len(score)      
    


if __name__=="__main__":     
    total_meteor = METEOR_scores()
    average_meteor = average(total_meteor)
  
    
#%%
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

   
def score(ref, sample):
    # ref and sample are both dict
  scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        #(Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
  ]
  final_scores = {}
  for scorer, method in scorers:
    print( 'computing %s score with COCO-EVAL...'%(scorer.method()))
    score, scores = scorer.compute_score(ref, sample)
    if type(score) == list:
        for m, s in zip(method, score):
            final_scores[m] = s
    else:
        final_scores[method] = score
  return final_scores 

      
#%%
rid = 0
ref = []
cand = []
encode_test = sorted(set(image_name_test))
for image_name in encode_test[:5000]:
    print(rid)
    reference = []
    candidate, _ = evaluate(image_name)
    cand.append(' '.join(word for word in candidate))
    r = validation_data_dict[image_name]
    ref.append(r)
    '''
    for ref in r:
        reference.append(ref.split())
    '''
    #reference = [tokenizer.index_word[i] for i in cap_test[rid] if i not in [0]]
    #ref_join = [' '.join(word for word in reference)]
    #ref.append(ref_join)
    #cand.append([' '.join(word for word in candidate)])
    rid+=1
#%%
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(cand)
dict3 = tokenizer.word_index  
#%%  
num = 643
reference_dict = dict(list(enumerate(ref[:5000], start=0)))
candidate_dict = dict(list(enumerate(cand[:5000], start=0)))
final_scores = score(reference_dict, candidate_dict)
    
    
#%%

rid = 0
encode_test = sorted(set(image_name_test))
#for image_name in encode_test:
image_name = 'val2014/COCO_val2014_000000154339.jpg'
ref = []
cand = []
print(rid)
reference = []
candidate, _ = evaluate(image_name)
cand.append([' '.join(word for word in candidate)])
r = validation_data_dict[image_name]
ref.append(r)
reference_dict = dict(list(enumerate(ref, start=0)))
candidate_dict = dict(list(enumerate(cand, start=0)))
final_scores = score(reference_dict, candidate_dict)
rid+=1






#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = 'val2014/COCO_val2014_000000115626.jpg'
image_path = 'val2014/COCO_val2014_000000571008.jpg'
result, attention_plot = evaluate(image_path)
caption = [' '.join(word for word in result)]
print(caption)
img=mpimg.imread(image_path)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()

plot_attention(image_path, result, attention_plot)
