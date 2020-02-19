from keras import layers
from keras import Input
from keras.models import Model

vocab_size = 50000
num_income_group = 10

posts_input = Input(shape= (None, ), dtype= 'int32', name= 'posts')
embedded_posts = layers.Embedding(256, vocab_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_pred = layers.Dense(1, name= 'age')(x) # Regression output
income_pred = layers.Dense(num_income_group, activation='softmax', name='income')(x) # Multi-class classification
gender_pred = layers.Dense(1, activation='sigmoid', name='gender')(x) # Binary class classification

model = Model(posts_input,
              [age_pred, income_pred, gender_pred])

model.summary()

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], # Different outputs will require different losses
              loss_weights=[0.25, 1., 10.]) # You can also specify different weights for these losses to how much they contribute for final loss

# Prepare your data and call model.fit

model.fit(posts, {'age': age_targets,
                  'income': income_targets,
                  'gender': gender_targets},
          epochs= 10,
          batch_size= 64)



