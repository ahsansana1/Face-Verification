import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
import seaborn as sns
sns.set(color_codes=True)
pal = sns.color_palette("Set2", 10)
sns.set_palette(pal)

df = pd.read_pickle('data2.pkl')
def person(person_num):
    image = df[person_num]
    print(image.shape)
    g = image.reshape(87,65)
    plt.imshow(g,cmap='gray')
    plt.axis('off')
    plt.savefig(f'static/images/{person_num}.png')
    return image

def build_and_load_model():
    K.clear_session()
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(24,24), activation='relu',padding='same', strides=1,
                     input_shape=(87, 65, 1)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=6, strides=None, padding='valid'))
    model.add(Conv2D(30, kernel_size=(12,12), activation='relu',padding='same', ))
    model.add(MaxPooling2D(pool_size=3, strides=None, padding='valid'))
    model.add(Conv2D(40, kernel_size=(6,6), activation='relu',padding='same', ))
    model.add(Flatten())
    model.add(Dense(12, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model = load_model('my_model2.h5')

    return model

def main(person_num):

    """Takes lists of celebrities, uses trained
    model to make predictions, puts results into
    dataframe, and plots results."""


    celeb = ['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush',
           'Gerhard Schroeder', 'Hugo Chavez', 'Jacques Chirac', 'Jean Chretien',
           'John Ashcroft', 'Junichiro Koizumi', 'Serena Williams', 'Tony Blair']

    image1 = person(person_num)

    #for testing:
    # print(f'IMAGE 1 IS: {image1}')
    # print(f'IMAGE 1 TYPE: {type(image1)}')

    model = build_and_load_model()
    t = model.predict_proba((image1).reshape(1,87,65,1))

    df6 = pd.DataFrame(t,columns=celeb)
    # print(df6)
    plt.figure()
    test_figure = df6.loc[0].plot(kind='barh',figsize=(50, 40))
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50) #.loc[0] extracts the Series
    fn = f'static/images/fig_person_{person_num}.png'
    print(fn)
    plt.savefig(fn)

    print(f'Created image for person {person_num}')
    print(f'\nCreated data for person {person_num}')
if __name__ == '__main__':
    num = int(input('Enter # of person to run: '))
    main(num)
