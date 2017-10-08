def define_model(shp, num_words):
    from keras.layers import Activation, Dense, Dropout, Embedding, Flatten
    from keras.regularizers import l2
    from keras.models import Sequential
    model = Sequential()

    class neurons:
        c = shp



    # model.add(Embedding(num_words, 128, input_length=shp)) #pretty damn awesome.
    model.add(Embedding(num_words, 30, input_length=shp))

    model.add(Activation('relu'))
    model.add(
        Dense(
            shp,
            input_shape=(128,),
            kernel_regularizer=l2(0.003),
            bias_regularizer=l2(0.003),
            activity_regularizer=l2(0.003)
        )
    )
    model.add(Flatten())
    #flatten after embedding instead of before softmax: 94 
    #128,32,4,2': 91
    #64,32,4,2': 86

    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.add(
        Dense(
            2,
            kernel_regularizer=l2(0.003),
            bias_regularizer=l2(0.003),
            activity_regularizer=l2(0.003)
        )
    )
    model.add(Activation('relu'))
    model.add(Dense(2))
    # kernel_regularizer=l2(0.003),
    # activity_regularizer=l2(0.003)))
    model.add(Activation('softmax'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['accuracy']
    )

    return model