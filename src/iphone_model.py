'''
To run this script you need to use Python 2.7
'''
from coremltools import converters
from PIL import Image

if __name__ == '__main__':
    keras_model = 'mymodel-2.h5'
    coreml_model = converters.keras.convert(keras_model, input_names='image',
                                            image_input_names='image',
                                            class_labels=['malign'])
    img = Image.open('test-2.png')
    pred = coreml_model.predict({'image':img})
    print('Testing Model...')
    print('Probability of image to be malign {:.2f}%'
                .format(pred['output1']['malign']*100))
    filename = 'Model1-IphoneCNNfromScratch.mlmodel'
    coreml_model.author = 'David Soto'
    coreml_model.short_description = 'CNN model to predict the probability of malign moles on skin'
    coreml_model.license = 'BSD'
    coreml_model.save(filename)
    print('Model saved as: ', filename)
