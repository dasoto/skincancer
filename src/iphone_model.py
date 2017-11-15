'''
To run this script you need to use Python 2.7
'''
from coremltools import converters
from PIL import Image
import sys

if __name__ == '__main__':
    if len(sys.argv)==3:
        keras_model = sys.argv[1]
        coreml_model = converters.keras.convert(keras_model, input_names='image',
                                                image_input_names='image',
                                                class_labels=['malign'])
        img = Image.open('images/test.png')
        pred = coreml_model.predict({'image':img})
        print('Testing Model...')
        print('Probability of image to be malign {:.2f}%'
                    .format(pred['output1']['malign']*100))
        filename = sys.argv[2]
        coreml_model.author = 'David Soto'
        coreml_model.short_description = 'CNN model to predict the probability of malign moles on skin'
        coreml_model.license = 'BSD'
        coreml_model.save(filename)
        print('Model saved as: ', filename)
    else:
        print('Use: python iphone_model.py input_model output_model')
