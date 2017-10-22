from moleimages import MoleImages

def resize_images():
    print('Resizing Benign')
    moles = MoleImages('data/benign/*.jpg')
    benigns = moles.resize_bulk()
    moles.save_png(benigns, 'data_scaled/benign', tag='img-st3')

    print('Resizing Malign')
    moles = MoleImages('data/malignant/*.jpg')
    malignants = moles.resize_bulk()
    moles.save_png(malignants,'data_scaled/malign', tag='img-st3')

if __name__ == '__main__':
    resize_images()
