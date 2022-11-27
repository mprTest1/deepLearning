import os
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(batch_size=4):
    flowers_dir = 'data/flowers/'
    file_path = os.listdir(flowers_dir)
    total = 0
    for i in file_path:
        dir = os.path.join(flowers_dir, i)
        num = len(os.listdir(dir))
        print(i + ':', num, '\n')
        total = total + num
    print('图片总数为:', total)
    img_size = (224, 224)
    batch_size = batch_size
    #数据生成，划分训练集和测试集
    img_dataset = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2)
    #加载训练集
    train_loader = img_dataset.flow_from_directory(
        flowers_dir,
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='training')
    #加载测试集
    test_loader = img_dataset.flow_from_directory(
        flowers_dir,  # 测试集目录
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical',
        subset='validation')
    return train_loader,test_loader