import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import seaborn as sns
from dataset import *
#获取VGG16模型
def get_model_vgg():
    input_shape = (224, 224, 3)
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    print(model.summary())
    return model
#训练模型
def train_model(train_loader, test_loader,model_sava_path,batch_size=4,epochs=10):
    model = get_model_vgg()
    try:
        model.load_weights(model_sava_path)
        print('加载模型权重继续训练模型')
    except:
        print('开始训练新模型')
    checkpoints = ModelCheckpoint(model_sava_path, monitor='val_accuracy',
                                  mode='max', save_best_only=True)
    history = model.fit(
        train_loader,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=test_loader,
        verbose=1,
        callbacks=[checkpoints]
    )
    plot_history(history)
#画出history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 画出history
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    plt.plot(epochs, loss_train, label='Training Loss')
    plt.plot(epochs, loss_val, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('model/loss.png')
    # plt.show()
#测试模型并且评估模型
def test_model(testloader,model_sava_path,batch_size=4):
    # 加载模型权重
    model = get_model_vgg()
    steps = test_loader.n / batch_size
    model.load_weights(model_sava_path)
    testloader.reset()
    # 模型预测
    pred = model.predict(testloader, steps=steps, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = (testloader.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    actual_class = testloader.classes
    actual = [labels[k] for k in actual_class]
    val_labels = [fn.split('/')[0] for fn in actual]

    flowers_names = list(testloader.class_indices.keys())
    cm = confusion_matrix(val_labels, predictions)
    # 画出混淆矩阵
    plt.figure(figsize=(12, 9))
    fig = sns.heatmap(cm, annot=True)
    fig.set(title='Confusion Matrix', xlabel='Labels', ylabel='True Labels')
    fig.set_xticklabels(flowers_names, rotation=20)
    fig.set_yticklabels(flowers_names, rotation=20)
    plt.show()

    # 模型评估
    report = classification_report(val_labels, predictions, digits=4)
    print("\n---分类报告---\n")
    print(report)
#根据输入文件夹 data/test/valid/文件夹下面的花判断花的品种，里面可以放一张图片，运行次程序可以测试需要判断的图片的类别
def test_picture(testloader,model_sava_path,flows_dir,batch_size=1):
    # 加载模型权重
    model = get_model_vgg()
    model.load_weights(model_sava_path)
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
    img_size = (224, 224)
    test_data = img_dataset.flow_from_directory(
        flows_dir,  # 测试集目录
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical',
        )
    # 模型预测
    pred = model.predict(test_data,steps=1, verbose=1)
    predicted_class = np.argmax(pred, axis=1)
    labels = (testloader.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class]
    print(predictions)
if __name__=='__main__':
    batch_size=4
    model_sava_path = 'model/mymodel_vgg.hdf5'
    # 是否训练模型
    train = False
    # 是否测试模型
    test = True
    # 是否使用图片文件夹测试
    test_pic = True
    train_loader, test_loader = get_data()
    if train:
        train_model(train_loader, test_loader, model_sava_path, batch_size=batch_size)
    if test:
        test_model(test_loader, model_sava_path, batch_size=batch_size)
    if test_pic:
        flows_dir = 'data/test/'
        test_picture(test_loader, model_sava_path, flows_dir)