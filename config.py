class Config(object):
    IMAGE_MIN_SIZE=512
    IMAGE_MAX_SIZE=512
    if(IMAGE_MIN_SIZE%2**5 !=0):
        print('image min size must can be divided by 2**5')
    if(IMAGE_MAX_SIZE%2**5 !=0):
        print('image max size must can be divided by 2**5')

    conv_weight_normal=0.2

    root_dir='/home/mameng/dataset/HED/HED-BSDS'

    batch_size=1
    iter_size=10
    eopch_update_lr=10
    max_epoch=35
    display=20
    lr=3e-5
    momentum=0.9
    weight_decay=0.0002

    gpu=True

    model_save_path='./snapshot'


cfg=Config()
