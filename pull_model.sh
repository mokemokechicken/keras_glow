rsync --exclude 'data/sample' --exclude '.idea/' --exclude '*.pyc' --exclude data/image/ --exclude log/ -ruv "ml:/home/data_lab/keras_glow/" ./
