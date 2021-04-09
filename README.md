# Getting started

## Setup DVC to fetch models

1. Install DVC: https://dvc.org/doc/install
2. Clone the repository.
3. Run the following commands:   

> `dvc pull model_resnet.hdf5` 

> `dvc pull vgg16-model-19-0.71.hdf5`

The models will be fetched from Gdrive. 

## Run Flask App

```shell

set FLASK_APP=app.py
set FLASK_ENV=development
python app.py

```
Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

Note: Images for testing purpose are located in 'static/' dir
