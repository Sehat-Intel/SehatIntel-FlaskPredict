# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:40:46 2021

@author: KrunalV
"""

from __future__ import division, print_function
import os
import numpy as np
import cv2

from pymongo import MongoClient
from bson.objectid import ObjectId
import predict

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# from flask_bootstrap import Bootstrap
# from flask_wtf import FlaskForm
# from flask_wtf.file import FileField, FileRequired, FileAllowed
# from wtforms import SubmitField

from pymongo import MongoClient
from bson.objectid import ObjectId
# Define a flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
client = MongoClient('localhost', 27017)
db = client['SehatIntel']
datas_collection = db['datas']
users_collection = db['users']


@app.route('/image', methods=['POST', 'GET'])
def pymongo_test():
    if request.method == 'POST':
        request_data = request.get_json()
        user_data = datas_collection.find_one({'_id': ObjectId(request_data['id'])})
        if( user_data):
            prediction = predict.main(user_data['labReportFileUrl'])
            filter = {
                '_id':  ObjectId(request_data['id'])
            }
            new_values = {
                "$set": {
                    'labReportDiagnosistics': str(prediction)
                }
            }
            datas_collection.update_one(filter, new_values)
            return "data updated"
        else:
            return 'Try again with different object id'
    else:
        data = []
        cursor  = datas_collection.find()
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            data.append(doc)
        return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)