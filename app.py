from flask import Flask, render_template, jsonify
from random import *
from flask_cors import CORS
import requests
from flask import request
from collections import Counter
import csv
import numpy as np
import pandas as pd

app = Flask(__name__,
			static_folder = "./dist/static",
			template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/')
def index():
	# start_capture()
	return render_template("hello.html")

@app.route('/api/getGenders')
def getGenders():
	classes = ["Male", "Female"]
	maleCount, femaleCount = 0, 0
	df = pd.read_csv('demographic.csv')
	counts = df['Gender'].value_counts(sort=False)
	gender = []

	totalCount = df.shape[0]
	for k in counts.to_dict():
		d = {}
		d['y'] = counts[k].item()
		d['label'] = k
		gender.append(d)

	bins = [0, 6, 12, 18, 26, 35, 50, 70,100]
	df['ageBuckets'] = pd.cut(df['Age'],bins)

	bucketCount = df['ageBuckets'].value_counts(sort=False)
	ageBuckets = []
	for k in bucketCount.to_dict():
		d = {}
		d['y'] = int(bucketCount[k])
		d['label'] = "{}-{}".format(k.left,k.right)
		ageBuckets.append(d)
	return jsonify({'Gender': gender, 'Age': ageBuckets, 'count': str(totalCount)})

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8981, debug=True)