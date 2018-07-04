from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
import subprocess
import collab7

app = Flask(__name__)

# app.config['MONGO_DBNAME'] = 'recDB'
# app.config['MONGO_URI'] = 'mongodb://localhost:27017/recDB'

mongo = PyMongo(app)


@app.route('/script/<user_id>/<item_id>',methods = ['GET'])
def subprocess_output(user_id,item_id):
  output = subprocess.run(['python', 'collab6.py',user_id,item_id] ,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)#.decode('utf-8')
  # print(str(output.stdout))
  # return output.stdout
  return jsonify(str(output.stdout))



@app.route('/rec/<user_id>', methods=['GET'])
@app.route('/rec/<user_id>/<item_id>', methods=['GET'])
def get_recs(user_id,item_id = None):
  x = collab7.rec(user_id,item_id)
  # print(x)
  return jsonify(x)
# @app.route('/star/', methods=['GET'])
# def get_one_star(name):
#   star = mongo.db.stars
#   s = star.find_one({'name' : name})
#   if s:
#     output = {'name' : s['name'], 'distance' : s['distance']}
#   else:
#     output = "No such name"
#   return jsonify({'result' : output})

# @app.route('/star', methods=['POST'])
# def add_star():
#   star = mongo.db.stars
#   name = request.json['name']
#   distance = request.json['distance']
#   star_id = star.insert({'name': name, 'distance': distance})
#   new_star = star.find_one({'_id': star_id })
#   output = {'name' : new_star['name'], 'distance' : new_star['distance']}
#   return jsonify({'result' : output})

if __name__ == '__main__':
    app.run(debug=True)