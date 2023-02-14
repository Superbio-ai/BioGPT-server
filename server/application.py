from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from flask_bcrypt import Bcrypt

from log import set_logging
from routes import init_routes

application = Flask(__name__)
CORS = CORS(application)
API = Api(application)
bcrypt = Bcrypt(application)

init_routes(API)
set_logging(application)

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8001)
