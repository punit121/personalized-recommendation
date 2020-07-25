import os
import json
import pickle
import sys
import signal
import traceback
import re
import flask


from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

prefix = "/opt/ml/"

PATH = Path(os.path.join(prefix, "model"))



class ScoringService(object):
    model = None  # Where we keep the model when it's loaded


    @classmethod
    def searching_all_files(cls, directory: Path):
        file_list = []  # A list for storing files existing in directories

        for x in directory.iterdir():
            if x.is_file():
                file_list.append(str(x))
            else:
                file_list.append(cls.searching_all_files(x))

        return file_list


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
            ScoringService.get_predictor_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


# @app.route("/execution-parameters", method=["GET"])
# def get_execution_parameters():
#     params = {
#         "MaxConcurrentTransforms": 3,
#         "BatchStrategy": "MULTI_RECORD",
#         "MaxPayloadInMB": 6,
#     }
#     return flask.Response(
#         response=json.dumps(params), status="200", mimetype="application/json"
#     )


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    K = 10  # number of recommendation to generate
    T_1 = time.time()
    ALREADY_RATED = []
    # loading numpy matrix
    USER_MAP = np.load("saved_model/user.npy", allow_pickle=True)
    ITEM_MAP = np.load("saved_model/item.npy", allow_pickle=True)
    ROW_FACTOR = np.load("saved_model/row.npy", allow_pickle=True)
    COL_FACTOR = np.load("saved_model/col.npy", allow_pickle=True)

    COL = ['categoryName', 'catId', 'subCategoryName', 'subCatId', 'productName', 'uniqueProductKey', 'productKey1',
           'productVal1']
    for client_id in USER_MAP[0:4000000]:
        user_idx = np.searchsorted(USER_MAP, client_id)  # searching index of client id
        user_rated = [np.searchsorted(ITEM_MAP, i) for i in ALREADY_RATED]
        recommendations = generate_recommendations(user_idx, user_rated, ROW_FACTOR, COL_FACTOR, K)
        at = [ITEM_MAP[i] for i in recommendations]  # mapping item index
        # formatting item in json
        json_data = []
        for line in at:
            json_data.append(dict(zip(COL, line.split(">>"))))
        for i in json_data:
            for key, val in list(i.items()):
                if val == 'NF':
                    del i[key]
        # changing data type
        res = []
        for i in json_data:
            i['catId'] = int(i['catId'])
            i['subCatId'] = int(float(i['subCatId']))
            res.append(i)
        final = {}
        final['payload'] = res
        final['size'] = len(res)
        # dummping recommendation
        json.dump(final, open(r"final_recommendation/prediction/{}.json".format(client_id), "w"))
    print("total time taken:", time.time() - T_1)

    return flask.Response(response=result, status=200, mimetype="application/json")
