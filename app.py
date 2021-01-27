import csv
from flask import Flask, request
import turicreate as tc
import pandas as pd

app = Flask(__name__)


def helper(v):
    data_frame = pd.read_json(v)
    transposed_data = data_frame.T
    f = open('users_data.csv', 'w')
    f.write(transposed_data.to_csv(index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC))
    f.close()


@app.route("/api", methods=["POST", "GET"])
def get_recommendations():
    incoming_user_data = request.form['data']
    helper(incoming_user_data)
    train_data_df = pd.read_csv('users_data.csv', sep=',')
    test_data_df = pd.read_csv('places_togo.csv', sep=',')
    # Convert the pandas dataframes to graph lab SFrames
    train_data_df['Latitude'] = pd.to_numeric(train_data_df["Latitude"])
    test_data_df['Latitude'] = pd.to_numeric(test_data_df["Latitude"])
    train_data = tc.SFrame(train_data_df)
    test_data = tc.SFrame(test_data_df)
    data = train_data + test_data
    model = tc.ranking_factorization_recommender.create(data, 'ID', 'PlaceID')
    results = model.recommend()
    first_place = list(data.filter_by(values=results[0]['PlaceID'], column_name='PlaceID'))
    second_place = list(data.filter_by(values=results[1]['PlaceID'], column_name='PlaceID'))
    third_place = list(data.filter_by(values=results[2]['PlaceID'], column_name='PlaceID'))
    return {'0': first_place, '1': second_place, '2': third_place}


if __name__ == "__main__":
    app.run()
