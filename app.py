from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import json

from flask_wtf.csrf import CSRFProtect, CSRFError

import os
from dotenv import load_dotenv, dotenv_values

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from util import Util 
from apriori import Apriori


load_dotenv()

env = os.getenv('FLASK_ENV')
secrets = os.getenv('SECRET_KEY')

app = Flask(__name__)

app.config['SECRET_KEY'] = secrets
app.config['FLASK_ENV'] = env

csrf = CSRFProtect(app)
csrf.init_app(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/loadpriview', methods=['POST'])
def loadview():
    df = pd.read_csv('./data/ob2023.csv')
    df['nama_brg']=Util.removewhitespaces(df['nama_brg'])
    return render_template('index.html', totalRow=len(df.axes[0]), preview=[df.head().to_html(classes="table", header='true')])

@app.route('/hitung', methods=['POST'])
def hitung():
    df = pd.read_csv('./data/ob2023.csv')
    df['nofarmasi']=='0DOT'
    df['nama_brg']=Util.removewhitespaces(df['nama_brg'])
    # df = Util.normalisasidata(df)
    # df = Util.formatdate(df)
    # groupbyHours = Util.transactionGroupbyHours(df)
    # groupbyWeeks = Util.transactionGroupbyWeeks(df)
    # groupbyMonth = Util.transactionGroupbyMonth(df)
    # top25 = Util.top25(df)
    # Apriori
    nofarmasis = Apriori.transform_dataset(df)
    table = nofarmasis.pivot_table(index='nofarmasi', columns='nama_brg', values='Number of nama_brgs', aggfunc='sum').fillna(0)


    apriori_data = Apriori.sort_datasetCount(df)
    apriori_basket = apriori_data.pivot_table(index = 'nofarmasi', columns = 'nama_brg', values = 'Count', aggfunc ='sum').fillna(0)

    apriori_basket_set = Apriori.apply_hotEncode(apriori_basket)


    #drop axis
    table = Util.drop_axis(table)
    apriori_basket_set = Util.drop_axis(apriori_basket_set)


    frequent_support = Apriori.frequent_patterns_support(apriori_basket_set)

    association_rules = Apriori.assosiationRules(frequent_support)
    
    # result = Apriori.result_fix_rule(frequent_lift)
    # jsonres = df.head().to_json(orient = 'index')
    # return jsonify(json.loads(jsonres)) 
    
    #jsonres = table.head().to_json(orient = 'index')
    #return jsonify(json.loads(jsonres))

    frequent_lift = Apriori.frequent_patterns_lift(frequent_support)

    sort_apriori_rules = Apriori.sortAprioriRules(frequent_lift)

    result = Apriori.result_fix_rule(sort_apriori_rules)

    return render_template('index.html', result=[result.to_html(classes='table', header="true")])

    # jsonres = result.to_json(orient = 'index')
    # return jsonify(json.loads(jsonres))



if __name__== "__main__":
    app.run(debug=True)
