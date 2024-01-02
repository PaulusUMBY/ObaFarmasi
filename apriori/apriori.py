import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from util.util import Util 

class Apriori:

    @staticmethod
    def transform_dataset(dataframe):
        # Apriori
        # Transformasi dataset
        return dataframe.groupby(['nofarmasi', 'nama_brg'])['nama_brg'].count().reset_index(name='Number of nama_brgs')
        # table = nofarmasis.pivot_table(index='nofarmasi', columns='nama_brg', values='Number of nama_brgs', aggfunc='sum').fillna(0)

    @staticmethod
    def sort_datasetCount(table):
        return table.groupby(['nofarmasi','nama_brg'])['nama_brg'].count().reset_index(name ='Count')

    @staticmethod
    def apply_hotEncode(apriori_basket):
        return apriori_basket.applymap(Util.hot_encode)

    @staticmethod
    def frequent_patterns_support(apriori_basket_set):
        # Pencarian frequent itemset berdasarkan support
        frequence = apriori(apriori_basket_set.astype('bool'), min_support=0.03, use_colnames=True)
        return frequence

    @staticmethod 
    def assosiationRules(frequence):
        rules = association_rules(frequence, metric="lift", min_threshold=1)
        rules.sort_values('confidence', ascending = False, inplace = True)
        return rules

    @staticmethod
    def frequent_patterns_lift(frequence):
        # Pencarian frequent itemset berdasarkan lift
        return association_rules(frequence, metric = 'lift', min_threshold = 0.01)

    @staticmethod
    def sortAprioriRules(apriori_rules):
        apriori_rules.sort_values('confidence', ascending = False, inplace = True)
        return apriori_rules

    @staticmethod 
    def result_fix_rule(apriori_rules):
        fix_rule = apriori_rules[['antecedents', 'consequents']]
        return fix_rule


