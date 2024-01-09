import pandas as pd

class Util:
    
    @staticmethod
    def removewhitespaces(data):
        newdataframe = [ x.rstrip() for x in data ]
        return newdataframe

    @staticmethod
    def normalisasidata(dataframe):
        dataframe['nofarmasi'] = dataframe['nofarmasi'].astype('Int64')
        # Normalisasi format data nama_brg
        dataframe['nama_brg'] = dataframe['nama_brg'].str.lower()
        dataframe['nama_brg'] = dataframe['nama_brg'].str.strip()
        return dataframe

    @staticmethod
    def formatdate(dataframe):
        # Memecah format date-time
        dataframe['Datetime'] = pd.to_datetime(dataframe['tglsave'])
        dataframe['date'] = dataframe['Datetime'].dt.date
        dataframe['month'] = dataframe['Datetime'].dt.month
        dataframe['day'] = dataframe['Datetime'].dt.weekday
        dataframe['hour'] = dataframe['Datetime'].dt.hour

        dataframe['day'] = dataframe['day'].replace((0,1,2,3,4,5,6), 
        ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))
        dataframe.drop(columns='tglsave',inplace=True)
        return dataframe

    @staticmethod 
    def transactionGroupbyHours(dataframe):
        # Exploratory Analysis
        # Transaksi berdasarkan jam
        countByHour = dataframe.groupby('hour')['nofarmasi'].count().reset_index()
        return countByHour.sort_values('hour',inplace=True)

    @staticmethod 
    def transactionGroupbyWeeks(dataframe):
        # Transaksi per hari dalam seminggu
        countByDay = dataframe.groupby('day')['nofarmasi'].count().reset_index()
        countByDay.loc[:,"orderOfDays"] = [4,0,5,6,3,1,2]
        return countByDay.sort_values("orderOfDays",inplace=True)
       
    @staticmethod 
    def transactionGroupbyMonth(dataframe):
        countByMonth = dataframe.groupby('month')['nofarmasi'].count().reset_index()
        return countByMonth.sort_values('month',inplace=True)

    @staticmethod 
    def top25(dataframe):
        values = dataframe.nama_brg.value_counts().head(25)
        return values

    @staticmethod
    def hot_encode(x): 
     if(x==0): 
        return 0
     if(x>0): 
        return 1

    @staticmethod
    def drop_axis(dataframe):
        return dataframe.drop(['-','--','7'], axis=1)






