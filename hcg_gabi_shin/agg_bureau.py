import numpy as np
import pandas as pd

os.chdir('/home/tiago/Documents/HCG_competition')


train = pd.read_csv('HCG_data/application_train.csv')
test = pd.read_csv('HCG_data/application_test.csv')
br = pd.read_csv('HCG_data/bureau.csv')

functions = ["count", "first", "last", "mode", "sum", "mean", "mad", "min", "max", "std"]

d_cat_func = {"object": ["count", "first", "last", "mode"],
				"float64": ["count", "first", "last", "sum", "mean", "mad", "min", "max", "std"],
				"int64": ["count", "first", "last", "mode", "sum", "min", "max"]}


drop_bureau = ["SK_ID_BUREAU"]

def agg_bureau(maindf, aggdf, drop_list, functions):

	aggdf = aggdf.drop(drop_list, axis = 1)
	gb_df = aggdf.groupby("SK_ID_CURR")
	maindf = maindf.set_index("SK_ID_CURR")

	for func in functions: 
		df_gb_func = gb_df.agg(func)

		for coluna in list(aggdf.columns)[1:-1]:
			if func is not in d_cat_func[str(dtype(aggdf.coluna))]:
			df_gb_func = df_gb_func.drop(coluna)

		for coluna in list(df_gb_func.columns):
			df_gb_func = df_gb.func.rename({coluna: "{}{}{}".format(coluna,"_", func)


		maindf = maindf.join(df_gb) #testar se nao merge
	