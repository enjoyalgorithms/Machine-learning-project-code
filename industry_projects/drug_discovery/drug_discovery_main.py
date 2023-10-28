import pandas as pd
from chembl_webresource_client.new_client import new_client
data = new_client.target
data_query = data.search('acetylcholinesterase')
targets = pd.DataFrame.from_dict(data_query)
data_features = pd.DataFrame(targets.data , columns=targets.feature_names)
data_new = new_client.activity
data1 = data_new.filter(target_chembl_id='CHEMBL220').filter(standard_type="IC50")
df = pd.DataFrame.from_dict(data1)
#dropping records which donot have values in columns standard_value and canonical_smiles
df2 = df[df.standard_value.notna()]
df2 = df2[df2.canonical_smiles.notna()]
#dropping records with duplicate canonical_smiles values to keep them unique
df2_unique = df2.drop_duplicates(['canonical_smiles'])
selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df3 = df2_unique[selection]
bioactivity_threshold = []
for i in df3.standard_value:
  if float(i) >= 10000:
    bioactivity_threshold.append("inactive")
  elif float(i) <= 1000:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")
bioactivity_class = pd.Series(bioactivity_threshold, name='bioactivity_class')
df4 = pd.concat([df3, bioactivity_class], axis=1)
import numpy as np
def pIC50(input):
	pIC50 = []
	count = 0
	for i in input['standard_value_norm']:
		try:
			molar = i*(10**-9) # Converts nM to M
			if(molar==0):
				molar = 10**-10
				count = count + 1
			pIC50.append(-np.log10(molar))
		except:
			pIC50.append("error")
	input['pIC50'] = pIC50
	x = input.drop('standard_value_norm', 1)        
	return x
def norm_value(input):
    norm = []
    for i in input['standard_value']:
        if float(i) > 100000000:
          i = 100000000
        norm.append(float(i))
    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)       
    return x
df_norm = norm_value(df4)
df_norm.standard_value_norm.describe()
df_final = pIC50(df_norm)
df_final.pIC50.describe()
df_2class = df_final[df_final.bioactivity_class != 'intermediate']
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x='bioactivity_class', data=df_2class, edgecolor='blue')
plt.xlabel('Bioactivity_class', fontsize=12)
plt.ylabel('Frequency of occurence', fontsize=12)
plt.savefig('plot_bioactivity_class.pdf')
selection = ['canonical_smiles','molecule_chembl_id']
df3_selection = df_final[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)
###after padel.sh is done we follow commands here
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
df = pd.read_csv('padel/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')
X = df.drop('pIC50', axis=1)
Y = df.pIC50
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
xgb_r = xg.XGBRegressor(objective ='reg:linear',n_estimators = 10, seed = 123)
xgb_r.fit(X_train, Y_train)
Y_pred = xgb_r.predict(X_test)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
sns.set_style("white")
ax = sns.regplot(Y_test, Y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)
plt.show
plt.savefig('prediction_plot.pdf')
#reg:squarederror
#here we try to plot model again with different squared error
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
df = pd.read_csv('padel/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')
X = df.drop('pIC50', axis=1)
Y = df.pIC50
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
xgb_r = xg.XGBRegressor(objective ='reg:squarederror',n_estimators = 10, seed = 123)
xgb_r.fit(X_train, Y_train)
Y_pred = xgb_r.predict(X_test)
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.regplot(Y_test, Y_pred)
ax.set_xlabel('Experimental_pIC50')
ax.set_ylabel('Predicted_pIC50')
plt.savefig('prediction_plot2.pdf')
#sns.set(color_codes=True)
#sns.set_style("white")
ax = sns.regplot(Y_test, Y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)
plt.show
plt.savefig('prediction_plot1.pdf')




###Drawing data analysis graphs
import matplotlib.pyplot as plt
df.hist()
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
plt.gcf().set_size_inches(15, 15)
cmap=sns.diverging_palette(500,10,as_cmap=True)
sns.heatmap(df.corr(),cmap=cmap,center=0,annot=False,square=True)
sns.heatmap(df.corr(),cmap='YlGnBu',center=0,square=True)
plt.show()
###convert data type from object to categorical to view further
df10 = df
error = 0
count = 0

for col in df.columns:
	if(df[col].dtype == 'object'):
		try:
			df[col] = df[col].astype('category')
			df[col] = df[col].cat.codes
			count = count + 1
		except :
			error = error + 1

"""
import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {'active':2400, 'inactive':1900}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (7, 7))

# creating the bar plot
plt.bar(courses, values,color=['#1f77b4', '#ff7f0e'])

plt.xlabel("bioactivity_class")
plt.ylabel("Frequency of occurence")
# plt.title("Students enrolled in different courses")
plt.savefig('bar.png')
plt.show()
"""