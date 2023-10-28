import pandas as pd
from chembl_webresource_client.new_client import new_client


data = new_client.target
data_query = data.search('acetylcholinesterase')
targets = pd.DataFrame.from_dict(data_query)



target = new_client.target
target_query = target.search('aromatase')
targets = pd.DataFrame.from_dict(target_query)



"""
Search for Target protein
Target search for coronavirus

Here, we will retrieve only bioactivity data for coronavirus 3C-like proteinase (CHEMBL3927)(selected target) that are reported as IC values in nM (nanomolar) unit.
"""

selected_target = targets.target_chembl_id[4]
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df = pd.DataFrame.from_dict(res)
df2 = df[df.standard_value.notna()]



"""
The bioactivity data is in the IC50 unit. Compounds having values of less than 1000 nM will be considered to be active while those greater than 10,000 nM will be considered to be inactive. As for those values in between 1,000 and 10,000 nM will be referred to as intermediate.
"""


bioactivity_class = []
for i in df2.standard_value:
  if float(i) >= 10000:
    bioactivity_class.append("inactive")
  elif float(i) <= 1000:
    bioactivity_class.append("active")
  else:
    bioactivity_class.append("intermediate")



selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
df3 = df2[selection]
df3


pd.concat([df3,pd.Series(bioactivity_class)], axis=1)

df3.to_csv('bioactivity_preprocessed_data.csv', index=False)