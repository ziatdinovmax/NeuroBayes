import requests
from io import StringIO
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from library import Library
import os.path

pd.options.mode.chained_assignment = None

# TO DO: Clean up process method of HTEM

# Featurizers
def simple_molecule_featurizer(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return {
                'MolWt': Descriptors.ExactMolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            }
        return None

class FreeSolv():
    def __init__(self):
        self.target = 'experimental value (kcal/mol)'
        self.npz = "freesolv.npz"
        if os.path.isfile(self.npz):
            data = np.load(self.npz)
            self.X = data["features"]
            self.y = data["targets"]
        else:
            self.X = None
            self.y = None
            data = self.download()
            self.featurize(data)
        return

    def download(self):
        url = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
    
        response = requests.get(url)
        response.raise_for_status()
    
        # Split the content into lines
        lines = response.text.split('\n')
    
        # Find the line with column descriptions
        header_line = next(line for line in lines if line.startswith('# compound id'))
    
        # Extract and clean column names
        column_names = [name.strip() for name in header_line.strip('# ').split(';')]
    
        # Join the data lines
        data_content = '\n'.join(line for line in lines if not line.startswith('#'))
    
        # Read the data into a DataFrame
        original_data = pd.read_csv(StringIO(data_content), delimiter=';', names=column_names, skipinitialspace=True)
        original_data.dropna(subset=self.target, inplace=True)    
        return original_data

    def featurize(self, df):
        feature_list = df['SMILES'].apply(simple_molecule_featurizer)
        descriptors_df = pd.DataFrame(feature_list.tolist())
        featurized_data = pd.concat([df, descriptors_df], axis=1)
        self.X = featurized_data[descriptors_df.columns]
        self.y = featurized_data[self.target] 
        return
    
    def save(self):
        np.savez(self.npz, features=self.X.values, targets=self.y.values)
        return


class ESOL():
    def __init__(self):
        self.target = 'measured log solubility in mols per litre'
        self.npz = "esol.npz"
        if os.path.isfile(self.npz):
            data = np.load(self.npz)
            self.X = data["features"]
            self.y = data["targets"]
        else:
            self.X = None
            self.y = None
            data = self.download()
            self.featurize(data)
        return

    def download(self):
        url = "https://raw.githubusercontent.com/peastman/delaney-processed/master/delaney-processed.csv"
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Read the CSV directly
        original_data = pd.read_csv(StringIO(response.text))
        original_data.dropna(subset=[self.target], inplace=True)
        return original_data

    def featurize(self, df):
        feature_list = df['smiles'].apply(simple_molecule_featurizer)
        descriptors_df = pd.DataFrame(feature_list.tolist())
        featurized_data = pd.concat([df, descriptors_df], axis=1)
        self.X = featurized_data[descriptors_df.columns]
        self.y = featurized_data[self.target]
        return
    
    def save(self):
        np.savez(self.npz, features=self.X.values, targets=self.y.values)
        return
    
    
class FatigueStrength():
    def __init__(self):
        self.original_data = pd.read_excel('files/Steel_Fatigue_Strength_IMMI2014.xlsx', sheet_name = 'nims.csv')
        self.featurized_data = self.original_data # will fix this later once I add magpie featurizer
        self.X = self.featurized_data[ self.featurized_data.columns[1:-1] ]
        self.y = self.featurized_data["Fatigue"]
        self.target = "Fatigue"
        return

    def save(self):
        np.savez("fatigue", features=self.X.values, targets=self.y.values)
        return

class HTEM():
    def __init__(self):
        self.target = 'fpm_conductivity'
        self.npz = "htem.npz"
        if os.path.isfile(self.npz):
            data = np.load(self.npz)
            self.X = data["features"]
            self.y = data["targets"]
        else:
            self.X = None
            self.y = None
            data = self.download()
            self.process(data)
        return

    def download(self):
        # Make a GET request to the API
        response = requests.get("https://htem-api.nrel.gov/api/sample_library/")
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            library_data = response.json()
        
        else:
            #print("Error:", response.status_code)
            return

        htem_library_df = pd.DataFrame(library_data)
        ele_library_df = htem_library_df[ (htem_library_df['has_ele'] != 0) & (htem_library_df['has_xrf'] != 0) ]
        ni_ele_library_df = ele_library_df[ele_library_df['elements'].apply(lambda x: 'Ni' in x)]
        sample_list = []
        for library_id in ni_ele_library_df.id: 
            for sample_id in list(Library(library_id).properties().sample_ids)[0]:
                
                url = 'https://htem-api.nrel.gov/api/sample/'+str(sample_id)
                response = requests.get(url)
                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the JSON response
                    sample_data = response.json()
        
                    # Access the data
                    sample_list.append( sample_data )
        
                #else:
                    #print("Error:", response.status_code)        
        ni_ele_sample_df = pd.DataFrame(sample_list)
        ni_ele_sample_df = ni_ele_sample_df[~ni_ele_sample_df['xrf_elements'].isin([ ['Ni'], ['Sn', 'Zn'] ])]
        ni_ele_library_df = ni_ele_library_df.rename(columns = {"id":"sample_library_id"})
        ni_merged_df = ni_ele_sample_df.merge(ni_ele_library_df, on="sample_library_id")
        ni_merged_df = ni_merged_df[ni_merged_df[ 'fpm_conductivity' ] .notna()]
        return ni_merged_df

    def process(self, df):
        ternary_df = df[df['xrf_compounds'].apply(lambda x: 'NiO' in x and 'CoO' in x and 'ZnO' in x)].reset_index()
        co_zn_binary_df = df[df['xrf_compounds'].apply(lambda x: 'CoO' in x and 'ZnO' in x and 'NiO' not in x)].reset_index()
        ni_co_binary_df = df[df['xrf_compounds'].apply(lambda x: 'CoO' in x and 'NiO' in x and 'ZnO' not in x)].reset_index()

        ternary_df.loc[:,'dict'] = ternary_df[['xrf_compounds', 'xrf_concentration']].apply(lambda data: {k: [y for x, y in zip(data[0], data[1]) if x == k] for k in data[0]}, axis=1)
        ternary_df = pd.concat( (ternary_df, pd.json_normalize(ternary_df['dict'])), axis=1 )
        co_zn_binary_df.loc[:,'dict'] = co_zn_binary_df[['xrf_compounds', 'xrf_concentration']].apply(lambda data: {k: [y for x, y in zip(data[0], data[1]) if x == k] for k in data[0]}, axis=1)
        co_zn_binary_df = pd.concat( (co_zn_binary_df, pd.json_normalize(co_zn_binary_df['dict'])), axis=1 )
        ni_co_binary_df.loc[:,'dict'] = ni_co_binary_df[['xrf_compounds', 'xrf_concentration']].apply(lambda data: {k: [y for x, y in zip(data[0], data[1]) if x == k] for k in data[0]}, axis=1)
        ni_co_binary_df = pd.concat( (ni_co_binary_df, pd.json_normalize(ni_co_binary_df['dict'])), axis=1 )

        co_zn_binary_df.loc[:,'NiO'] = [0.0]*len(co_zn_binary_df)
        ni_co_binary_df.loc[:,'ZnO'] = [0.0]*len(ni_co_binary_df)

        X_columns = ['thickness', 'absolute_temp_c', 'deposition_sample_time_min',
                     'deposition_base_pressure_mtorr', 'deposition_initial_temp_c', 'sciround', 'NiO', 'CoO', 'ZnO']

        conc_df = pd.concat((ternary_df, co_zn_binary_df, ni_co_binary_df))

        dep_power_columns = [ f'dep_power_{i}' for i in range(3) ]
        conc_df = conc_df[conc_df['deposition_power'].notna()]
        dep_power_df = pd.DataFrame( conc_df.deposition_power.to_list(), columns = dep_power_columns )
        
        dep_flow_columns = [ f'dep_flow_{i}' for i in range(3) ]
        conc_df = conc_df[conc_df['deposition_gas_flow_sccm'].notna()]
        dep_flow_df = pd.DataFrame( conc_df.deposition_gas_flow_sccm.to_list(), columns = dep_flow_columns )

        X = pd.concat( (conc_df[X_columns].reset_index(), dep_power_df, dep_flow_df), axis=1 ).drop(columns=['dep_flow_2'])
        X['dep_power_2'] = X['dep_power_2'].fillna(0.0)
        X['NiO'] = X['NiO'].apply(lambda x: x[0] if type(x) == list else x )
        X['CoO'] = X['CoO'].apply(lambda x: x[0] if type(x) == list else x )
        X['ZnO'] = X['ZnO'].apply(lambda x: x[0] if type(x) == list else x )

        self.X = X.drop(columns = ['absolute_temp_c', 'dep_flow_0', 'index']) 
        self.y = conc_df[self.target]
        return

    def save(self):
        np.savez(self.npz, features=self.X.values, targets=self.y.values)
        return


def Dataset(name="fatigue strength"):
    name = name.lower()
    datasets = {
        "freesolv": FreeSolv,
        "esol": ESOL,
        "fatigue strength": FatigueStrength,
        "htem": HTEM,
    }
    return datasets[name]()
