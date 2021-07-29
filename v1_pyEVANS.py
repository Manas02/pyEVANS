#! /usr/bin/env python
# Author : Manas Mahale <manas.mahale@bcp.edu.in>

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChemC
import argparse


print("""\t\t██████  ██    ██ ███████ ██    ██  █████  ███    ██ ███████
\t\t██   ██  ██  ██  ██      ██    ██ ██   ██ ████   ██ ██
\t\t██████    ████   █████   ██    ██ ███████ ██ ██  ██ ███████
\t\t██         ██    ██       ██  ██  ██   ██ ██  ██ ██      ██
\t\t██         ██    ███████   ████   ██   ██ ██   ████ ███████ """)


def run(args):
	filename = args.input # these match the "dest": dest="input"
	output_filename = args.output # from dest="output"
	data = pd.read_csv(filename, sep='\t').filter(['Ligand SMILES', 'IC50 (nM)'])
	lig, ic50 = data['Ligand SMILES'].to_list(), data['IC50 (nM)'].to_list()
	mols = [Chem.AddHs(Chem.MolFromSmiles(i)) for i in lig]
	m = [Chem.MolFromSmiles(i) for i in lig]
	m2 = [Chem.AddHs(i) for i in m]
	[AllChem.EmbedMolecule(i, randomSeed=0xf00d) for i in m2]
	[AllChem.MMFFOptimizeMolecule(i) for i in m2]
	with Chem.SDWriter(f'{output_filename}.sdf') as f:
	    for m in m2:
	        f.write(m)
	print("Done !!")

def main():
	parser=argparse.ArgumentParser(description="pyEVANS : Eigen Value ANalySis (EVANS) ‒ A Tool to Address Pharmacodynamic, Pharmacokinetic and Toxicity Issues")
	parser.add_argument("-in",help="Smiles input file" ,dest="input", type=str, required=True)
	parser.add_argument("-out",help="SDF output filename" ,dest="output", type=str, required=True)
	parser.set_defaults(func=run)
	args=parser.parse_args()
	args.func(args)


if __name__=="__main__":
	main()
