import importlib
import torch
import numpy as np
from .optimization import AA_TO_I as aa_to_i
from tqdm import tqdm

potts_is_installed = importlib.util.find_spec("potts") is not None
if not (potts_is_installed):
    error_msg = '''Potts library is not installed. 
             Run: `git clone git@github.com:hnisonoff/potts.git`
                  `cd potts`
                  `pip install .`
    '''
    raise ImportError(error_msg)
else:
    from potts import Potts


def add_codon_model_constraints(model_codon, codons_with_gaps, codon_table, add_stop_codon=False):
    '''
    model_codon: Potts model defined at codon level
    codon_with_gaps: starting codon sequence 
    codon_table: dictionary mappint codons to amino acids
    add_stop_codon: whether to add a stop codon at the end of the sequence
    '''
    codons, aas = map(list, zip(*codon_table.items()))
    codon_to_aa = {c:a for c, a in zip(codons, aas)}
    codon_to_i = { c:i for i,c in enumerate(codons + ['XXX']) }

    L, A = model_codon.L, model_codon.A
    h = model_codon.h.reshape(L,A).detach().clone()
    W = model_codon.W.weight.detach().clone()
    # add methionine constraint
    methionine_idx = codon_to_i['ATG']
    for i in range(65):
        if i != methionine_idx:
            h[0, i] += 10000

    # add gap constraints
    for i, codon in enumerate(codons_with_gaps):
        if codon == 'XXX':
            gap_idx = codon_to_i[codon]        
            for idx in range(65):
                if idx != gap_idx:
                    h[i, idx] += 10000

        else:
            gap_idx = codon_to_i['XXX']        
            for idx in range(65):
                if idx == gap_idx:
                    h[i, idx] += 10000

    if add_stop_codon:
        stop_codons = [c for c, aa in codon_table.items() if aa == "*"]            
        stop_codon_idxs = [codon_to_i[c] for c in stop_codons]
        for i in range(65):
            if i not in stop_codon_idxs:
                h[-1, i] += 10000
    model_with_constraints = Potts(h=h, W=W)
    return model_with_constraints


def amino_potts_to_codon_potts_with_stop(model, codon_table):
    codons, aas = map(list, zip(*codon_table.items()))
    codon_to_aa = {c:a for c, a in zip(codons, aas)}
    codon_to_i = { c:i for i,c in enumerate(codons + ['XXX']) }

    h = model.h.reshape(
            (model.L, model.A)).detach().cpu().numpy()
    W = model.reshape_to_L_L_A_A().detach().cpu().numpy()


    codon_h = np.zeros((model.L+1, 65))
    codon_W = np.zeros((model.L+1, model.L+1, 65, 65))
    for pos in range(model.L):
        for codon_idx, codon in enumerate(codons):
            aa = codon_to_aa[codon]
            if aa == "*":
                codon_h[pos, codon_idx] = 10000
            else:
                codon_h[pos, codon_idx] = h[pos, aa_to_i[aa]]
        codon_h[pos, -1] = h[pos, aa_to_i['-']]


    for pos_i in tqdm(range(model.L-1)):
        for pos_j in range(pos_i+1, model.L):
            for codon_idx_i, codon_i in enumerate(codons):
                for codon_idx_j, codon_j in enumerate(codons):
                    aa_i = codon_to_aa[codon_i]
                    aa_j = codon_to_aa[codon_j]

                    if aa_i == "*" or aa_j == "*":
                        continue
                    aa_idx_i = aa_to_i[aa_i]
                    aa_idx_j = aa_to_i[aa_j]
                    codon_W[pos_i, pos_j, codon_idx_i, codon_idx_j] = W[pos_i, pos_j, aa_idx_i, aa_idx_j]
                    codon_W[pos_j, pos_i, codon_idx_j, codon_idx_i] = W[pos_j, pos_i, aa_idx_j, aa_idx_i]
            for codon_idx_i, codon_i in enumerate(codons):
                aa_i = codon_to_aa[codon_i]
                if aa_i == "*":
                    continue
                aa_idx_i = aa_to_i[aa_i]
                aa_idx_j = aa_to_i['-']
                codon_W[pos_i, pos_j, codon_idx_i, -1] = W[pos_i, pos_j, aa_idx_i, aa_idx_j]
                codon_W[pos_i, pos_j, -1, codon_idx_i] = W[pos_i, pos_j, aa_idx_j, aa_idx_i]

                codon_W[pos_j, pos_i, -1, codon_idx_i] = W[pos_j, pos_i, aa_idx_j, aa_idx_i]
                codon_W[pos_j, pos_i, codon_idx_i, -1] = W[pos_j, pos_i, aa_idx_i, aa_idx_j]

            aa_idx_i = aa_to_i['-']
            aa_idx_j = aa_to_i['-']
            codon_W[pos_j, pos_i, -1, -1] = W[pos_j, pos_i, aa_idx_j, aa_idx_i]
            codon_W[pos_i, pos_j, -1, -1] = W[pos_i, pos_j, aa_idx_j, aa_idx_i]


    new_model = Potts(h=torch.tensor(codon_h), W=torch.tensor(codon_W ).transpose(2,1).reshape((model.L+1)*65, (model.L+1)*65))
    return new_model


def amino_potts_to_codon_potts(model, codon_table):
    codons, aas = map(list, zip(*codon_table.items()))
    codon_to_aa = {c:a for c, a in zip(codons, aas)}
    codon_to_i = { c:i for i,c in enumerate(codons + ['XXX']) }

    h = model.h.reshape(
            (model.L, model.A)).detach().cpu().numpy()
    W = model.reshape_to_L_L_A_A().detach().cpu().numpy()


    codon_h = np.zeros((model.L, 65))
    codon_W = np.zeros((model.L, model.L, 65, 65))
    for pos in range(model.L):
        for codon_idx, codon in enumerate(codons):
            aa = codon_to_aa[codon]
            if aa == "*":
                codon_h[pos, codon_idx] = 10000
            else:
                codon_h[pos, codon_idx] = h[pos, aa_to_i[aa]]
        codon_h[pos, -1] = h[pos, aa_to_i['-']]


    for pos_i in tqdm(range(model.L-1)):
        for pos_j in range(pos_i+1, model.L):
            for codon_idx_i, codon_i in enumerate(codons):
                for codon_idx_j, codon_j in enumerate(codons):
                    aa_i = codon_to_aa[codon_i]
                    aa_j = codon_to_aa[codon_j]

                    if aa_i == "*" or aa_j == "*":
                        continue
                    aa_idx_i = aa_to_i[aa_i]
                    aa_idx_j = aa_to_i[aa_j]
                    codon_W[pos_i, pos_j, codon_idx_i, codon_idx_j] = W[pos_i, pos_j, aa_idx_i, aa_idx_j]
                    codon_W[pos_j, pos_i, codon_idx_j, codon_idx_i] = W[pos_j, pos_i, aa_idx_j, aa_idx_i]
            for codon_idx_i, codon_i in enumerate(codons):
                aa_i = codon_to_aa[codon_i]
                if aa_i == "*":
                    continue
                aa_idx_i = aa_to_i[aa_i]
                aa_idx_j = aa_to_i['-']
                codon_W[pos_i, pos_j, codon_idx_i, -1] = W[pos_i, pos_j, aa_idx_i, aa_idx_j]
                codon_W[pos_i, pos_j, -1, codon_idx_i] = W[pos_i, pos_j, aa_idx_j, aa_idx_i]

                codon_W[pos_j, pos_i, -1, codon_idx_i] = W[pos_j, pos_i, aa_idx_j, aa_idx_i]
                codon_W[pos_j, pos_i, codon_idx_i, -1] = W[pos_j, pos_i, aa_idx_i, aa_idx_j]

            aa_idx_i = aa_to_i['-']
            aa_idx_j = aa_to_i['-']
            codon_W[pos_j, pos_i, -1, -1] = W[pos_j, pos_i, aa_idx_j, aa_idx_i]
            codon_W[pos_i, pos_j, -1, -1] = W[pos_i, pos_j, aa_idx_j, aa_idx_i]


    new_model = Potts(h=torch.tensor(codon_h), W=torch.tensor(codon_W).transpose(2,1).reshape((model.L)*65, (model.L)*65))
    return new_model


def get_entangled_idxs(ent_nt_seq, large_codons_with_gaps, small_codons_with_gaps, frameshift=1, debug=False):
    if frameshift < 0:
        raise ValueError("Negative frameshift not implemented")
    smaller_gene = ''.join([nt for nt in ent_nt_seq if nt.isupper()])[frameshift:]
    smaller_gene_as_codons = [smaller_gene[i*3:(i+1)*3] for i in range(len(smaller_gene)//3)]
    ent_nt_seq_as_codons = [ent_nt_seq[i*3:(i+1)*3] for i in range(len(ent_nt_seq)// 3)]
    large_gene_idx = 0
    small_gene_idx = 0
    ent_nt_seq_as_codons_idx = 0
    ent_nt_seq_as_codons_small_idx = 0
    entangled_positions = []
    for large_gene_idx in range(len(large_codons_with_gaps)):
        large_gene_codon = large_codons_with_gaps[large_gene_idx]
        ent_nt_seq_codon = ent_nt_seq_as_codons[ent_nt_seq_as_codons_idx]
        if ent_nt_seq_codon.islower():
            if large_gene_codon == "XXX":
                continue
            ent_nt_seq_as_codons_idx += 1
            assert(ent_nt_seq_codon.upper() == large_gene_codon)
        else:
            if small_gene_idx >= len(small_codons_with_gaps):
                break
            small_gene_codon = small_codons_with_gaps[small_gene_idx]
            ent_nt_seq_small_codon = smaller_gene_as_codons[ent_nt_seq_as_codons_small_idx]
            assert(ent_nt_seq_codon[frameshift:] == ent_nt_seq_small_codon[:-frameshift])
            if large_gene_codon == "XXX":
                if debug:
                    print('large', large_gene_codon)
                if small_gene_codon == "XXX":
                    if debug:
                        print('small', small_gene_codon)
                    small_gene_idx += 1
                continue
            while small_gene_codon == "XXX":
                small_gene_idx += 1
                small_gene_codon = small_codons_with_gaps[small_gene_idx]
            assert(small_gene_codon == ent_nt_seq_small_codon)
            assert(large_gene_codon[frameshift:] == small_gene_codon[:-frameshift])
            entangled_positions.append((large_gene_idx, small_gene_idx))
            if debug:
                print(small_gene_codon, large_gene_codon)
            ent_nt_seq_as_codons_idx += 1        
            ent_nt_seq_as_codons_small_idx += 1
            small_gene_idx += 1
        if small_gene_idx >= len(small_codons_with_gaps):
            break
    return entangled_positions
