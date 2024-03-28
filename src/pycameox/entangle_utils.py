from potts import Potts
from potts.potts import get_subset_potts_optimized
from.ilp import AA_TO_I as aa_to_i, CODON_TABLE as codon_table
import torch
import numpy as np
from itertools import product

tetra_nts = list(map(''.join, product('ACGT', repeat=4)))
i_to_tetra_nt = {i:tetra_nt for i,tetra_nt in enumerate(tetra_nts)}
tetra_nt_to_i = {tetra_nt:i for i,tetra_nt in i_to_tetra_nt.items()}

def to_model_without_gaps(model, aa_seq):
    large_non_gap_idxs = [i for i, c in enumerate(aa_seq) if c != '-']
    aa_seq_no_gaps = ''.join([aa_seq[i] for i in large_non_gap_idxs])
    L, A = model.L, model.A
    h = model.h.reshape(L, A)
    W = model.W.weight
    h_new, W_new = get_subset_potts_optimized(h, W, large_non_gap_idxs, [aa_to_i[aa] for aa in aa_seq])
    new_L = h_new.shape[0]
    h_new_no_gap = h_new[:, :20]
    W_new_no_gap = W_new.reshape(new_L, A, new_L, A)[:, :20, :, :20].reshape(new_L*20, new_L*20)
    model_no_gaps = Potts(h=h_new_no_gap, W=W_new_no_gap)
    return model_no_gaps


def get_insertions(nt_seq, aa_seq_no_gaps):
    codons_without_gaps = [nt_seq[i*3:(i+1)*3].upper() for i in range(len(nt_seq)//3)]
    insertions = []
    codon_idx = 0
    for i, aa in enumerate(aa_seq_no_gaps):
        codon = codons_without_gaps[codon_idx].upper()
        while codon_table[codon] != aa_seq_no_gaps[i]:
            # codon part of insertion
            insertions.append((codon_idx, codon))
            codon_idx += 1
            codon = codons_without_gaps[codon_idx].upper()
        codon_idx += 1
    return insertions


def from_no_gaps_to_with_insertions(model_no_gaps, nt_seq, aa_seq_no_gaps):
    codons_without_gaps = [nt_seq[i*3:(i+1)*3].upper() for i in range(len(nt_seq)//3)]
    insertions = []
    codon_idx = 0
    for i, aa in enumerate(aa_seq_no_gaps):
        codon = codons_without_gaps[codon_idx].upper()
        while codon_table[codon] != aa_seq_no_gaps[i]:
            # codon part of insertion
            insertions.append((codon_idx, codon))
            codon_idx += 1
            codon = codons_without_gaps[codon_idx].upper()
        codon_idx += 1

    L, A = model_no_gaps.L, model_no_gaps.A
    model_no_gaps_with_ins = Potts(h=model_no_gaps.h.reshape(L,A).detach().clone(),
                                                                        W=model_no_gaps.W.weight.detach().clone())
    for idx, codon in insertions:
        L, A = model_no_gaps_with_ins.L, model_no_gaps_with_ins.A
        h_c = model_no_gaps_with_ins.h.reshape(L,20).detach().clone()
        W_c = model_no_gaps_with_ins.W.weight.reshape(L,20, L, 20).transpose(1,2).detach().clone()
        new_h = torch.concat((h_c[:idx], torch.zeros(1, 20),  h_c[idx:]), dim=0,)
        new_W = torch.zeros(L+1, L+1, A, A)
        new_W[:idx, :idx] = W_c[:idx, :idx].clone()
        new_W[idx+1:, idx+1:] = W_c[idx:, idx:].clone()
        new_W[:idx, idx+1:] = W_c[:idx, idx:].clone()
        new_W[idx+1:, :idx] = W_c[idx:, :idx].clone()
        model_no_gaps_with_ins = Potts(h=new_h, W=new_W.transpose(1,2).reshape((L+1)*A, (L+1)*A))
    return model_no_gaps_with_ins
        



def aa_model_to_tetra(model, codon_slice):
    h = model.h.reshape(
            (model.L, model.A)).detach().cpu().numpy()
    W = model.reshape_to_L_L_A_A().detach().cpu().numpy()

    A = len(tetra_nts)
    L = model.L
    tetra_h = np.zeros((L, A))
    tetra_W = np.zeros((L, L, A, A))
    for pos in range(L):
        for tetra_nt in tetra_nts:
            tetra_idx = tetra_nt_to_i[tetra_nt]
            codon = tetra_nt[codon_slice]
            aa = codon_table[codon]
            if aa == "*":
                tetra_h[pos, tetra_idx] = 10000
            else:
                tetra_h[pos, tetra_idx] = h[pos, aa_to_i[aa]]



    pairwise_idxs_in = []
    pairwise_idxs_out = []
    for tetra_nt_i in tetra_nts:
        tetra_idx_i = tetra_nt_to_i[tetra_nt_i]
        codon_i = tetra_nt_i[codon_slice]
        aa_i = codon_table[codon_i]
        if aa_i == "*":
            continue
        for tetra_nt_j in tetra_nts:
            tetra_idx_j = tetra_nt_to_i[tetra_nt_j]
            codon_j = tetra_nt_j[codon_slice]
            aa_j = codon_table[codon_j]
            if aa_j == "*":
                continue
            aa_idx_i = aa_to_i[aa_i]
            aa_idx_j = aa_to_i[aa_j]
            pairwise_idxs_in.append((tetra_idx_i, tetra_idx_j))
            pairwise_idxs_out.append((aa_idx_i, aa_idx_j))
    ii, ij = map(np.asarray, zip(*pairwise_idxs_in))
    iio, ijo = map(np.asarray, zip(*pairwise_idxs_out))

    #for pos_i in tqdm(range(L-1)):
    for pos_i in range(L-1):
        posi = np.full(len(ii), pos_i)
        for pos_j in range(pos_i+1, L):
            posj = np.full(len(ij), pos_j)
            tetra_W[posi, posj, ii, ij] = \
                W[posi, posj, iio, ijo]
            tetra_W[posj, posi, ij, ii] = \
                W[posj, posi, ijo, iio]    

    # add tetra nt constraints
    for i in range(L-1):
        j = i+1
        for tetra_nt_i in tetra_nts:
            tetra_idx_i = tetra_nt_to_i[tetra_nt_i]
            for tetra_nt_j in tetra_nts:
                tetra_idx_j = tetra_nt_to_i[tetra_nt_j]
                if tetra_nt_i[-1] != tetra_nt_j[0]:
                    tetra_W[i,j,tetra_idx_i, tetra_idx_j] += 10000.0
                    tetra_W[j,i,tetra_idx_j, tetra_idx_i] += 10000.0

    model_tetra = Potts(h=torch.tensor(tetra_h), W=torch.tensor(tetra_W).transpose(2,1).reshape(L*A, L*A))
    return model_tetra


def add_stop_codon_to_tetra(model, codon_slice):
    L, A = model.L, model.A
    h = model.h.reshape(L, A)
    W = model.W.weight.reshape(L, A, L, A).transpose(1,2)
    new_h = torch.zeros(L+1, A)
    new_h[:L, :] = h

    new_W = torch.zeros(L+1, L+1, A, A)
    new_W[:L, :L, :, :] = W

    for tetra_nt in tetra_nts:
        tetra_idx = tetra_nt_to_i[tetra_nt]
        codon = tetra_nt[codon_slice]
        if codon_table[codon] != "*":
            new_h[L, tetra_idx] += 10000.
    model_with_stop = Potts(h=new_h, W=new_W.transpose(2,1).reshape((L+1)*A, (L+1)*A))
    return model_with_stop



def add_start_codon_to_tetra(model, codon_slice):
    L, A = model.L, model.A
    h = model.h.reshape(L, A).detach().clone()
    for tetra_nt in tetra_nts:
        tetra_idx = tetra_nt_to_i[tetra_nt]
        codon = tetra_nt[codon_slice]
        if codon_table[codon] != "M":
            h[0, tetra_idx] += 10000.
    model_with_start = Potts(h=h.detach().clone(), W=model.W.weight.detach().clone())
    return model_with_start


def entangle_potts(model1, model2, weight1, weight2):
    h1, W1 = model1.h.reshape(model1.L, model1.A).detach().clone(), model1.W.weight.detach().clone()
    h2, W2 = model2.h.reshape(model2.L, model2.A).detach().clone(), model2.W.weight.detach().clone()

    h_new = (weight1*h1)+(weight2*h2)
    W_new = (weight1*W1)+(weight2*W2)
    ent_potts = Potts(h=h_new, W=W_new)
    return ent_potts
