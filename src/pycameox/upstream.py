"""Python module for CAMEOX upstream pipeline"""

# ## Initialization
# ### Dependencies
from pathlib import Path, PosixPath
import sys
from typing import Set, List, Any, Dict, Optional

from pycameox.config import Id, Seq

from Bio import SeqIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

# Workaround for evcouplings if Python >= 3.10
MAJOR, MINOR, *_ = sys.version_info
EVC_WORKAROUND_NEEDED = (MAJOR == 3 and MINOR >= 10)
if EVC_WORKAROUND_NEEDED:
    import collections.abc
    collections.Iterable = collections.abc.Iterable
    collections.Mapping = collections.abc.Mapping
    collections.MutableSet = collections.abc.MutableSet
    collections.MutableMapping = collections.abc.MutableMapping
    print('NOTE: Using workaround to be able to work with Evcouplings')
from evcouplings.align.alignment import Alignment
from evcouplings.align.protocol import modify_alignment, cut_sequence


# ## Special functions

# ### reformat_MSA
def reformat_MSA(protein: str, srcfmt='a3m', tgtfmt='fasta') -> None:
    """Convert MSA format from source to target"""
    
    src_path = Path(protein, protein).with_suffix('.' + srcfmt)
    tgt_path = Path(protein, protein).with_suffix('.' + tgtfmt)

    with open(src_path) as src, open(tgt_path, 'w') as tgt:
        alig = Alignment.from_file(src, srcfmt)
        print(f'INFO: Input MSA has {alig.N} sequences and {alig.L} columns')
        alig.write(tgt, format=tgtfmt, width=sys.maxsize)  # No width limit (for fasta)
        print(f'Convert {src_path} --> {tgt_path} OK!')


# ### postMSA
def post_msa(
    protein: str, work_dir: PosixPath, hhfilter: PosixPath = '',
    fmt: str = 'stockholm', met_start: bool = True,
    seqid_filter: Optional[int] = None,
    min_seq_cov: int = 50, min_col_cov: int = 50,
    num_eff_seqs: bool = False, theta: float = 0.8
    ):
    """Postprocess (filter) HMMER alignments"""

    target_fname: PosixPath = Path(work_dir, f'{protein}.fa')
    msa_ext: str
    if fmt == 'stockholm':
        msa_ext = '.sto'
    elif fmt == 'fasta':
        msa_ext = '.fasta'
    elif fmt == 'a3m':
        msa_ext = '.a3m'
    else:
        raise ValueError(f'Unknown format "{fmt}"!')

    msa_fname: PosixPath = Path(work_dir, protein).with_suffix(msa_ext)
    output_prefix: PosixPath = Path(work_dir, protein)

    # Load target/wt sequence from fasta file (typically wt.fa)
    tgseq = SeqIO.read(target_fname, format='fasta')

    # Get raw alignment from stockholm file
    with open(msa_fname) as fsto:
        ali_raw = Alignment.from_file(fsto, fmt)
    # Force M always in the 1st column if present in the WT
    if met_start and tgseq.seq[0] == 'M':
        ali_raw = ali_raw.replace('-', 'M', columns=[0])
        print(f'postMSA WARNING: Replacing initial gap in alignment with Met')
    print(f'postMSA INFO: Input MSA has {ali_raw.N} seqs, {ali_raw.L} cols')

    # center alignment around focus/search sequence
    focus_cols = np.array([c != "-" for c in ali_raw[0]])
    focus_ali = ali_raw.select(columns=focus_cols)

    if fmt == 'stockholm':
        assert len(tgseq.seq) == len(focus_ali[0]), (
            f'{len(focus_cols)} focus cols, expected {len(tgseq.seq)}')
    else:
        if len(tgseq.seq) != len(focus_ali[0]):
            print(
                f'WARNING! {len(focus_cols)} focus cols, expected {len(tgseq.seq)}')

    TARGET_SEQ_INDEX = 0
    REGION_START = 0
    kwargs = {
        'prefix': str(output_prefix),
        'seqid_filter': seqid_filter,
        # 'seqid_filter' corresponds to "threshold" in run_hhfilter (default: 95): Sequence identity
        #  threshold for maximum pairwise identity (between 0 and 100)
        'hhfilter': str(hhfilter),
        # 'hhfilter' corresponds to "binary" in run_hhfilter: Path to hhfilter binary
        # Use integer in [0, 100] or real in [0.0, 1.0] (Chloe: 50; test: 98)
        'minimum_sequence_coverage': min_seq_cov,
        # Use integer in [0, 100] or real in [0.0, 1.0] (Chloe: 70; test: 0)
        'minimum_column_coverage': min_col_cov,
        # 'minimum_column_coverage' makes columns with too many gaps lowercase; 0 covers all columns.
        'compute_num_effective_seqs': num_eff_seqs,
        'theta': theta,  # value of theta in computing sequence weights
    }
    mod_outcfg, ali = modify_alignment(
        focus_ali, TARGET_SEQ_INDEX, tgseq.id, REGION_START, **kwargs)
    print(f'postMSA INFO: Output MSA has {ali.N} sequences, {ali.L} columns')


# ### reformat_msa
def reformat_msa(protein: str, work_dir: PosixPath, 
                 srcfmt='a3m', tgtfmt='fasta') -> None:
    """Convert MSA format from source to target"""
    
    src_path = Path(work_dir, protein).with_suffix('.' + srcfmt)
    tgt_path = Path(work_dir, protein).with_suffix('.' + tgtfmt)

    with open(src_path) as src, open(tgt_path, 'w') as tgt:
        alig = Alignment.from_file(src, srcfmt)
        print(f'INFO: Input MSA has {alig.N} sequences and {alig.L} columns')
        alig.write(tgt, format=tgtfmt, width=sys.maxsize)  # No width limit (for fasta)
        print(f'Reformat {src_path} as {tgt_path} OK!')
