export interface ChainInfo {
  chain_id: string;
  sequence: string;
  length: number;
  resSeqs: number[];
}

export interface HotspotResidue {
  chain: string;
  residue: number;
  score: number;
}

export interface RFD3Design {
  index: number;
  batch: number;
  design_in_batch: number;
  name: string;
  pdb_path: string;
  pdb_content: string;
  plddt: number;
  mock?: boolean;
}

export interface MPNNResult {
  index: number;
  name: string;
  sequence: string;
  pdb_path: string;
  pdb_content: string;
  score: number;
  mock?: boolean;
}

export interface RF3Result {
  design_idx: number;
  sequence: string;
  rmsd: number;
  avg_plddt: number;
  passed: boolean;
  plddt?: number[];
  pae?: number[][];
  per_res_rmsd?: number[];
  mock?: boolean;
}
