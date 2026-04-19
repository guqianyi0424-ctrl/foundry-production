export interface ChainInfo {
  chain_id: string;
  sequence: string;
  length: number;
}

export interface HotspotResidue {
  chain: string;
  residue: number;
  score: number;
}

export interface RFD3Design {
  index: number;
  rank: number;
  plddt: number;
  pdb_path?: string;
  pdb_content?: string;
}

export interface MPNNResult {
  design_idx: number;
  sequence: string;
  score: number;
  seq_idx: number;
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

export interface DesignJob {
  id: string;
  target_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  rfd3_results?: {
    success: boolean;
    designs: RFD3Design[];
    mock?: boolean;
    error?: string;
  };
  mpnn_results?: MPNNResult[];
  rf3_results?: RF3Result[];
}
