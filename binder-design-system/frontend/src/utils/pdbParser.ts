const THREE_TO_ONE: Record<string, string> = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C',
  GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P',
  SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  SEC: 'U', PYL: 'O', ASX: 'B', GLX: 'Z', XLE: 'J',
  MSE: 'M',
}

export interface ParsedChain {
  chain_id: string
  sequence: string
  length: number
  resSeqs: number[]
}

export function parsePdb(content: string): ParsedChain[] {
  const lines = content.split('\n')
  const chainMap = new Map<string, Map<number, string>>()

  for (const line of lines) {
    if (!line.startsWith('ATOM') && !line.startsWith('HETATM')) continue
    if (line.length < 26) continue

    const chainId = line.substring(21, 22).trim()
    const resSeq = parseInt(line.substring(22, 26).trim(), 10)
    const resName = line.substring(17, 20).trim()

    const oneLetter = THREE_TO_ONE[resName]
    if (!oneLetter) continue

    if (!chainMap.has(chainId)) {
      chainMap.set(chainId, new Map())
    }
    const residueMap = chainMap.get(chainId)!
    if (!residueMap.has(resSeq)) {
      residueMap.set(resSeq, oneLetter)
    }
  }

  const chains: ParsedChain[] = []
  for (const [chainId, residueMap] of chainMap) {
    const sortedResidues = [...residueMap.entries()].sort((a, b) => a[0] - b[0])
    const sequence = sortedResidues.map(([, aa]) => aa).join('')
    const resSeqs = sortedResidues.map(([seq]) => seq)
    chains.push({
      chain_id: chainId,
      sequence,
      length: sequence.length,
      resSeqs,
    })
  }

  return chains
}

export function parseCif(content: string): ParsedChain[] {
  const lines = content.split('\n')
  const atomLines: Array<{ chainId: string; resSeq: number; resName: string }> = []

  let inAtomBlock = false
  let labelAsymIdCol = -1
  let labelSeqIdCol = -1
  let labelCompIdCol = -1
  let groupPdbCol = -1

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (line.startsWith('_atom_site.')) {
      inAtomBlock = true
      continue
    }

    if (inAtomBlock && line.startsWith('loop_')) {
      continue
    }

    if (inAtomBlock && (line.startsWith('_') || line.startsWith('#') || line.trim() === '')) {
      if (labelAsymIdCol >= 0 && labelSeqIdCol >= 0 && labelCompIdCol >= 0) {
        break
      }
      inAtomBlock = false
      continue
    }

    if (inAtomBlock && labelAsymIdCol === -1) {
      const allHeaders = lines.slice(0, i).filter(l => l.startsWith('_atom_site.')).map(l => l.trim())
      labelAsymIdCol = allHeaders.findIndex(h => h === '_atom_site.label_asym_id' || h === '_atom_site.auth_asym_id')
      labelSeqIdCol = allHeaders.findIndex(h => h === '_atom_site.label_seq_id' || h === '_atom_site.auth_seq_id')
      labelCompIdCol = allHeaders.findIndex(h => h === '_atom_site.label_comp_id' || h === '_atom_site.auth_comp_id')
      groupPdbCol = allHeaders.findIndex(h => h === '_atom_site.group_PDB')

      if (labelAsymIdCol === -1 || labelSeqIdCol === -1 || labelCompIdCol === -1) {
        inAtomBlock = false
        continue
      }
    }

    if (inAtomBlock && labelAsymIdCol >= 0) {
      const cols = line.trim().split(/\s+/)
      if (cols.length > Math.max(labelAsymIdCol, labelSeqIdCol, labelCompIdCol)) {
        if (groupPdbCol >= 0 && cols[groupPdbCol] !== 'ATOM') continue

        const chainId = cols[labelAsymIdCol]
        const resSeq = parseInt(cols[labelSeqIdCol], 10)
        const resName = cols[labelCompIdCol]

        if (!isNaN(resSeq)) {
          atomLines.push({ chainId, resSeq, resName })
        }
      }
    }
  }

  const chainMap = new Map<string, Map<number, string>>()
  for (const { chainId, resSeq, resName } of atomLines) {
    const oneLetter = THREE_TO_ONE[resName]
    if (!oneLetter) continue

    if (!chainMap.has(chainId)) {
      chainMap.set(chainId, new Map())
    }
    const residueMap = chainMap.get(chainId)!
    if (!residueMap.has(resSeq)) {
      residueMap.set(resSeq, oneLetter)
    }
  }

  const chains: ParsedChain[] = []
  for (const [chainId, residueMap] of chainMap) {
    const sortedResidues = [...residueMap.entries()].sort((a, b) => a[0] - b[0])
    const sequence = sortedResidues.map(([, aa]) => aa).join('')
    const resSeqs = sortedResidues.map(([seq]) => seq)
    chains.push({
      chain_id: chainId,
      sequence,
      length: sequence.length,
      resSeqs,
    })
  }

  return chains
}

export function parseStructureFile(content: string, filename: string): ParsedChain[] {
  const ext = filename.toLowerCase()
  if (ext.endsWith('.cif')) {
    return parseCif(content)
  }
  return parsePdb(content)
}
