def get_rel_sup(relfile, supfile):
  relevances = []
  supports = []
  fd = open(relfile,'r')
  for line in fd:
    relevance = line.strip()
    if relevance == "relevant":
      rel = 1
    else:
      rel = 0
    relevances.append(rel) 
  fd.close()

  fd = open(supfile,'r')
  for line in fd:
    support = line.strip()
    if support == "oppose":
      sup = 0
    elif support == "neutral":
      sup = 1
    else:
      sup = 2
    supports.append(sup) 
  fd.close()
  return (relevances, supports)
