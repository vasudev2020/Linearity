with open('../data/Idioms.tsv') as fp:
    data = fp.readlines()

PosSamples, NegSamples = [],[]
for line in data[1:]:
    fields = line.split('\t')
    sent=fields[0].strip()
    idiom = fields[1].strip()
    literal = fields[2].strip()
    # if idiom not in sent:   print("mismatch (I)<SEP>",sent,'<SEP>', idiom)
    # if literal not in sent:   print("mismatch (L)<SEP>",sent, '<SEP>',literal)
    # if literal not in sent or idiom not in sent:   print("mismatch <SEP>",sent, '<SEP>',idiom, '<SEP>',literal)

    assert idiom in sent and literal in sent

    ii = sent.index(idiom)
    li = sent.index(literal)
    if ii<li:
        if ii+len(idiom)>li:    print(sent,"<SEP>",idiom,"<SEP>",literal)
    else:
        if li+len(literal)>ii:  print(sent,"<SEP>",idiom,"<SEP>",literal)
    # if len(idiom.split())!=len(literal.split()):    print('len mismatch:',idiom,literal)
    # else:   
    #     PosSamples.append([sent,idiom])
    #     NegSamples.append([sent,literal])
