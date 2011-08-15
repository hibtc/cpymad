import json

f=file('lhc.jmd.json','r')
jf=json.load(f)
f.close()

for model in jf:
    print jf[model].keys()
