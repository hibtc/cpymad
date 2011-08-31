
import json

# some fixed conversions:
_VALUE_MAP = {'true' : True, 'false' : False, 'PLUS': 1, 'MINUS': -1}

def _convert_key(key):
    if key.startswith('@'):
        return key[1:]
    else:
        return key

def _try_float_convert(value):
    try:
        return float(value)
    except:
        return None

def _convert_value(value):
    if _VALUE_MAP.has_key(value):
        return _VALUE_MAP[value]
    
    newval = _try_float_convert(value)
    if not newval == None:
        return newval
    
    return value

def _convert_recursively(item):
    if  isinstance(item, list):
        convert2dict=True
        for nextItem in item:
            if not '@name' in nextItem:
                convert2dict=False
        if convert2dict:
            newdict = {}
            for nextItem in item:
                key=nextItem['@name']
                del nextItem['@name']
                newdict[key]=_convert_recursively(nextItem)
            return newdict
        else:
            newlist = []
            for nextItem in item:
                newlist.append(_convert_recursively(nextItem));
            return newlist
    elif isinstance(item, dict):
        newdict = {}    
        for key, value in item.items():
            thiskey=_convert_key(key)
            # we trick out the first key
            if thiskey=='jmad-model-definition':
                thiskey=value['@name']
                del value['@name']
            elif thiskey.split('-')[0]=='default':
                if '@ref-name' in value:
                    value=value['@ref-name']
            newdict[thiskey] = _convert_recursively(value)
        return newdict
    else:
        return _convert_value(item)

def _add_default_cpymads(new_dict):
    '''
     Add some default stuff needed in the cpymad
     definition.
    '''
    for mname,model in new_dict.items():
        model['dbdirs']=['/afs/cern.ch/eng/']
        for seqname,sequence in model['sequences'].items():
            sequence['aperfiles']=[]
        
def convert_dict(indict):
    '''
    converts a jmad-model definition to one which is more nicely readble from cpymad
    '''
    new_dict=_convert_recursively(indict)
    _add_default_cpymads(new_dict)
    return new_dict

def convert_file(infilename, outfilename):
    indict = json.loads(file(infilename, 'r').read())
    outdict = convert_dict(indict);
    file(outfilename, 'w').write(json.dumps(outdict, indent=2))

if __name__ == "__main__":
    print "Converting lhc"
    convert_file('lhc-1.jmd.json', 'lhc-1-out.jmd.json')
    
